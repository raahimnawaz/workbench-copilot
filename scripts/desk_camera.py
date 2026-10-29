"""The desk camera: print its tags, register them to the arm by touch, check the fit, and find things on the desk.

A second webcam looking down at the desk lets locate say where something is in the arm's frame
(wbc/perception/desk.py). Settings are in config/desk.yaml.

1. tags (free): writes .wbc/vision/desk-tags.pdf, six 50 mm tags (ids 20-25). Print it at 100% ("Actual size") and
   check the 100 mm bar. Cut the tags out along the gray lines, keeping their white border, and tape them flat on the
   desk, spread around the arm's reach where the desk camera sees them all and nothing will push them around.
       python scripts/desk_camera.py tags
2. touch (motors off, reads only): put the tool point, the middle of the closed jaws' tips, on each tag's center in
   turn with the gripper about straight down, and press Enter at each. Forward kinematics puts each center in the
   arm's frame, saved to .wbc/desk_tags.json. Touch them again whenever a tag moves.
       python scripts/desk_camera.py --port /dev/cu.usbmodem5B910297421 touch
3. check (free, from Terminal.app): finds the desk camera, maps a frame through the tags it shows, and says how far
   the other tags place each one from where it was touched, and where the camera is. The drawing in
   .wbc/vision/desk-check-<time>.png rings each touched center in a top-down view: the rings should sit on the tags.
       python scripts/desk_camera.py check
4. locate (PAID, $0.0025: one SAM 3.1 request, from Terminal.app): where a named thing is, in the arm's frame. Give
   the height of anything standing up, such as a cup (--height 95), because one camera can't see it.
       python scripts/desk_camera.py locate cup --height 95

check and locate take a still from the desk camera (saved as .wbc/vision/desk-photo-<time>.png), or --photo PATH to use
a saved one. locate saves SAM's reply as .wbc/vision/desk-<time>.json, and --replay PATH reads one again for free.
touch writes .wbc/logs/desk-touch-<time>.json. Close any other program using the arm's port before touch.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_joint_map as cjm  # noqa: E402
import make_carrier  # noqa: E402
from wbc import config  # noqa: E402
from wbc.config import ROOT  # noqa: E402
from wbc.motion import sts  # noqa: E402
from wbc.motion.joints import JOINTS, JointMap  # noqa: E402
from wbc.motion.kinematics import load_chain, pose_to_task  # noqa: E402
from wbc.perception import camera, desk, sam_client  # noqa: E402
from wbc.perception import tags as tagmap  # noqa: E402

OUT = ROOT / ".wbc" / "vision"
DPI = 300
LETTER_MM = (215.9, 279.4)  # portrait
SPOTS = [(55, 55), (160, 55), (55, 140), (160, 140), (55, 225), (160, 225)]  # tag centers on the sheet, mm
BORDER = 10.0   # mm of white kept around each tag when it's cut out: more than the one cell the detector needs
STEEP = 60.0    # degrees: a touch with the gripper pointing less steeply down than this gets a note
FLAT = 6.0      # mm the touched centers' heights may spread before it's mentioned: they should all be on the table
CHECK_MM = 5.0  # mm a tag may land from its touch, placed by the others, before check calls it out
REACH = 460.0   # mm: about how far the arm reaches at table height


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def shown(path: Path) -> str:
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def sheet(cfg: dict) -> np.ndarray:
    """The page of desk tags, at DPI: each with its id under it and a gray line to cut along."""
    s = DPI / 25.4
    size, ids = cfg["tags"]["size_mm"], cfg["tags"]["ids"]
    if len(ids) > len(SPOTS):
        raise ValueError(f"the sheet has room for {len(SPOTS)} tags, and config/desk.yaml lists {len(ids)}")
    img = np.full((int(round(LETTER_MM[1] * s)), int(round(LETTER_MM[0] * s)), 3), 255, np.uint8)
    spots = dict(zip(ids, SPOTS))
    tagmap.draw(img, {"tag_size_mm": size, "tag_family": cfg["tags"]["family"], "tags": spots}, s)
    half = size / 2 + BORDER
    for tag, (cx, cy) in spots.items():
        cv2.rectangle(img, (int((cx - half) * s), int((cy - half) * s)), (int((cx + half) * s), int((cy + half) * s)),
                      make_carrier.GRAY, max(1, int(0.2 * s)), cv2.LINE_AA)
        make_carrier.text(img, s, f"tag {tag}", (cx, cy + size / 2 + 6.5), 2.2, make_carrier.GRAY, centered=True)
    x0, y = 20.0, LETTER_MM[1] - 12
    cv2.rectangle(img, (int(x0 * s), int(y * s)), (int((x0 + 100) * s), int((y + 1.5) * s)), (0, 0, 0), -1)
    make_carrier.text(img, s, "this bar is 100 mm: print at 100% (Actual size)", (x0 + 104, y + 1.5), 2.5)
    make_carrier.text(img, s, f"Workbench Co-Pilot desk tags  {cfg['tags']['family']}  {size} mm  touch each center "
                              "with the arm", (20.0, 12.0), 2.5, make_carrier.GRAY)
    return img


def cmd_tags(args, cfg: dict) -> int:
    from PIL import Image

    img = sheet(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "desk-tags.png"), img)
    Image.fromarray(img[:, :, ::-1]).convert("RGB").save(OUT / "desk-tags.pdf", resolution=DPI)
    print(f"wrote {shown(OUT / 'desk-tags.pdf')}. Print it at 100%, check the 100 mm bar, cut the tags out along the gray "
          "lines and tape them flat around the arm's reach, all where the desk camera sees them.")
    return 0


def run_touch(bus, robot_cfg: dict, cfg: dict, ids_to_touch: list, path, record: dict, ask=input, say=print) -> bool:
    """The guided touch, on an open bus with the motors off. True if enough tags were touched, and saved."""
    ids = cjm.servo_ids(robot_cfg)
    silent = [m for m, i in ids.items() if not bus.ping(i)]
    if silent:
        say(f"No answer from the {cjm.listed(silent)}. Check that the servo power is on and the cable chain reaches "
            "every servo.")
        return False
    servos = cjm.read_servos(bus, ids)
    on = [m for m in JOINTS if servos[m]["torque"]]
    if on:
        say(f"Torque is on for the {cjm.listed(on)}. This needs the motors off, so you can move the arm by hand. Switch "
            "the servo power off and on again (they come back with torque off), then run this again.")
        return False
    try:
        chain = load_chain(robot_cfg, ROOT)
        joint_map = JointMap.from_config(robot_cfg, servos, chain)
    except ValueError as e:
        say(f"The joint map can't be built: {e}.")
        return False
    say("Every motor is off, so the arm is limp: support it as you move it. Close the gripper: the tool point is the "
        "middle of the closed jaws' tips. Touch each tag's center with the gripper pointing about straight down.")
    touched, pitches = {}, {}
    record["touches"] = []  # every touch in order, a tag touched twice included
    try:
        for n, tag in enumerate(ids_to_touch, 1):
            say(f"\n{n}/{len(ids_to_touch)}: tag {tag}'s center, the middle of its black square")
            if cjm.choose(ask, "  Put the tool point on it, hold still, then press Enter (s = skip, q = quit): ") == "s":
                continue
            T = chain.fk(joint_map.q(cjm.read_positions(bus, ids, cjm.READS)))
            tip, pitch = 1000 * T[:3, 3], math.degrees(pose_to_task(T)[3])
            if pitch < STEEP:
                say(f"  The gripper points {pitch:.0f}° down. Nearer straight down is better: any error in where the "
                    "tool point sits along the gripper then only shifts the height.")
            touched[tag], pitches[tag] = tip, round(pitch, 1)
            record["touches"].append({"tag": tag, "tool": [round(float(v), 2) for v in tip], "pitch": round(pitch, 1)})
            say(f"  x {tip[0]:.0f}, y {tip[1]:.0f}, z {tip[2]:+.0f} mm, the gripper {pitch:.0f}° down")
    except (cjm.Stopped, KeyboardInterrupt, EOFError):
        say("\nStopped.")
    except sts.STSError as e:
        say(f"\nStopped: {e}.")
    record["touched"] = {str(k): [round(float(v), 2) for v in p] for k, p in touched.items()}
    record["pitch"] = {str(k): p for k, p in pitches.items()}
    need = max(4, cfg["min_tags"])
    if len(touched) < need:
        say(f"\n{len(touched)} tag{'s' if len(touched) != 1 else ''} touched: a frame needs at least {need} in view, "
            "and 5 or more let check test each one against the rest. Nothing saved.")
        return False
    heights = [p[2] for p in touched.values()]
    if max(heights) - min(heights) > FLAT:
        say(f"\nThe touched centers' heights spread {max(heights) - min(heights):.0f} mm, and they should all be on the "
            "table. Check the tags lie flat, and touch any that read high or low again.")
    desk.save_touched(path, touched, {"pitch": {str(k): p for k, p in pitches.items()}})
    say(f"\nSaved {len(touched)} tags to {shown(path)}. Next, run check from Terminal.app: it says how well the camera's "
        "view agrees with these touches.")
    return True


def cmd_touch(args, cfg: dict) -> int:
    if not args.port:
        print("touch needs --port, the arm's servo adapter, e.g. /dev/cu.usbmodem5B910297421")
        return 1
    try:
        wanted = [int(t) for t in args.what] or list(cfg["tags"]["ids"])
    except ValueError:
        print(f"touch takes tag ids, such as {' '.join(map(str, cfg['tags']['ids']))}")
        return 1
    unknown = sorted(set(wanted) - set(cfg["tags"]["ids"]))
    if unknown:
        print(f"config/desk.yaml has no tag {', '.join(map(str, unknown))}: its tags are {cfg['tags']['ids']}")
        return 1
    try:
        robot_cfg = config.load_yaml("robot.yaml")
        cjm.servo_ids(robot_cfg)
    except ValueError as e:
        print(f"config/robot.yaml: {e}")
        return 1
    try:
        bus = sts.Bus(args.port)
    except Exception as e:  # noqa: BLE001
        print(f"Can't open {args.port}: {e}")
        return 1
    record = {"port": args.port, "tags": wanted}
    ok = False
    try:
        ok = run_touch(bus, robot_cfg, cfg, wanted, args.file, record)
    except sts.STSError as e:
        print(f"\nStopped: {e}")
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    finally:
        bus.close()
        if record.get("touched"):
            log = ROOT / ".wbc" / "logs" / f"desk-touch-{stamp()}.json"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(json.dumps(record, indent=1), encoding="utf-8")
            print(f"log: {log}")
    return 0 if ok else 1


def frame_from(args, cfg: dict, touched: dict):
    """(camera index or None, BGR frame): --photo, --camera, or the camera that shows the desk tags."""
    if args.photo:
        image = cv2.imread(args.photo)
        if image is None:
            raise RuntimeError(f"couldn't read {args.photo}")
        return None, image
    cam = cfg["camera"]
    if args.camera is not None:
        index, image = args.camera, camera.capture(args.camera, cam["width"], cam["height"])
    else:
        index, image = desk.find_camera(cfg, touched, camera.capture)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"desk-photo-{stamp()}.png"
    cv2.imwrite(str(path), image)
    print(f"camera {index}, photo: {shown(path)}")
    return index, image


def map_frame(frame: np.ndarray, cfg: dict, touched: dict) -> desk.DeskFit:
    size = (frame.shape[1], frame.shape[0])
    return desk.fit(tagmap.detect(frame, cfg["tags"]["family"]), touched, size, desk.focal_for(cfg, size[0]),
                    cfg["min_tags"])


def cmd_check(args, cfg: dict) -> int:
    touched = desk.load_touched(args.file)
    _, frame = frame_from(args, cfg, touched)
    fit = map_frame(frame, cfg, touched)
    size = (frame.shape[1], frame.shape[0])
    print(f"tags: saw {', '.join(map(str, fit.tags))}" + (f"; missing {', '.join(map(str, fit.missing))}" if fit.missing
                                                         else ""))
    if fit.errors:
        print("How far the other tags place each one from where it was touched:")
        for tag, mm in sorted(fit.errors.items()):
            print(f"  tag {tag}: {mm:4.1f} mm")
        print(f"  rms {fit.rms_mm:.1f} mm.")
        if fit.rms_mm > CHECK_MM:
            bad = fit.suspect
            if bad is not None:
                print(f"! Tag {bad} is the one that's off: without it the rest agree to {fit.rest[bad]:.1f} mm. A bad tag "
                      f"pulls where the others land too. Touch tag {bad} again (touch {bad}), or check it hasn't moved.")
            else:
                print(f"! That's over {CHECK_MM:g} mm" + ("" if fit.rest else ", and 6 tags in view would show which "
                                                          "one is off") + ". Check the tags lie flat and haven't moved, "
                      "and touch them all again.")
    else:
        print("With 4 tags in view the fit passes through all of them, so it can't check any: 5 can, and 6 can say "
              "which one is off.")
    C = fit.camera
    print(f"The camera is {C[2]:.0f} mm above the table, over x {C[0]:.0f}, y {C[1]:.0f} mm in the arm's frame, looking "
          f"{desk.tilt_deg(fit.H, size, fit.focal_px):.0f}° from straight down (focal length taken as "
          f"{fit.focal_px:.0f} px).")
    implied = desk.focal_from(fit.H, size)
    if implied:
        note = ("" if abs(implied / fit.focal_px - 1) < 0.1 else
                f" That's over 10% from config/desk.yaml's: set focal_px to {implied * desk.NOMINAL_WIDTH / size[0]:.0f}.")
        print(f"The tags put its focal length at {implied:.0f} px.{note}")
    view, to_arm = desk.top_down(frame, fit, cfg["region_mm"], cfg["px_per_mm"])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"desk-check-{stamp()}.png"
    cv2.imwrite(str(path), desk.draw(view, to_arm, touched))
    print(f"the top-down view, with a ring at each touched center: {shown(path)}")
    return 0


def describe(n: int, f: desk.Found) -> str:
    far = "  OUT OF REACH" if f.reach_mm > REACH else ""
    score = "" if f.score is None else f" ({f.score:.2f})"
    return (f"  {n}. {f.phrase}{score}: x {f.center[0]:.0f}, y {f.center[1]:.0f} mm, {f.reach_mm:.0f} mm from the "
            f"base; {f.length_mm:.0f} x {f.width_mm:.0f} mm, long axis {f.heading_deg:+.0f}°; top at z "
            f"{f.top_z:.0f} mm{far}")


def cmd_locate(args, cfg: dict) -> int:
    phrase = " ".join(args.what).strip()
    touched = desk.load_touched(args.file)
    if args.replay:
        saved = json.loads(Path(args.replay).read_text(encoding="utf-8"))
        phrase, height = saved["phrase"], saved["height_mm"]
        frame = cv2.imread(saved["photo"])
        if frame is None:
            raise RuntimeError(f"couldn't read the photo {saved['photo']} that {args.replay} was made from")

        def segment(image, p):
            return sam_client.Segmentation(p, *sam_client.parse(saved["raw"], p), saved["raw"])
    else:
        if not phrase:
            print("locate needs the thing to find, such as: locate cup --height 95")
            return 1
        from wbc.agent.budget import Budget

        height = args.height
        models = config.load_yaml("models.yaml")
        budget = Budget.from_config(models, ROOT)
        budget.check()
        _, frame = frame_from(args, cfg, touched)
        client = config.meta_client(models)

        def segment(image, p):
            print(f"1 request to {models['eyes']['model']}, about $0.0025 (spent so far ${budget.spent:.2f} of "
                  f"${budget.limit:.0f})")
            return sam_client.segment(client, models["eyes"], budget, image, p)
    fit = map_frame(frame, cfg, touched)
    hits, view, seg = desk.locate(frame, fit, cfg, segment, phrase, height)
    when = stamp()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.replay:
        photo = OUT / f"desk-{when}-photo.png"
        cv2.imwrite(str(photo), frame)
        (OUT / f"desk-{when}.json").write_text(json.dumps({"phrase": phrase, "height_mm": height, "photo": str(photo),
                                                           "raw": seg.raw}, indent=1), encoding="utf-8")
    view_path = OUT / f"desk-{when}.png"
    cv2.imwrite(str(view_path), desk.draw(view, np.linalg.inv(desk.view_matrix(cfg["region_mm"], cfg["px_per_mm"])),
                                          touched, hits))
    if not hits:
        print(f"SAM found no {phrase} on the table in view.")
    for n, f in enumerate(hits, 1):
        print(describe(n, f))
    for p in seg.problems:
        print(f"  ! {p}")
    if height == 0 and hits:
        print("Taken as lying flat. For something standing up, give its height (--height), or its middle reads "
              "further from the camera than it is.")
    print(f"drawn: {shown(view_path)}" + ("" if args.replay else f"; reply saved as {shown(OUT / f'desk-{when}.json')}"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["tags", "touch", "check", "locate"])
    ap.add_argument("what", nargs="*", help="touch: tag ids (all of config/desk.yaml's when none); locate: the thing to "
                                            "find, such as cup")
    ap.add_argument("--port", help="touch: the arm's servo adapter, e.g. /dev/cu.usbmodem5B910297421")
    ap.add_argument("--height", type=float, default=0.0, help="locate: how tall the thing stands, mm (0: lying flat)")
    ap.add_argument("--photo", help="check, locate: use this saved photo instead of taking one")
    ap.add_argument("--camera", type=int, help="check, locate: this camera index, instead of finding the desk camera")
    ap.add_argument("--replay", type=Path, metavar="PATH", help="locate: read a saved desk-<time>.json again, free")
    ap.add_argument("--file", type=Path, default=ROOT / desk.TOUCHED, metavar="PATH",
                    help="where touch saves the tags' centers and the others read them (default .wbc/desk_tags.json)")
    args = ap.parse_args(argv)
    cfg = config.load_yaml("desk.yaml")
    try:
        return {"tags": cmd_tags, "touch": cmd_touch, "check": cmd_check, "locate": cmd_locate}[args.mode](args, cfg)
    except (RuntimeError, ValueError, FileNotFoundError) as e:  # no camera, too few tags in view, nothing touched yet
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
