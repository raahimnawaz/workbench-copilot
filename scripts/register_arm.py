"""Register the arm to the carrier by hand, with the motors off: touch known points, and check pointing before it moves.

point_at needs to know where the printed carrier lies in the arm's frame. touch asks you to put the tool point, the
middle of the closed jaws' tips, on known points one at a time, the tag centers by default, and reads the joints at
each. Forward kinematics gives the tool point in the arm's frame, and wbc/motion/carrier.py fits the turn about z and
the shift that take the carrier's mm there. It saves the fit to .wbc/registration.json when the touched points sit
within MAX_RMS (rms) and MAX_ERROR (worst) of it. Register again whenever the arm or the sheet moves.

verify is the touch test without power: touch holes (or tags) and it says how far the tool point is from each, as the
fit sees it, across the sheet and in height. It passes when the mean across the sheet is TOUCH_TEST or less. watch
keeps saying which hole the tool point is over. plan says, for one hole, the pose point_at would move to and how low
the two moves go, without moving anything: it works with the motors on or off.

Like check_joint_map.py, nothing is written to the servos: it only sends ping and read requests, through
wbc/motion/sts.py, never LeRobot. touch, verify and watch need the motors off, so the arm is limp: support it.

    python scripts/register_arm.py --port /dev/cu.usbmodem5B910297421 touch            # tag10 tag11 tag12 tag13
    python scripts/register_arm.py --port /dev/cu.usbmodem5B910297421 touch tag10 tag11 a1 j63 tag13
    python scripts/register_arm.py --port /dev/cu.usbmodem5B910297421 verify a5 a26 j1 j63
    python scripts/register_arm.py --port /dev/cu.usbmodem5B910297421 watch
    python scripts/register_arm.py --port /dev/cu.usbmodem5B910297421 plan a5

Points are tags (tag10-tag13, their centers on the sheet) or breadboard holes (a5, on the board's top, which
config/bench.yaml's carrier.board_top_mm puts above the sheet). At each, Enter reads it, s skips it and q stops. touch
and verify write .wbc/logs/register-<mode>-<time>.json. Close any other program using the port first.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_joint_map as cjm  # noqa: E402
from wbc.config import ROOT, load_yaml  # noqa: E402
from wbc.motion import sts  # noqa: E402
from wbc.motion.arm import FLOOR_JOINTS, JOINT_FLOOR, TOOL_FLOOR  # noqa: E402
from wbc.motion.carrier import (MIN_POINTS, Registration, aim, fit, point_mm,  # noqa: E402
                                registration_path)
from wbc.motion.joints import ARM, JOINTS, NAMES, JointMap  # noqa: E402
from wbc.motion.kinematics import load_chain, pose_to_task  # noqa: E402
from wbc.motion.trajectory import lowest  # noqa: E402

TAGS = ["tag10", "tag11", "tag12", "tag13"]  # touch's points when none are named: the sheet's four corners
HOLES = ["a1", "j1", "a63", "j63", "a5"]     # verify's: the board's four corners, and the Act 1 bug's hole
MAX_RMS = 2.0      # mm the touched points may sit from the fit, rms, for it to be saved
MAX_ERROR = 3.0    # mm the worst one may
TOUCH_TEST = 3.0   # mm the mean distance across the sheet may be for verify to pass: the plan's touch test
STEEP = 60.0       # degrees: a touch with the gripper pointing less steeply down than this gets a note
SCALE_NOTE = 0.015  # the touched points' spread may differ from the sheet's by this share before it's mentioned
LINE = 50          # points along each move's joint-space line that plan checks for how low it goes


def describe(name: str) -> str:
    if name.startswith("tag"):
        return f"tag {name[3:]}'s center, the middle of its black square, on the sheet"
    return f"hole {name}, on the breadboard's top"


def tool_point(chain, joint_map: JointMap, raw: dict) -> tuple[np.ndarray, float]:
    """Where the tool point is in the arm's frame, metres, and how far below level the gripper points, degrees."""
    T = chain.fk(joint_map.q(raw))
    return T[:3, 3], math.degrees(pose_to_task(T)[3])


def touch_each(bus, ids, joint_map, chain, bench, parts, names, ask, say) -> list:
    """Each named point touched in turn: [{name, carrier_mm, tool (m), pitch}], skipped ones left out. Stops early,
    keeping what's done, on q, Ctrl-C or a servo going silent."""
    touched = []
    try:
        for i, name in enumerate(names, 1):
            where = point_mm(bench, parts, name)
            say(f"\n{i}/{len(names)}: {describe(name)} (carrier x {where[0]:.1f}, y {where[1]:.1f} mm)")
            if cjm.choose(ask, "  Put the tool point on it, hold still, then press Enter (s = skip, q = quit): ") == "s":
                continue
            tip, pitch = tool_point(chain, joint_map, cjm.read_positions(bus, ids, cjm.READS))
            if pitch < STEEP:
                say(f"  The gripper points {pitch:.0f}° down. Nearer straight down is better: point_at aims that way, "
                    "and any error in where the tool point sits along the gripper then only shifts the height.")
            touched.append({"name": name, "carrier_mm": where, "tool": tip, "pitch": round(pitch, 1)})
    except (cjm.Stopped, KeyboardInterrupt, EOFError):
        say("\nStopped.")
    except sts.STSError as e:
        say(f"\nStopped: {e}.")
    return touched


def logged(touched: list) -> list:
    """touch_each's points, as the log keeps them."""
    return [{**t, "carrier_mm": [float(v) for v in t["carrier_mm"]], "tool": [float(v) for v in t["tool"]]}
            for t in touched]


def run_touch(bus, ids, joint_map, chain, bench, parts, names, record, path, ask=input, say=print) -> bool:
    """The guided registration. True if the fit was good enough to save, and saved."""
    say("Every motor is off, so the arm is limp: support it as you move it. Close the gripper: the tool point is the "
        "middle of the closed jaws' tips. Touch each point with the gripper pointing about straight down.")
    touched = touch_each(bus, ids, joint_map, chain, bench, parts, names, ask, say)
    record["touched"] = logged(touched)
    if len(touched) < MIN_POINTS:
        say(f"\n{len(touched)} point{'s' if len(touched) != 1 else ''} touched: a fit needs at least {MIN_POINTS}, and "
            "4 or more show how well it went. Nothing saved.")
        return False
    try:
        reg = fit([t["name"] for t in touched], [t["carrier_mm"] for t in touched], [t["tool"] for t in touched])
    except ValueError as e:
        say(f"\nNo fit: {e}. Nothing saved.")
        return False
    record["fit"] = {"yaw_deg": round(math.degrees(reg.yaw), 3), "shift_mm": [round(1000 * v, 2) for v in reg.shift],
                     "scale": round(reg.scale, 4), "rms_mm": round(reg.rms_mm, 2)}
    say("\nHow far each point sits from the fit:")
    for p in reg.points:
        say(f"  {p['name']:<8}{p['error_mm']:5.1f} mm")
    corner = reg.shift * 1000
    say(f"  rms {reg.rms_mm:.1f} mm. The sheet's x axis runs {math.degrees(reg.yaw):+.1f}° from the arm's straight "
        f"ahead, and its top-left corner is at x {corner[0]:.0f}, y {corner[1]:.0f}, z {corner[2]:+.0f} mm.")
    if abs(reg.scale - 1) > SCALE_NOTE:
        say(f"  The touched points spread {100 * (reg.scale - 1):+.1f}% against the sheet's distances. Check the sheet "
            "was printed at 100% (\"Actual size\"), and that each touch was on its mark.")
    worst = reg.worst
    if reg.rms_mm > MAX_RMS or worst["error_mm"] > MAX_ERROR:
        say(f"\nNot saved: that's over the {MAX_RMS:g} mm rms or {MAX_ERROR:g} mm worst a registration may have. "
            f"{worst['name']} is the furthest off: touch it again, or all of them.")
        return False
    if len(reg.points) == MIN_POINTS:
        say(f"  With only {MIN_POINTS} points the fit shows little of how well the touching went: verify will.")
    reg.save(path)
    say(f"\nSaved to {path}. Check it with verify before pointing, and register again if the arm or the sheet moves.")
    return True


def run_verify(bus, ids, joint_map, chain, bench, parts, names, record, path, ask=input, say=print) -> bool:
    """The touch test, motors off. True when the mean distance across the sheet is TOUCH_TEST or less."""
    reg = Registration.load(path)
    say("Every motor is off, so the arm is limp: support it as you move it. Touch each point with the closed jaws' "
        "tips, the gripper pointing about straight down.")
    touched = touch_each(bus, ids, joint_map, chain, bench, parts, names, ask, say)
    record["touched"] = logged(touched)
    results = []
    for t in touched:
        d = reg.to_carrier(t["tool"]) - t["carrier_mm"]
        results.append({"name": t["name"], "across_mm": round(float(math.hypot(d[0], d[1])), 2),
                        "dx_mm": round(float(d[0]), 2), "dy_mm": round(float(d[1]), 2), "dz_mm": round(float(d[2]), 2)})
    record["verify"] = results
    if not results:
        say("\nNothing touched, so nothing checked.")
        return False
    say("\nWhere the tool point was, against each point, as the registration sees it (carrier mm, y down the page):")
    for r in results:
        say(f"  {r['name']:<8}{r['across_mm']:5.1f} mm across (x {r['dx_mm']:+.1f}, y {r['dy_mm']:+.1f}), "
            f"{r['dz_mm']:+.1f} mm in height")
    mean = sum(r["across_mm"] for r in results) / len(results)
    heights = [r["dz_mm"] for r in results if not r["name"].startswith("tag")]
    ok = mean <= TOUCH_TEST
    say(f"  Mean {mean:.1f} mm across the sheet: " + ("within" if ok else "over") + f" the {TOUCH_TEST:g} mm touch test.")
    if heights:
        say(f"  On the board the tool point read {sum(heights) / len(heights):+.1f} mm from the board_top_mm height, on "
            "average: change carrier.board_top_mm in config/bench.yaml by that if it holds across holes.")
    return ok


def run_watch(bus, ids, joint_map, chain, bench, parts, path, say=print, sleep=time.sleep, clock=time.monotonic,
              seconds: float | None = None) -> bool:
    """The hole the tool point is over and how high above the board's top, about every cjm.EVERY seconds, until
    Ctrl-C (or for `seconds`)."""
    from wbc.perception.breadboard import boards_from

    reg = Registration.load(path)
    boards, top = boards_from(bench, parts), float(bench["carrier"]["board_top_mm"])
    say("Watching (read only), in carrier mm. Ctrl-C to stop.")
    t0 = clock()
    try:
        while seconds is None or clock() - t0 < seconds:
            tip, _ = tool_point(chain, joint_map, cjm.read_positions(bus, ids))
            x, y, z = reg.to_carrier(tip)
            hit = min((b.locate((x, y)) for b in boards), key=lambda h: h.distance_mm)
            say(f"\rx {x:6.1f}  y {y:6.1f}  z {z:6.1f} mm   over {hit.hole} ({hit.distance_mm:.1f} mm off), "
                f"{z - top:+.0f} mm from the board's top   ", end="", flush=True)
            sleep(cjm.EVERY)
    except KeyboardInterrupt:
        say("")
    return True


def run_plan(bus, ids, joint_map, chain, bench, parts, name, path, say=print) -> bool:
    """What point_at would do for one point, from where the arm is now, moving nothing. True if it can point there."""
    reg = Registration.load(path)
    raw = cjm.read_positions(bus, ids, cjm.READS)
    q0 = joint_map.q(raw)
    try:
        a = aim(chain, joint_map, reg.to_arm(point_mm(bench, parts, name)), q0)
    except ValueError as e:
        say(f"point_at {name}: {e}.")
        return False
    say(f"point_at {name}: the gripper {a.pitch:.0f}° down, its line {1000 * a.miss:.1f} mm from the spot.")
    for label, q in (("first, further back", a.q_above), ("then", a.q)):
        say(f"  {label}: " + ", ".join(f"{NAMES[m]} {math.degrees(v):+.1f}°" for m, v in zip(ARM, q)))
    for label, q1, q_from in (("The first move", a.q_above, q0), ("The second", a.q, a.q_above)):
        tool, axis = lowest(chain, np.linspace(q_from, q1, LINE), FLOOR_JOINTS)
        under = tool < TOOL_FLOOR or axis < JOINT_FLOOR
        say(f"  {label} takes the tool point down to {1000 * tool:.0f} mm above the table and the elbow and wrist to "
            f"{1000 * axis:.0f} mm" + (": under a floor, so move_to will refuse it from here. Start from a higher pose."
                                       if under else "."))
    past = joint_map.past_limits(raw)
    if past:
        say(f"  The {cjm.listed(past)} is past where moves may take it now, so the driver will refuse to start until "
            "it's moved back in.")
    return True


def run(bus, mode: str, robot_cfg: dict, bench: dict, parts: dict, names: list, record: dict, path, ask=input,
        say=print, sleep=time.sleep, clock=time.monotonic, seconds: float | None = None) -> bool:
    """One session on an open bus: all six servos answering, the motors off except for plan, the joint map from
    config/robot.yaml and the servos' own limits, then the mode."""
    ids = cjm.servo_ids(robot_cfg)
    silent = [m for m, i in ids.items() if not bus.ping(i)]
    if silent:
        say(f"No answer from the {cjm.listed(silent)}. Check that the servo power is on and the cable chain reaches "
            "every servo.")
        return False
    servos = cjm.read_servos(bus, ids)
    on = [m for m in JOINTS if servos[m]["torque"]]
    if on and mode != "plan":
        say(f"Torque is on for the {cjm.listed(on)}. This needs the motors off, so you can move the arm by hand. Switch "
            "the servo power off and on again (they come back with torque off), then run this again.")
        return False
    try:
        chain = load_chain(robot_cfg, ROOT)
        joint_map = JointMap.from_config(robot_cfg, servos, chain)
    except ValueError as e:
        say(f"The joint map can't be built: {e}.")
        return False
    try:
        if mode == "touch":
            return run_touch(bus, ids, joint_map, chain, bench, parts, names, record, path, ask=ask, say=say)
        if mode == "verify":
            return run_verify(bus, ids, joint_map, chain, bench, parts, names, record, path, ask=ask, say=say)
        if mode == "watch":
            return run_watch(bus, ids, joint_map, chain, bench, parts, path, say=say, sleep=sleep, clock=clock,
                             seconds=seconds)
        return run_plan(bus, ids, joint_map, chain, bench, parts, names[0], path, say=say)
    except FileNotFoundError as e:
        say(f"{str(e)[0].upper()}{str(e)[1:]}.")
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["touch", "verify", "watch", "plan"],
                    help="touch: register. verify: the touch test. watch: the hole under the tool point. plan: what "
                         "point_at would do")
    ap.add_argument("points", nargs="*", help=f"tags or holes: touch takes {' '.join(TAGS)} and verify "
                                              f"{' '.join(HOLES)} when none are named; plan takes one")
    ap.add_argument("--port", required=True, help="the arm's servo adapter, e.g. /dev/cu.usbmodem5B910297421")
    ap.add_argument("--file", type=Path, default=registration_path(ROOT), metavar="PATH",
                    help="where the registration is saved and read (default .wbc/registration.json)")
    args = ap.parse_args(argv)
    try:
        robot_cfg, bench, parts = load_yaml("robot.yaml"), load_yaml("bench.yaml"), load_yaml("parts.yaml")
        cjm.servo_ids(robot_cfg)
    except ValueError as e:
        print(f"config: {e}")
        return 1
    names = args.points or {"touch": TAGS, "verify": HOLES}.get(args.mode, [])
    if args.mode == "plan" and len(names) != 1:
        ap.error("plan takes one point, like a5")
    if args.mode == "watch" and names:
        ap.error("watch takes no points")
    for name in names:
        try:
            point_mm(bench, parts, name)
        except ValueError as e:
            ap.error(str(e))
    try:
        bus = sts.Bus(args.port)
    except Exception as e:  # noqa: BLE001
        print(f"Can't open {args.port}: {e}")
        return 1
    record = {"port": args.port, "mode": args.mode, "points": names}
    ok = False
    try:
        ok = run(bus, args.mode, robot_cfg, bench, parts, names, record, args.file)
    except sts.STSError as e:
        print(f"\nStopped: {e}")
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    finally:
        bus.close()
        if record.get("touched"):
            log = ROOT / ".wbc" / "logs" / f"register-{args.mode}-{datetime.now():%Y%m%d-%H%M%S}.json"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(json.dumps(record, indent=1), encoding="utf-8")
            print(f"log: {log}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
