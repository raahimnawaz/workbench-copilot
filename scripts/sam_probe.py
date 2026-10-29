"""See what SAM 3.1 finds on the bench: grab a still from the camera, segment it, and draw the masks.

Run from Terminal.app: macOS only lets a terminal it has asked about use the camera.

1. Find the camera (free). The C920 shows up by name; its OpenCV index is usually 0 on a Mac with no
   built-in camera, but an iPhone nearby can add a Continuity Camera.
       python scripts/sam_probe.py cameras
2. Grab a still (free). It goes to .wbc/vision/photo-<time>.png, lossless, at 1920x1080.
       python scripts/sam_probe.py capture --camera 0
3. Segment it (PAID, $0.0025 per phrase: one request per phrase). Each phrase is one short noun phrase.
   --crop X,Y,W,H sends only that part of the photo, to compare against the whole frame.
       python scripts/sam_probe.py segment .wbc/vision/photo-<time>.png --phrase "jumper wire"
4. Draw a saved run again (free): every reply is kept in its .json, so it parses again without paying.
       python scripts/sam_probe.py replay .wbc/vision/sam-<time>.json

segment and replay write the masks over the photo to .wbc/vision/sam-<time>.png and list each match: its
box, its area in pixels and SAM's confidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from wbc import config  # noqa: E402
from wbc.agent.budget import Budget  # noqa: E402
from wbc.perception import camera, sam_client  # noqa: E402

OUT = config.ROOT / ".wbc" / "vision"


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def shown(path: Path) -> str:
    """A path as typed from the repo root, when it's inside the repo."""
    try:
        return str(path.relative_to(config.ROOT))
    except ValueError:
        return str(path)


def cmd_cameras(args) -> int:
    names = camera.camera_names()
    print("macOS sees: " + (", ".join(names) if names else "no cameras (or system_profiler failed)"))
    found = camera.list_cameras()
    if not found:
        print(f"OpenCV opened none. {camera.NOT_ALLOWED}")
        return 1
    for c in found:
        size = "no frame" if c["size"] is None else f"{c['size'][0]}x{c['size'][1]}"
        print(f"  index {c['index']}: {size}")
    print("The C920 gives 1920x1080. If two cameras do, cover one lens and capture from each.")
    return 0


def cmd_capture(args) -> int:
    try:
        frame = camera.capture(args.camera, args.width, args.height)
    except RuntimeError as e:
        print(e)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"photo-{stamp()}.png"
    cv2.imwrite(str(path), frame)
    print(f"{frame.shape[1]}x{frame.shape[0]} still saved to {shown(path)}")
    return 0


def crop_box(text: str | None, image) -> tuple:
    if not text:
        return 0, 0, image.shape[1], image.shape[0]
    x, y, w, h = (int(v) for v in text.split(","))
    x, y = max(0, x), max(0, y)
    return x, y, min(w, image.shape[1] - x), min(h, image.shape[0] - y)


def report(results: dict, image, photo: str, crop: tuple, base: Path) -> None:
    """Print each phrase's matches, and save the overlay and a JSON record of the run."""
    drawn, summary = [], {}
    for phrase, seg in results.items():
        print(f'"{phrase}": {len(seg.instances)} match{"" if len(seg.instances) == 1 else "es"} '
              f"({seg.seconds:.1f} s)")
        for p in seg.problems:
            print(f"  ! {p}")
        if seg.instances:
            print("  id   box x, y, w, h        area px   score")
        for inst in sorted(seg.instances, key=lambda i: (i.box[1], i.box[0])):
            left, top, right, bottom = inst.box
            score = "  n/a" if inst.score is None else f"{inst.score:5.2f}"
            print(f"  {inst.object_id:<4} {left:>4}, {top:>4}, {right - left:>4}, {bottom - top:>4}  "
                  f"{inst.area:>9}   {score}")
        drawn += seg.instances
        summary[phrase] = {"seconds": round(seg.seconds, 2), "problems": seg.problems, "raw": seg.raw,
                           "matches": [{"id": i.object_id, "box": list(i.box), "area": i.area, "score": i.score}
                                       for i in seg.instances]}
    cv2.imwrite(str(base.with_suffix(".png")), sam_client.overlay(image, drawn))
    base.with_suffix(".json").write_text(json.dumps({"photo": photo, "crop": list(crop), "phrases": summary},
                                                    indent=1))
    print(f"masks drawn on the photo: {shown(base.with_suffix('.png'))}")


def cmd_segment(args) -> int:
    image = cv2.imread(args.photo)
    if image is None:
        print(f"couldn't read {args.photo}")
        return 1
    crop = crop_box(args.crop, image)
    x, y, w, h = crop
    image = image[y:y + h, x:x + w]
    models = config.load_yaml("models.yaml")
    eyes = models["eyes"]
    budget = Budget.from_config(models, config.ROOT)
    budget.check()
    price = models["budget"]["prices"][eyes["model"]]["per_1k_images"] / 1000
    print(f"{len(args.phrase)} request(s) to {eyes['model']} on a {w}x{h} image, about "
          f"${price * len(args.phrase):.4f} (spent so far ${budget.spent:.2f} of ${budget.limit:.0f})")
    results = sam_client.segment_many(config.meta_client(models), eyes, budget, image, args.phrase)
    OUT.mkdir(parents=True, exist_ok=True)
    report(results, image, args.photo, crop, OUT / f"sam-{stamp()}")
    return 0


def cmd_replay(args) -> int:
    run = json.loads(Path(args.run).read_text())
    image = cv2.imread(run["photo"])
    if image is None:
        print(f"couldn't read the photo {run['photo']}")
        return 1
    x, y, w, h = run["crop"]
    image = image[y:y + h, x:x + w]
    results = {}
    for phrase, saved in run["phrases"].items():
        instances, problems = sam_client.parse(saved["raw"], phrase)
        results[phrase] = sam_client.Segmentation(phrase, instances, problems, saved["raw"], saved["seconds"])
    report(results, image, run["photo"], tuple(run["crop"]), Path(args.run).with_name(Path(args.run).stem + "-replay"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cameras", help="list the cameras (free)")
    c = sub.add_parser("capture", help="save a still from the camera (free)")
    c.add_argument("--camera", type=int, default=0, help="OpenCV camera index (see: cameras)")
    c.add_argument("--width", type=int, default=1920)
    c.add_argument("--height", type=int, default=1080)
    s = sub.add_parser("segment", help="PAID: segment a photo, one request per phrase")
    s.add_argument("photo")
    s.add_argument("--phrase", action="append", required=True, help='a short noun phrase, e.g. "jumper wire"')
    s.add_argument("--crop", metavar="X,Y,W,H", help="send only this part of the photo, in pixels")
    r = sub.add_parser("replay", help="parse and draw a saved run again (free)")
    r.add_argument("run", help=".wbc/vision/sam-<time>.json")
    args = ap.parse_args(argv)
    return {"cameras": cmd_cameras, "capture": cmd_capture, "segment": cmd_segment, "replay": cmd_replay}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
