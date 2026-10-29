"""Read the breadboard the way the agent's read_breadboard does, from the camera or a saved photo.

Run from Terminal.app: macOS only lets a terminal it has asked use the camera.

1. check (free): finds the carrier's tags, cuts out the top-down view of the board and fits its hole grid.
   The drawing in .wbc/vision/check-<time>.png puts an orange ring on every hole the reader expects: they
   should sit on the real holes, with row 1 and column a where the labels say.
       python scripts/read_board.py check
2. read (PAID, $0.0025: one SAM 3.1 request): the whole read. SAM finds the jumper wires, each wire's ends
   snap to holes, and the nets are checked against pins.h once bench.yaml says where the modules sit.
       python scripts/read_board.py read

Both take a still from the camera in bench.yaml (saved as .wbc/vision/photo-<time>.png, for the golden set),
or --photo PATH to read a saved one instead.
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
from wbc.perception import camera, reader  # noqa: E402

OUT = config.ROOT / ".wbc" / "vision"


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def shown(path: Path) -> str:
    try:
        return str(path.relative_to(config.ROOT))
    except ValueError:
        return str(path)


def frame_from(args, bench: dict):
    if args.photo:
        image = cv2.imread(args.photo)
        if image is None:
            raise RuntimeError(f"couldn't read {args.photo}")
        return image
    cam = bench["camera"]
    image = camera.capture(cam["index"], cam["width"], cam["height"])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"photo-{stamp()}.png"
    cv2.imwrite(str(path), image)
    print(f"photo: {shown(path)}")
    return image


def cmd_check(args, bench: dict, parts: dict) -> int:
    image = frame_from(args, bench)
    fit, boards, area, view = reader.locate_boards(image, bench, parts)
    print(f"tags: saw {fit.tags}" + (f", missing {fit.missing}" if fit.missing else "") +
          f"; corners land {fit.rms_mm:.2f} mm (rms) from where they're printed")
    found = []
    for board, report in reader.register_all(boards, view, area, bench["reader"]):
        found.append(board)
        print(f"board {report['board']}: holes found {report['shift_mm'][0]:+.2f}, {report['shift_mm'][1]:+.2f} mm "
              f"and {report['turn_deg']:+.2f}° from its outline, looking {report.get('scale', 1):.3f}x life size "
              f"(its top stands above the sheet), contrast {report['contrast']:.1f}"
              + (f"\n  ! {report['problem']}" if "problem" in report else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"check-{stamp()}.png"
    cv2.imwrite(str(path), reader.draw_grid(view, found, area, bench["reader"]["px_per_mm"]))
    print(f"the fitted grid over the board: {shown(path)}")
    return 0


def cmd_read(args, bench: dict, parts: dict) -> int:
    models = config.load_yaml("models.yaml")
    budget = Budget.from_config(models, config.ROOT)
    budget.check()
    image = frame_from(args, bench)
    print(f"1 request to {models['eyes']['model']}, about $0.0025 (spent so far ${budget.spent:.2f} of "
          f"${budget.limit:.0f})")
    result = reader.read_board(image, bench, parts, config.meta_client(models), models["eyes"], budget,
                               config.ROOT, out_dir=OUT)
    for w in result["wires"]:
        print(f"  {w['id']:<8} {w['color']:<7} {' to '.join(e or '?' for e in w['ends']):<14} "
              f"off by {', '.join(f'{r:.2f}' for r in w['residual_mm'])} mm")
    for row in result["netlist"]:
        print(f"  {row['signal']:<10} {row['pin']:<8} -> GPIO {row['esp32_gpio']}  ({row['wire']})")
    for m in result["mismatches"]:
        print(f"  MISMATCH {m['signal']}: {m['problem']}, wired to {m['wired_to']}, pins.h line {m['line']} says "
              f"{m['pins_h']}" + (f" (at {m['hole']})" if m.get("hole") else ""))
    for p in result["problems"]:
        print(f"  ! {p}")
    timing = ", ".join(f"{k} {v:.1f} s" for k, v in result["_timing"].items())
    print(f"ends {json.dumps(result['endpoint_residual_mm'])} mm from their holes; {timing}")
    print(f"drawn: {shown(Path(result['_saved']))}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["check", "read"])
    ap.add_argument("--photo", help="read this saved photo instead of taking one")
    args = ap.parse_args(argv)
    bench, parts = config.load_yaml("bench.yaml"), config.load_yaml("parts.yaml")
    try:
        return {"check": cmd_check, "read": cmd_read}[args.cmd](args, bench, parts)
    except (RuntimeError, ValueError) as e:  # no camera, or not enough tags in view
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
