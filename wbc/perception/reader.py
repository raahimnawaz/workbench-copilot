"""read_breadboard: one camera frame in, the board's wires, netlist and mismatches against pins.h out.

1. Map the frame onto the printed carrier through its AprilTags (tags.py).
2. Cut out a top-down view of the boards at reader.px_per_mm, under SAM's 1008 px input, so thin wires
   keep their detail and SAM's masks come back in board millimeters.
3. SAM 3.1 segments the wires in that view while code finds each board's hole grid in it (breadboard.py).
4. Trace each wire to its two holes, join the nets, and compare them with pins.h (netlist.py).
5. Draw it all on the view, for the HUD and for Spark.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import breadboard, netlist, sam_client, tags

RED = (40, 40, 220)
GRID = (205, 205, 205)


def locate_boards(frame: np.ndarray, bench: dict, parts: dict) -> tuple:
    """(the carrier fit, the boards as configured, the view's carrier region, the top-down view)."""
    reader = bench["reader"]
    fit = tags.fit(tags.detect(frame, bench["carrier"]["tag_family"]), bench["carrier"])
    boards = breadboard.boards_from(bench, parts)
    area = breadboard.region(boards, reader.get("margin_mm", 5))
    view, _ = tags.rectify(frame, fit.H, area, reader["px_per_mm"])
    return fit, boards, area, view


def register_all(boards: list, view: np.ndarray, area, reader: dict) -> list:
    return [breadboard.register(b, view, area, reader["px_per_mm"], reader.get("register_mm", 1.2),
                                reader.get("register_deg", 1.0)) for b in boards]


def read_board(frame: np.ndarray, bench: dict, parts: dict, client, eyes: dict, budget, root: Path,
               out_dir: Path | None = None) -> dict:
    t0 = time.monotonic()
    reader = bench["reader"]
    s = reader["px_per_mm"]
    fit, boards, area, view = locate_boards(frame, bench, parts)
    timing = {"tags": time.monotonic() - t0}
    problems = [f"carrier tags {fit.missing} not seen, mapped from {fit.tags}"] if fit.missing else []
    with ThreadPoolExecutor(max_workers=1) as pool:  # SAM's round trip covers the grid search
        sam = pool.submit(sam_client.segment, client, eyes, budget, view, reader["wire_phrase"])
        registered = register_all(boards, view, area, reader)
        seg = sam.result()
    timing["sam"] = seg.seconds
    boards = [b for b, _ in registered]
    problems += [r["problem"] for _, r in registered if "problem" in r] + [f"SAM: {p}" for p in seg.problems]

    t1 = time.monotonic()
    wires = netlist.trace(seg.instances, view, area, s, boards, reader)
    pins = netlist.module_pins(bench.get("modules") or {}, parts, boards)
    fw = bench["firmware"]
    rows, mismatches = [], []
    if pins:
        rows, mismatches = netlist.check(wires, pins, fw, netlist.read_defines(root / fw["file"]), fw["file"])
    else:
        problems.append("bench.yaml places no modules yet, so the wiring can't be checked against pins.h")
    problems += [f"{w.id}: {p}" for w in wires for p in w.problems]
    timing["trace"] = time.monotonic() - t1

    residuals = [e.residual_mm for w in wires for e in w.ends if e.snapped]
    result = {
        "ok": True,
        "wires": [{"id": w.id, "color": w.color, "ends": [e.hole for e in w.ends],
                   "residual_mm": [e.residual_mm for e in w.ends]} for w in wires],
        "netlist": rows,
        "mismatches": mismatches,
        "endpoint_residual_mm": {"mean": round(float(np.mean(residuals)), 2) if residuals else None,
                                 "max": round(float(np.max(residuals)), 2) if residuals else None},
        "board": {"tags_rms_mm": round(fit.rms_mm, 2), "registration": [r for _, r in registered]},
        "problems": problems,
    }
    image = annotate(view, wires, mismatches, boards, area, s)
    if out_dir is not None:  # the view, the drawing and SAM's reply, so a read can be checked or replayed
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / f"board-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        cv2.imwrite(str(base.with_suffix(".png")), image)
        cv2.imwrite(str(base) + "-view.png", view)
        base.with_suffix(".json").write_text(json.dumps({**result, "sam_raw": seg.raw}, indent=1))
        result["_saved"] = str(base.with_suffix(".png"))
    result["_image"] = cv2.imencode(".png", image)[1].tobytes()  # the router shows this to Spark
    result["_timing"] = timing
    return result


def _px(xy_mm, area, s) -> tuple:
    return int(round((xy_mm[0] - area[0]) * s)), int(round((xy_mm[1] - area[1]) * s))


def _label(image, text, origin, color, scale=0.42) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_grid(image: np.ndarray, boards: list, area, s: float) -> np.ndarray:
    """Each board's predicted holes and rails, to check the grid sits on the real holes."""
    out = image.copy()
    for b in boards:
        _, holes = b.holes()
        for xy in holes:
            cv2.circle(out, _px(xy, area, s), max(1, int(0.35 * s)), (0, 160, 255), 1, cv2.LINE_AA)
        for name, step in b.spec.get("rails", {}).items():
            p0, p1 = _px(b.at(1, step), area, s), _px(b.at(b.spec["rows"], step), area, s)
            cv2.line(out, p0, p1, (60, 60, 230) if "+" in name else (200, 120, 40), 1, cv2.LINE_AA)
        for row in (1, 10, 20, 30, 40, 50, 60):
            _label(out, str(row), _px(b.at(row, -1.2), area, s), (0, 0, 0), 0.35)
        for col, step in b.spec["columns"].items():
            _label(out, col, _px(b.at(0.2, step), area, s), (0, 0, 0), 0.35)
    return out


def annotate(view: np.ndarray, wires: list, mismatches: list, boards: list, area, s: float) -> np.ndarray:
    """The top-down view with each wire outlined, its ends on their holes, and mismatches in red."""
    out = view.copy()
    for b in boards:  # the fitted hole grid, faintly
        for xy in b.holes()[1]:
            cv2.circle(out, _px(xy, area, s), 1, GRID, -1)
    wrong = {m["wire"]: m for m in mismatches if m.get("wire")}
    for i, w in enumerate(wires):
        if w.mask is None:
            continue
        miss = wrong.get(w.id)
        color = RED if miss else sam_client.PALETTE[i % len(sam_client.PALETTE)]
        contours, _ = cv2.findContours(w.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2 if miss else 1, offset=tuple(int(v) for v in w.box[:2]))
        for e in w.ends:
            p = _px(e.xy_mm, area, s)
            cv2.circle(out, p, max(3, int(0.8 * s)), color if e.snapped else RED, 1, cv2.LINE_AA)
            beyond = (np.asarray(e.xy_mm) + 3.0 * np.asarray(e.direction))
            _label(out, e.hole or "?", _px(beyond, area, s), color)
        if w.ends:
            mid = np.mean([e.xy_mm for e in w.ends], axis=0)
            text = w.id if not miss else (f"{miss['signal']}: GPIO {miss['wired_to']}, "
                                          f"pins.h says {miss['pins_h']}")
            _label(out, text, _px(mid, area, s), color, 0.45 if miss else 0.38)
    header = f"SAM 3.1: {len(wires)} wire{'s' if len(wires) != 1 else ''}, {len(mismatches)} mismatch" + \
             ("" if len(mismatches) == 1 else "es")
    _label(out, header, (6, 16), RED if mismatches else (0, 0, 0), 0.5)
    return out
