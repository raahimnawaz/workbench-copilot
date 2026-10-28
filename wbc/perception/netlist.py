"""From wire masks to a netlist, and the netlist against pins.h.

A wire's ends: thin its mask to a center line, take the longest path along it, and follow each end of the
path out to the tip of the mask. A flat jumper's legs go down into its holes at its tips, so each tip snaps
to the nearest hole; reader.tip_inset_mm corrects for a mask that stops short of the hole or runs past it.
Loose jumper leads stand up off the board, and the camera sees their tips displaced, so the reader assumes
flat ones.

The netlist: each strip and rail is a node, wires join nodes, and module pins sit in holes (bench.yaml).
Plain code then compares the nets with pins.h.
"""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_DEFINE = re.compile(r"^[ \t]*#define[ \t]+(\w+)[ \t]+(\d+)\b", re.M)


@dataclass
class End:
    xy_mm: tuple  # where the wire's leg goes into the board, on the carrier
    direction: tuple  # unit vector pointing out of the wire at this end
    hole: str | None = None
    node: str | None = None
    residual_mm: float | None = None  # from the end to its hole's center
    snapped: bool = False


@dataclass
class Wire:
    id: str
    ends: list
    length_mm: float
    width_mm: float
    color: str
    score: float | None
    problems: list = field(default_factory=list)
    box: tuple = field(default=(0, 0, 0, 0), repr=False)  # SAM's box and mask, in the view's pixels, to draw
    mask: np.ndarray | None = field(default=None, repr=False)


def thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning: a one-pixel-wide center line of a binary mask."""
    img = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p2, p3, p4, p5 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:], img[2:, 2:]
            p6, p7, p8, p9 = img[2:, 1:-1], img[2:, :-2], img[1:-1, :-2], img[:-2, :-2]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            count = sum(ring[:8])
            transitions = sum(((ring[k] == 0) & (ring[k + 1] == 1)).astype(np.uint8) for k in range(8))
            if step == 0:
                keep = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                keep = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (img[1:-1, 1:-1] == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & keep
            if remove.any():
                img[1:-1, 1:-1][remove] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)


def _farthest(pixels: set, start: tuple) -> tuple:
    """Breadth-first over 8-connected pixels: (the farthest pixel from start, each pixel's parent)."""
    parent, queue, last = {start: None}, deque([start]), start
    while queue:
        last = queue.popleft()
        for dy, dx in _NEIGHBORS:
            nxt = (last[0] + dy, last[1] + dx)
            if nxt in pixels and nxt not in parent:
                parent[nxt] = last
                queue.append(nxt)
    return last, parent


def center_line(mask: np.ndarray) -> list:
    """The longest path along the mask's thinned center line, as (row, col) pixels from one end to the other."""
    pixels = set(zip(*np.nonzero(thin(mask))))
    if not pixels:
        return []
    a, _ = _farthest(pixels, next(iter(pixels)))
    b, parent = _farthest(pixels, a)
    path = [b]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path


def largest_part(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask.astype(bool)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def _tip(mask: np.ndarray, start: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """March from start along direction to the mask's edge, in half-pixel steps; (x, y) in pixels."""
    point, h, w = start.astype(float), mask.shape[0], mask.shape[1]
    for _ in range(4 * (h + w)):
        nxt = point + 0.5 * direction
        x, y = int(round(nxt[0])), int(round(nxt[1]))
        if not (0 <= x < w and 0 <= y < h) or not mask[y, x]:
            return point + 0.25 * direction  # the edge lies between the last pixel in and the first out
        point = nxt
    return point


def color_name(bgr) -> str:
    """A rough name for a wire's color, for the HUD and for Spark to say."""
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    hue, sat, val = int(hsv[0]) * 2, int(hsv[1]), int(hsv[2])
    if val < 60:
        return "black"
    if sat < 50:
        return "white" if val > 170 else "gray"
    for name, top in (("red", 15), ("orange", 40), ("yellow", 70), ("green", 165), ("blue", 255), ("purple", 330)):
        if hue < top:
            return name
    return "red"


def trace(instances: list, view: np.ndarray, region_mm, px_per_mm: float, boards: list, cfg: dict) -> list:
    """Each wire mask from SAM (in the top-down view's pixels) as a Wire with both ends snapped to holes."""
    wires = []
    for n, inst in enumerate(sorted(instances, key=lambda i: (i.box[0], i.box[1])), start=1):
        left, top = inst.box[:2]
        mask = largest_part(inst.mask)
        problems = []
        path = center_line(mask)
        area_mm2 = mask.sum() / px_per_mm ** 2
        if len(path) < 3:
            wires.append(Wire(f"wire:{n}", [], 0.0, 0.0, "?", inst.score, ["too small to trace"], inst.box, mask))
            continue
        length_px = sum(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(path, path[1:]))
        width_mm = area_mm2 / max(length_px / px_per_mm, 1e-6)
        reach = min(len(path) - 1, max(2, int(round(3 * width_mm * px_per_mm))))
        ends = []
        for tip_i, back_i in ((0, reach), (-1, -1 - reach)):
            (ty, tx), (by, bx) = path[tip_i], path[back_i]
            d = np.array([tx - bx, ty - by], float)
            d /= max(np.linalg.norm(d), 1e-9)
            tip = _tip(mask, np.array([tx, ty], float), d)
            end_px = tip - cfg.get("tip_inset_mm", 0.0) * px_per_mm * d
            xy = np.array([left + end_px[0], top + end_px[1]]) / px_per_mm + np.asarray(region_mm[:2])
            ends.append(End(tuple(xy), tuple(d)))
        for end in ends:
            hit = min((b.locate(end.xy_mm) for b in boards), key=lambda h: h.distance_mm)
            end.hole, end.node, end.residual_mm = hit.hole, hit.node, round(hit.distance_mm, 2)
            end.snapped = hit.distance_mm <= cfg.get("snap_mm", 1.2)
            if not end.snapped:
                problems.append(f"an end is {hit.distance_mm:.1f} mm from the nearest hole ({hit.hole})")
        if ends[0].snapped and ends[1].snapped and ends[0].node == ends[1].node:
            problems.append(f"both ends are in {ends[0].node}")
        if _branches(mask, path, width_mm * px_per_mm):
            problems.append("its mask branches: it may be two wires that touch")
        ys, xs = np.nonzero(mask)
        bgr = np.median(view[top + ys, left + xs], axis=0) if len(ys) else (0, 0, 0)
        wires.append(Wire(f"wire:{n}", ends, round(length_px / px_per_mm, 1), round(width_mm, 2),
                          color_name([int(c) for c in bgr]), inst.score, problems, inst.box, mask))
    return wires


def _branches(mask: np.ndarray, path: list, width_px: float) -> bool:
    """Whether the center line has an arm off its main path longer than a few wire widths."""
    rest = thin(mask)
    for y, x in path:
        rest[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = False
    if not rest.any():
        return False
    n, _, stats, _ = cv2.connectedComponentsWithStats(rest.astype(np.uint8), connectivity=8)
    return bool(n > 1 and stats[1:, cv2.CC_STAT_AREA].max() > 3 * max(width_px, 2))


class Nets:
    """Union-find over node names."""

    def __init__(self):
        self.parent = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)


def module_pins(modules: dict, parts: dict, boards: list) -> dict:
    """{"esp32.GPIO13": its node, ...} from where bench.yaml says each header starts and which way it runs."""
    pins = {}
    for name, m in modules.items():
        part = parts["modules"][m["part"]]
        for header, place in m["headers"].items():
            board_name, _, start = place["start"].rpartition(":")
            board = next((b for b in boards if b.name == board_name), boards[0]) if board_name else boards[0]
            column, row = start[0], int(start[1:])
            for i, pin in enumerate(part[header]):
                hit = board.locate(board.at(row + i * place.get("step", 1), board.spec["columns"][column]))
                pins[f"{name}.{pin}"] = hit.node
    return pins


def read_defines(path) -> dict:
    """{macro: (number, line)} for every `#define NAME number` in a header."""
    text = open(path, encoding="utf-8").read()
    return {m.group(1): (int(m.group(2)), text.count("\n", 0, m.start()) + 1) for m in _DEFINE.finditer(text)}


def _row(hole: str) -> str | None:
    """"row:41" for a hole such as "c41" (or "main:c41"); None for a rail."""
    name = hole.rpartition(":")[2]
    return f"row:{name[1:]}" if name[1:].isdigit() else None


def check(wires: list, pins: dict, firmware: dict, defines: dict, file: str) -> tuple:
    """(netlist rows, mismatches): each signal's display pin, the GPIO its net reaches, and pins.h's number."""
    nets = Nets()
    traced = [w for w in wires if len(w.ends) == 2 and all(e.snapped for e in w.ends)]
    for w in traced:
        nets.join(w.ends[0].node, w.ends[1].node)
    gpio_prefix = f"{firmware['gpio_module']}.GPIO"
    rows, mismatches = [], []
    for signal, pin in firmware["signals"].items():
        if pin not in pins or signal not in defines:
            continue
        net = nets.find(pins[pin])
        gpio_nodes = {p: node for p, node in pins.items() if p.startswith(gpio_prefix) and nets.find(node) == net}
        gpios = sorted(int(p[len(gpio_prefix):]) for p in gpio_nodes)
        wire = next((w for w in traced if nets.find(w.ends[0].node) == net), None)
        wired_to = gpios[0] if len(gpios) == 1 else gpios or None
        rows.append({"signal": signal, "wire": wire.id if wire else None, "display_pin": pin.split(".", 1)[1],
                     "esp32_gpio": wired_to})
        value, line = defines[signal]
        if gpios == [value]:
            continue
        miss = {"signal": signal, "pins_h": value, "wired_to": wired_to, "wire": wire.id if wire else None,
                "file": file, "line": line,
                "problem": ("not wired to any GPIO" if not gpios else "wired to several GPIOs" if len(gpios) > 1
                            else "wired to a different GPIO")}
        if wire:  # point at the wire's far end: on the GPIO it reaches, or wherever it goes instead
            far = next((e for e in wire.ends if e.node in gpio_nodes.values()),
                       next((e for e in wire.ends if e.node != pins[pin]), wire.ends[0]))
            miss["hole"], miss["row"] = far.hole, _row(far.hole)
        mismatches.append(miss)
    return rows, mismatches
