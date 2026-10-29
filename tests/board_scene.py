"""A synthetic bench photo for the board reader's tests.

The printed carrier with its four tags, a breadboard with its holes, and the Act 1 force sensor board's flat wires, seen by a slightly tilted camera at about the C920's scale (0.22 mm per pixel), with
blur and noise. The breadboard can sit off its configured spot, as a real one always does a little.
"""
import copy

import cv2
import numpy as np

from wbc.config import load_yaml
from wbc.perception import breadboard, tags

PARTS = load_yaml("parts.yaml")
S = 10  # px per mm of the rendered carrier; pixel i covers mm [i, i + 1) / S
WIRE_MM = 2.0

# The Act 1 board: (end, end, BGR color). An end is a hole such as "a5", or a rail and a row such as ("a-", 30).
# The force sensor's jumper leaves GPIO 4's strip (a5) while pins.h says 15.
WIRES = [
    ("a5", "a26", (0, 140, 255)),       # the sensor's jumper, from GPIO4 (b5): the planted bug
    ("b26", "b30", (235, 235, 235)),    # the 10 kOhm resistor, which SAM reads as a wire
    ("a27", ("a+", 27), (40, 40, 200)),  # the sensor's other pin to the 3.3 V rail
    ("a30", ("a-", 30), (30, 30, 30)),  # the resistor's far end to the ground rail
]


def end_name(end) -> str:
    """The name the reader gives an end: the hole, or the rail without its row."""
    return end if isinstance(end, str) else end[0]


def wire_ends(wires=WIRES) -> set:
    return {tuple(sorted((end_name(a), end_name(z)))) for a, z, _ in wires}


def bench_config() -> dict:
    bench = copy.deepcopy(load_yaml("bench.yaml"))  # the DevKit and the sensor where bench.yaml places them
    bench["reader"]["tip_inset_mm"] = 0.0  # these wires end right on their holes, unlike the real bench's
    return bench


def _poly(points_mm, scale, origin=(0.0, 0.0), center_offset=0.0) -> np.ndarray:
    """Polygon corners in 1/16 px fixed point, for cv2.fillConvexPoly(..., shift=4)."""
    px = (np.asarray(points_mm, float) - np.asarray(origin)) * scale - center_offset
    return np.round(px * 16).astype(np.int32)


def _point(board, end) -> np.ndarray:
    if isinstance(end, str):
        return board.hole(end)
    rail, row = end
    return board.at(row, board.spec["rails"][rail])


def bar(board, a, b, width=WIRE_MM) -> np.ndarray:
    """A flat jumper's outline on the carrier: a bar from one end's center to the other's."""
    pa, pb = _point(board, a), _point(board, b)
    d = (pb - pa) / np.linalg.norm(pb - pa)
    n = np.array([-d[1], d[0]]) * width * board.scale / 2
    return np.array([pa + n, pb + n, pb - n, pa - n])


def carrier_image(bench: dict, wires=WIRES, shift_mm=(0.0, 0.0), turn_deg=0.0, scale=1.0) -> tuple:
    """(the carrier at S px/mm with the board and wires, the board where it actually lies). scale > 1 draws the
    board larger than life, as the camera sees a board whose top stands above the sheet."""
    w_mm, h_mm = bench["carrier"]["paper_mm"]
    img = np.full((int(round(h_mm * S)), int(round(w_mm * S)), 3), 255, np.uint8)
    tags.draw(img, bench["carrier"], S)
    (board,) = breadboard.boards_from(bench, PARTS)
    board = board.moved(shift_mm, turn_deg, scale)
    cv2.fillConvexPoly(img, _poly(board.body(), S, center_offset=0.5), (226, 226, 226), cv2.LINE_AA, 4)
    u, v = np.asarray(board.u), np.asarray(board.v)
    for c in board.holes()[1]:  # holes about 1 mm square
        square = [c + a * 0.5 * u + b * 0.5 * v for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        cv2.fillConvexPoly(img, _poly(square, S, center_offset=0.5), (50, 50, 50), cv2.LINE_AA, 4)
    for name, step in board.spec["rails"].items():
        p0, p1 = _poly([board.at(1, step), board.at(board.spec["rows"], step)], S, center_offset=0.5)
        cv2.line(img, tuple(p0), tuple(p1), (40, 40, 200) if "+" in name else (190, 110, 40), 4, cv2.LINE_AA, 4)
    for a, b, color in wires:
        cv2.fillConvexPoly(img, _poly(bar(board, a, b), S, center_offset=0.5), color, cv2.LINE_AA, 4)
    return img, board


def camera_homography() -> np.ndarray:
    """Carrier mm -> camera pixels: 0.22 mm per pixel, turned 3 degrees, the camera slightly tilted."""
    t = np.radians(3)
    k = 1 / 0.22
    turn = np.array([[k * np.cos(t), -k * np.sin(t), 0], [k * np.sin(t), k * np.cos(t), 0], [0, 0, 1]])
    tilt = np.array([[1, 0, 0], [0, 1, 0], [2e-4, -1.5e-4, 1]])
    center = np.array([[1, 0, -139.7], [0, 1, -107.95], [0, 0, 1]])
    return np.array([[1, 0, 960], [0, 1, 540], [0, 0, 1]]) @ turn @ tilt @ center


def photo(shift_mm=(0.0, 0.0), turn_deg=0.0, wires=WIRES, seed=0, scale=1.0) -> tuple:
    """(a 1920 x 1080 camera frame of the bench, the board where it actually lies, the bench config)."""
    bench = bench_config()
    img, board = carrier_image(bench, wires, shift_mm, turn_deg, scale)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)  # the print's own softness, and no aliasing when the camera shrinks it
    px_to_mm = np.array([[1 / S, 0, 0.5 / S], [0, 1 / S, 0.5 / S], [0, 0, 1]])
    cam = cv2.warpPerspective(img, camera_homography() @ px_to_mm, (1920, 1080), flags=cv2.INTER_LINEAR,
                              borderValue=(95, 95, 95))
    cam = cv2.GaussianBlur(cam, (0, 0), 0.6)
    noise = np.random.default_rng(seed).normal(0, 3, cam.shape)
    return np.clip(cam + noise, 0, 255).astype(np.uint8), board, bench


def wire_masks(bench: dict, board, wires=WIRES, score=0.95) -> tuple:
    """Each wire as SAM would return it for the reader's top-down view: ([(box, mask, score)], (width, height))."""
    reader = bench["reader"]
    s = reader["px_per_mm"]
    area = breadboard.region(breadboard.boards_from(bench, PARTS), reader["margin_mm"])
    size = (int(round((area[2] - area[0]) * s)), int(round((area[3] - area[1]) * s)))
    out = []
    for a, b, _ in wires:
        full = np.zeros(size[::-1], np.uint8)
        cv2.fillConvexPoly(full, _poly(bar(board, a, b), s, origin=area[:2]), 1, cv2.LINE_8, 4)
        ys, xs = np.nonzero(full)
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        out.append((box, full[box[1]:box[3], box[0]:box[2]], score))
    return out, size
