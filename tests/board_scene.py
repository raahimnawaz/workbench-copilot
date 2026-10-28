"""A synthetic bench photo for the board reader's tests.

The printed carrier with its four tags, a breadboard with its holes, and flat jumpers from the display's
strips to the ESP32's, seen by a slightly tilted camera at about the C920's scale (0.22 mm per pixel), with
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

# The Act 1 board: (hole at the display's strip, hole at the ESP32's strip, BGR color). The CS wire lands on
# GPIO 13's strip (b22) while pins.h says 5.
WIRES = [
    ("e42", "g16", (0, 215, 255)),  # SCK: display D0 -> GPIO18 (j16)
    ("d43", "f10", (200, 90, 30)),  # MOSI: D1 -> GPIO23 (j10)
    ("e46", "a22", (0, 140, 255)),  # CS -> GPIO13 (b22), the planted bug
    ("b45", "h19", (60, 170, 40)),  # DC -> GPIO16 (j19)
    ("a44", "i18", (235, 235, 235)),  # RES -> GPIO17 (j18)
]


def bench_config() -> dict:
    bench = copy.deepcopy(load_yaml("bench.yaml"))
    bench["modules"] = {
        "esp32": {"part": "esp32_devkit_30",
                  "headers": {"left": {"start": "b10", "step": 1}, "right": {"start": "j10", "step": 1}}},
        "oled": {"part": "ssd1306_spi_7", "headers": {"pins": {"start": "c40", "step": 1}}},
    }
    return bench


def _poly(points_mm, scale, origin=(0.0, 0.0), center_offset=0.0) -> np.ndarray:
    """Polygon corners in 1/16 px fixed point, for cv2.fillConvexPoly(..., shift=4)."""
    px = (np.asarray(points_mm, float) - np.asarray(origin)) * scale - center_offset
    return np.round(px * 16).astype(np.int32)


def bar(board, a: str, b: str, width=WIRE_MM) -> np.ndarray:
    """A flat jumper's outline on the carrier: a bar from hole a's center to hole b's."""
    pa, pb = board.hole(a), board.hole(b)
    d = (pb - pa) / np.linalg.norm(pb - pa)
    n = np.array([-d[1], d[0]]) * width / 2
    return np.array([pa + n, pb + n, pb - n, pa - n])


def carrier_image(bench: dict, wires=WIRES, shift_mm=(0.0, 0.0), turn_deg=0.0) -> tuple:
    """(the carrier at S px/mm with the board and wires, the board where it actually lies)."""
    w_mm, h_mm = bench["carrier"]["paper_mm"]
    img = np.full((int(round(h_mm * S)), int(round(w_mm * S)), 3), 255, np.uint8)
    tags.draw(img, bench["carrier"], S)
    (board,) = breadboard.boards_from(bench, PARTS)
    board = board.moved(shift_mm, turn_deg)
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


def photo(shift_mm=(0.0, 0.0), turn_deg=0.0, wires=WIRES, seed=0) -> tuple:
    """(a 1920 x 1080 camera frame of the bench, the board where it actually lies, the bench config)."""
    bench = bench_config()
    img, board = carrier_image(bench, wires, shift_mm, turn_deg)
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
