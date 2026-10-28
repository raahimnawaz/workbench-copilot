"""Breadboard geometry: where every hole is, which holes connect, and which hole a point on the carrier is at.

A board is placed by its hole a1 on the carrier and two unit vectors: the way its row numbers grow and the
way its letters run from a to j (config/bench.yaml). In each row the a-e holes form one strip and f-j
another, and the power rails run the board's length (config/parts.yaml).

A board never lies exactly where its printed outline says, so register() finds it in the top-down view.
The holes are small dark squares: it shifts and turns the grid, by less than half a pitch, until the
predicted holes land on the darkest spots.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Hit:
    hole: str  # "c41", or a rail's name such as "a+"; prefixed with the board's name when there are several
    node: str  # the connected strip or rail: "41a-e" or "rail a+"
    distance_mm: float
    row: int | None  # None on a rail
    xy_mm: tuple  # the hole's center on the carrier; on a rail, the nearest point of the rail's line


@dataclass(frozen=True)
class Board:
    name: str
    a1: tuple  # hole a1's center on the carrier, mm
    u: tuple  # unit vector the way row numbers grow
    v: tuple  # unit vector from column a toward j
    spec: dict  # parts.yaml breadboard
    prefix: str = ""  # "name:" when the bench has several boards

    @property
    def pitch(self) -> float:
        return self.spec["pitch_mm"]

    def at(self, row: float, step: float) -> np.ndarray:
        """The carrier point at a row and a column step (pitches from column a); both may be fractional."""
        return (np.asarray(self.a1) + (row - 1) * self.pitch * np.asarray(self.u)
                + step * self.pitch * np.asarray(self.v))

    def coords(self, xy) -> tuple:
        """(row, step) of a carrier point: the inverse of at()."""
        d = np.asarray(xy, float) - np.asarray(self.a1)
        return float(d @ np.asarray(self.u)) / self.pitch + 1, float(d @ np.asarray(self.v)) / self.pitch

    def hole(self, name: str) -> np.ndarray:
        """The center of a hole such as "c41"."""
        name = name.removeprefix(self.prefix)
        return self.at(int(name[1:]), self.spec["columns"][name[0]])

    def holes(self) -> tuple:
        """Every terminal hole: (names, an N x 2 array of centers on the carrier)."""
        names = [f"{c}{r}" for r in range(1, self.spec["rows"] + 1) for c in self.spec["columns"]]
        return names, np.array([self.hole(n) for n in names])

    def center(self) -> np.ndarray:
        steps = list(self.spec["columns"].values())
        return self.at((self.spec["rows"] + 1) / 2, (min(steps) + max(steps)) / 2)

    def body(self) -> np.ndarray:
        """The corners of the plastic body on the carrier, assuming the holes sit centered in it."""
        length, width = self.spec["body_mm"]
        c, u, v = self.center(), np.asarray(self.u), np.asarray(self.v)
        return np.array([c + su * length / 2 * u + sv * width / 2 * v for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))])

    def strip(self, column: str) -> str:
        return next(s for s in self.spec["strips"] if column in s)

    def locate(self, xy) -> Hit:
        """The terminal hole or rail nearest a carrier point."""
        row_f, step_f = self.coords(xy)
        cols = self.spec["columns"]
        row = int(np.clip(round(row_f), 1, self.spec["rows"]))
        col = min(cols, key=lambda c: abs(cols[c] - step_f))
        center = self.at(row, cols[col])
        strip = self.strip(col)
        best = Hit(f"{self.prefix}{col}{row}", f"{self.prefix}{row}{strip[0]}-{strip[-1]}",
                   float(np.linalg.norm(np.asarray(xy) - center)), row, tuple(center))
        if 0.5 <= row_f <= self.spec["rows"] + 0.5:  # beside the rails' run
            for name, step in self.spec.get("rails", {}).items():
                distance = abs(step_f - step) * self.pitch
                if distance < best.distance_mm:
                    best = Hit(f"{self.prefix}{name}", f"{self.prefix}rail {name}", distance, None,
                               tuple(self.at(row_f, step)))
        return best

    def moved(self, shift_mm, turn_deg: float) -> "Board":
        """This board turned by turn_deg about its center, then shifted."""
        t = np.radians(turn_deg)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        c = self.center()
        a1 = c + R @ (np.asarray(self.a1) - c) + np.asarray(shift_mm)
        return Board(self.name, tuple(a1), tuple(R @ np.asarray(self.u)), tuple(R @ np.asarray(self.v)), self.spec,
                     self.prefix)


def boards_from(bench: dict, parts: dict) -> list:
    """The bench's boards from bench.yaml, named in hole names only when there are several."""
    entries = bench["boards"]
    several = len(entries) > 1
    out = []
    for name, b in entries.items():
        u, v = (np.asarray(b[k], float) / np.linalg.norm(b[k]) for k in ("rows_toward", "letters_toward"))
        if abs(float(u @ v)) > 1e-6:
            raise ValueError(f"board {name}: rows_toward and letters_toward must be at right angles")
        out.append(Board(name, tuple(b["a1_mm"]), tuple(u), tuple(v), parts["breadboard"], f"{name}:" if several else ""))
    return out


def region(boards: list, margin_mm: float) -> tuple:
    """The carrier rectangle (x0, y0, x1, y1) covering every board's body plus a margin."""
    corners = np.vstack([b.body() for b in boards])
    (x0, y0), (x1, y1) = corners.min(axis=0) - margin_mm, corners.max(axis=0) + margin_mm
    return float(x0), float(y0), float(x1), float(y1)


def _sample(score: np.ndarray, points_mm: np.ndarray, region_mm, px_per_mm: float) -> np.ndarray:
    """The score image at carrier points of any leading shape (..., 2), bilinear."""
    px = ((points_mm - np.asarray(region_mm[:2])) * px_per_mm).reshape(-1, 2).astype(np.float32)
    n, cols = len(px), 4096  # remap's maps must stay under 32767 columns
    grid = np.zeros((-(-n // cols) * cols, 2), np.float32)
    grid[:n] = px
    vals = cv2.remap(score, grid[:, 0].reshape(-1, cols), grid[:, 1].reshape(-1, cols), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return vals.reshape(-1)[:n].reshape(points_mm.shape[:-1])


def register(board: Board, view: np.ndarray, region_mm, px_per_mm: float, max_mm: float = 1.2,
             max_deg: float = 1.0) -> tuple:
    """Find the board in the top-down view: (the board where its holes are, a report).

    The search stays within max_mm and max_deg of where the board is supposed to be, under half a pitch,
    because a grid shifted by a whole pitch fits the holes as well as the right one does.
    """
    gray = view if view.ndim == 2 else cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    k = int(round(2.5 * px_per_mm)) | 1  # wider than a hole (about 1 mm) and narrower than the gaps between them
    score = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    score = np.minimum(score, np.percentile(score, 99)).astype(np.float32)  # wires and parts don't dominate
    _, holes = board.holes()
    center = board.center()

    def search(turns, shifts, around=(0.0, 0.0)) -> tuple:
        offsets = np.array([(dx, dy) for dy in shifts for dx in shifts]) + np.asarray(around)
        best, means = (-1.0, 0.0, (0.0, 0.0)), []
        for turn in turns:
            t = np.radians(turn)
            R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            turned = (holes - center) @ R.T + center
            m = _sample(score, turned[None, :, :] + offsets[:, None, :], region_mm, px_per_mm).mean(axis=1)
            means.append(m)
            i = int(np.argmax(m))
            if m[i] > best[0]:
                best = (float(m[i]), float(turn), tuple(offsets[i]))
        return best, np.concatenate(means)

    coarse = np.arange(-max_mm, max_mm + 1e-9, 0.1)
    (peak, turn, shift), means = search(np.arange(-max_deg, max_deg + 1e-9, 0.25), coarse)
    typical = float(np.median(means))
    (peak, turn, shift), _ = search(np.arange(turn - 0.25, turn + 0.25 + 1e-9, 0.05),
                                    np.arange(-0.12, 0.12 + 1e-9, 0.02), around=shift)
    contrast = peak / max(typical, 1e-6)
    report = {"board": board.name, "shift_mm": [round(float(shift[0]), 2), round(float(shift[1]), 2)],
              "turn_deg": round(float(turn), 2), "contrast": round(float(contrast), 2)}
    if contrast < 1.5:
        report["problem"] = (f"board {board.name}: couldn't find its hole grid (contrast {contrast:.2f}), so its "
                             f"printed position is used")
        return board, report
    if max(abs(shift[0]), abs(shift[1])) > max_mm - 0.15:
        report["problem"] = (f"board {board.name}: its holes fit best at the edge of the search, so it may sit a "
                             f"pitch off its outline")
    return board.moved(shift, turn), report
