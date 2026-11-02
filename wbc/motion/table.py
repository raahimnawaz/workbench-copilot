"""The table under the arm, in the arm's frame: the height the driver's floors are measured from.

The desk isn't at the URDF's z 0. Near the base it's about 15 mm under it, where the base stands about 0.6 in tall, and
it reads lower the further out the arm reaches: touched with the arm, the desk's height falls about 9 mm for every
10 cm of reach (five desk tags, 95-399 mm from the pan axis, all within 1.3 mm of the line). That fall
is the arm's own reading, most likely its joints' zeros, and the arm moves by the same reading, so the floors follow
what the touches measured rather than a level table: a straight line in reach, fitted to the touched points
(Surface.fit). Reach is measured from the base's pan axis, 38.8 mm in front of the URDF's origin, so turning only the
base doesn't change it.

The floors use that line plus its worst miss (margin), and only trust it where the touches were. Nearer the base or
further out than any touch, the table is taken as the higher of the line there and the line at the nearest touched
end. Past the touched bearings it rises to the highest the line reaches over the touched reach, evenly over
BEARING_SLACK, so a fraction of a degree of the base's own error moves it a fraction of a millimetre, never a step.
So going where nothing was touched never lowers a floor. A level table is Surface.flat(z).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MIN_POINTS = 3      # touched points a fit needs
MIN_SPAN = 0.10     # metres of reach the touched points have to span between them
MAX_MISS = 0.008    # metres: a touch further than this off the line through the others is refused, it probably missed
MIN_MARGIN = 0.002  # metres the floors stay above the fitted line at least, whatever the fit's misses
MAX_SLOPE = 0.2     # metres of height per metre of reach (11°): a steeper desk is more likely a mistake
Z_RANGE = (-0.060, 0.010)  # metres: where the table may be, from the URDF's z 0, anywhere the line is used
FAR = 0.5           # metres: the reach checked against Z_RANGE when the line isn't held further in
BEARING_SLACK = math.radians(10)  # past the touched bearings, the turn over which the table rises to its held height
PAN_AXIS = (0.0388353, 0.0)  # metres: where the base's pan axis stands in the URDF's frame (the SO-101 URDF)


@dataclass(frozen=True)
class Surface:
    at_base: float                  # metres: the line's height at the pan axis
    slope: float                    # metres of height per metre of reach
    reach: tuple = (0.0, math.inf)  # metres: the nearest and furthest touched reach, where the line holds
    margin: float = 0.0             # metres the floors are measured from above the line: the fit's worst miss
    axis: tuple = (0.0, 0.0)        # metres: where reach is measured from, the base's pan axis
    bearing: tuple = (-math.pi, math.pi)  # radians: the touched bearings, from the arm's straight ahead toward its left

    def __post_init__(self):
        values = (self.at_base, self.slope, self.margin, *self.axis, *self.bearing, self.reach[0])
        if not all(math.isfinite(v) for v in values) or math.isnan(self.reach[1]):
            raise ValueError(f"the table can't be used: it has a value that isn't a number ({self}). Check "
                             "config/robot.yaml's table")
        problems = []
        far = self.reach[1] if math.isfinite(self.reach[1]) else FAR
        for where, r in (("at the base", 0.0), (f"{1000 * far:.0f} mm out", far)):
            z = self.line(r)
            if not Z_RANGE[0] <= z <= Z_RANGE[1]:
                problems.append(f"its height {where}, {1000 * z:.0f} mm, is outside {1000 * Z_RANGE[0]:.0f} to "
                                f"{1000 * Z_RANGE[1]:.0f} mm")
        if abs(self.slope) > MAX_SLOPE:
            problems.append(f"its slope, {1000 * self.slope:.0f} mm per m, is steeper than {1000 * MAX_SLOPE:.0f}")
        if not 0 <= self.margin <= MAX_MISS:
            problems.append(f"its margin, {1000 * self.margin:.1f} mm, is outside 0 to {1000 * MAX_MISS:.0f} mm")
        if not 0 <= self.reach[0] <= self.reach[1]:
            problems.append(f"its reach, {self.reach}, isn't nearest then furthest")
        if not -math.pi <= self.bearing[0] <= self.bearing[1] <= math.pi:
            problems.append(f"its bearings, {self.bearing}, aren't from -180° to 180°, least first")
        if problems:
            raise ValueError("the table can't be used: " + "; ".join(problems) + ". Check config/robot.yaml's table "
                             "(a mistake in units?)")

    @classmethod
    def flat(cls, z: float) -> Surface:
        """A level table at z metres."""
        return cls(z, 0.0)

    @classmethod
    def from_config(cls, robot_cfg: dict) -> Surface:
        """config/robot.yaml's table, or a level table_mm, or z 0."""
        t = robot_cfg.get("table")
        if t is None:
            return cls.flat(robot_cfg.get("table_mm", 0) / 1000)
        return cls(t["at_base_mm"] / 1000, t["slope_mm_per_m"] / 1000, tuple(r / 1000 for r in t["reach_mm"]),
                   t["margin_mm"] / 1000, tuple(v / 1000 for v in t["axis_mm"]),
                   tuple(math.radians(b) for b in t["bearing_deg"]))

    @classmethod
    def fit(cls, points, axis=PAN_AXIS) -> Surface:
        """The line in reach from `axis` through touched points on the table, (x, y, z) in metres in the arm's frame,
        by least squares, with its worst miss as the margin (MIN_MARGIN at least). ValueError when there are too few
        points, they span too little reach, or one is more than MAX_MISS off the line through the others (left out of
        the fit in turn, so a bad touch at an end of the reach can't pull the line onto itself)."""
        p = np.asarray(points, float).reshape(-1, 3)
        dx, dy = p[:, 0] - axis[0], p[:, 1] - axis[1]
        r, z = np.hypot(dx, dy), p[:, 2]
        if len(p) < MIN_POINTS or r.max() - r.min() < MIN_SPAN:
            raise ValueError(f"fitting the table takes {MIN_POINTS} touched points or more, spread over "
                             f"{1000 * MIN_SPAN:.0f} mm of reach or more")

        def line_through(keep):
            (a, b), *_ = np.linalg.lstsq(np.column_stack([np.ones(keep.sum()), r[keep]]), z[keep], rcond=None)
            return a, b

        if len(p) > MIN_POINTS:  # the touch whose leaving out puts the others nearest one line, if it's far off it
            best = None
            for i in range(len(p)):
                keep = np.arange(len(p)) != i
                if r[keep].max() - r[keep].min() >= MIN_SPAN:
                    a, b = line_through(keep)
                    rest = float(np.abs(z[keep] - (a + b * r[keep])).max())
                    if best is None or rest < best[0]:
                        best = (rest, abs(float(z[i] - (a + b * r[i]))), i)
            if best is not None and best[1] > MAX_MISS >= best[0]:
                raise ValueError(f"the touched point at {1000 * r[best[2]]:.0f} mm out is {1000 * best[1]:.1f} mm off "
                                 f"the line through the others, over {1000 * MAX_MISS:.0f} mm: touch it again")
        at_base, slope = line_through(np.ones(len(p), bool))
        miss = np.abs(z - (at_base + slope * r))
        if miss.max() > MAX_MISS:
            raise ValueError(f"the touches are up to {1000 * miss.max():.1f} mm off the line through them, over "
                             f"{1000 * MAX_MISS:.0f} mm: touch them again")
        bearings = np.arctan2(dy, dx)
        return cls(float(at_base), float(slope), (float(r.min()), float(r.max())),
                   max(float(miss.max()), MIN_MARGIN), (float(axis[0]), float(axis[1])),
                   (float(bearings.min()), float(bearings.max())))

    def line(self, r: float) -> float:
        return self.at_base + self.slope * r

    def desk(self, x: float, y: float) -> float:
        """The table's height under (x, y), metres, as the touches measured it: the line, held where they didn't
        reach."""
        dx, dy = x - self.axis[0], y - self.axis[1]
        r, bearing = math.hypot(dx, dy), math.atan2(dy, dx)
        if self.reach[0] <= r <= self.reach[1]:
            inside = self.line(r)
        else:
            inside = max(self.line(r), self.line(self.reach[0] if r < self.reach[0] else self.reach[1]))
        past = max(self.bearing[0] - bearing, bearing - self.bearing[1], 0.0)
        if past == 0.0:
            return inside
        ends = [self.line(self.reach[0])] + ([self.line(self.reach[1])] if math.isfinite(self.reach[1]) else [])
        highest = max([self.line(r)] + ends)  # the highest it reached
        return inside + min(1.0, past / BEARING_SLACK) * (highest - inside)

    def held(self, x: float, y: float) -> float:
        """How far the table under (x, y) is held over the line there, metres: 0 where the touches measured it."""
        return self.desk(x, y) - self.line(math.hypot(x - self.axis[0], y - self.axis[1]))

    def z(self, x: float, y: float) -> float:
        """What the floors are measured from under (x, y), metres: the desk plus the margin."""
        return self.desk(x, y) + self.margin

    def config(self) -> dict:
        """The table as config/robot.yaml's table takes it, rounded the safe way: the margin up, the reach and the
        bearings outward."""
        return {"at_base_mm": round(1000 * self.at_base, 1), "slope_mm_per_m": round(1000 * self.slope, 1),
                "reach_mm": [math.floor(1000 * self.reach[0]), math.ceil(1000 * self.reach[1])],
                "margin_mm": math.ceil(10000 * self.margin - 1e-6) / 10,
                "axis_mm": [round(1000 * v, 4) + 0.0 for v in self.axis],
                "bearing_deg": [math.floor(math.degrees(self.bearing[0])), math.ceil(math.degrees(self.bearing[1]))]}
