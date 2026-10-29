"""Where the printed carrier lies in the arm's frame, and the arm pose that points the gripper at a spot on it.

Carrier mm (config/bench.yaml) run from the sheet's top-left corner, x to the right and y down the page, with z up from
the sheet. The arm's frame is the URDF's base frame: z up, with the base's underside, on the table, at z = 0.
Registration finds the turn about z and the shift that take one to the other, from points whose carrier mm are known
and that the tool point was put on by hand with the motors off: forward kinematics says where the tool point was in
the arm's frame. The table is level and the carrier lies flat on it, so the fit turns only about z: four numbers (the
turn, and a shift in x, y and z), which three points well apart already pin down. More points average out the
touching, and their residuals show how well it went. The turn is measured, not assumed: "facing forward" by eye read
the base 5° and 9° off the URDF's zero on Oct 28.

Pointing puts the tool point HOVER back from the spot along the line the gripper points along, steepest first: straight
down where the arm can, tipping toward the arm's own reach where it can't. point_at() gets there in two moves, first
to ABOVE further back along the same line, then in, so the last part is a short approach along the line. It only
calls the driver's move_to(), which keeps every refusal and cut it has, and it never powers the arm on or clears a
stop.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .arm import FLOOR_JOINTS, JOINT_FLOOR, RANGE_MARGIN, TOOL_FLOOR
from .joints import ARM, NAMES
from .kinematics import rot_x, rot_y, rot_z, solve_ik
from .trajectory import lowest

FLIP = np.diag([1.0, -1.0, 1.0])  # carrier mm have y down the page: flipping it makes a right-handed frame, z up
TAG = "tag"                # a tag's center is named tag<id>, e.g. tag10: it's printed on the sheet, so its z is 0
MIN_POINTS = 3             # touched points a fit needs
MIN_SPREAD = 0.02          # metres the points have to spread across their narrower direction: 3 in a line pin nothing
HOVER = 0.03               # metres from the spot back to the tool point, along the line the gripper points along
ABOVE = 0.06               # metres further back along that line that point_at() stops at first
PITCHES = range(90, 35, -5)  # degrees below level the gripper may point, tried steepest first
MISS = 0.001               # metres the gripper's line may pass from the spot
REACH = 0.47               # metres: nothing farther from the base's axis is worth solving for (about 0.46 at the table)
LINE_TRIES = 3             # solves per pose: each puts the tool point back on the line the last one's yaw gives
ROLL = ARM.index("wrist_roll")


def point_mm(bench: dict, parts: dict, name: str) -> np.ndarray:
    """Carrier mm (x, y, z) of a named point: a tag's center ("tag10"), on the sheet, or a breadboard hole ("a5", or
    "main:a5" with several boards), on the board's top, carrier.board_top_mm above the sheet. ValueError for anything
    else, rails included: a rail's holes have no names."""
    from wbc.perception.breadboard import boards_from  # cv2 comes with it, so only when a hole is asked for

    tags = bench["carrier"]["tags"]
    if name.startswith(TAG):
        tag = name[len(TAG):]
        if not tag.isdigit() or int(tag) not in tags:
            raise ValueError(f"there's no {name} on the carrier: its tags are {', '.join(f'{TAG}{t}' for t in tags)}")
        x, y = tags[int(tag)]
        return np.array([x, y, 0.0])
    boards = boards_from(bench, parts)
    board = next((b for b in boards if name.startswith(b.prefix)), None)
    if board is None:
        raise ValueError(f"{name} needs its board's name first, like {boards[0].prefix}a5: the bench has several")
    hole = name.removeprefix(board.prefix)
    spec = board.spec
    if len(hole) < 2 or hole[0] not in spec["columns"] or not hole[1:].isdigit() or \
            not 1 <= int(hole[1:]) <= spec["rows"]:
        raise ValueError(f"{name} isn't a tag or a hole: a hole is a column a-j and a row 1-{spec['rows']}, like a5")
    x, y = board.hole(hole)
    return np.array([x, y, float(bench["carrier"]["board_top_mm"])])


@dataclass
class Registration:
    yaw: float              # radians the carrier's flipped axes are turned about z from the arm's
    shift: np.ndarray       # metres: where the carrier's origin, the sheet's top-left corner, is in the arm's frame
    scale: float = 1.0      # how far apart the touched points are, against the sheet: only reported, never applied
    points: list = field(default_factory=list)  # [{name, carrier_mm, tool_mm, error_mm}]
    when: str = ""

    def to_arm(self, xyz_mm) -> np.ndarray:
        """A carrier point, mm, in the arm's frame, metres."""
        return rot_z(self.yaw) @ FLIP @ (np.asarray(xyz_mm, float) / 1000) + self.shift

    def to_carrier(self, p) -> np.ndarray:
        """A point in the arm's frame, metres, in carrier mm."""
        return 1000 * FLIP @ rot_z(-self.yaw) @ (np.asarray(p, float) - self.shift)

    @property
    def rms_mm(self) -> float:
        return float(np.sqrt(np.mean([p["error_mm"] ** 2 for p in self.points]))) if self.points else 0.0

    @property
    def worst(self) -> dict | None:
        return max(self.points, key=lambda p: p["error_mm"], default=None)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"yaw": self.yaw, "shift": list(self.shift), "scale": self.scale,
                                    "points": self.points, "when": self.when}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> Registration:
        """FileNotFoundError, saying how to make one, when there's none at path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"the arm isn't registered to the carrier ({path} is missing): run "
                                    "scripts/register_arm.py touch, with the motors off")
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(float(d["yaw"]), np.array(d["shift"], float), float(d.get("scale", 1.0)), d.get("points", []),
                   d.get("when", ""))


def fit(names, carrier_mm, tool) -> Registration:
    """The turn about z and the shift that best take carrier_mm (N x 3, mm) to tool (N x 3, the tool point in the arm's
    frame, metres), least squares, with each point's distance from where the fit puts it. ValueError for under
    MIN_POINTS points, a name twice, or points too close to one line to fix the turn."""
    names = list(names)
    a = (np.asarray(carrier_mm, float) / 1000) @ FLIP
    b = np.asarray(tool, float)
    if len(names) < MIN_POINTS or a.shape != (len(names), 3) or b.shape != a.shape:
        raise ValueError(f"a fit needs at least {MIN_POINTS} touched points, each with its carrier mm and its tool "
                         f"point, not {len(names)}")
    if len(set(names)) != len(names):
        raise ValueError("a point was touched twice: " + ", ".join(sorted({n for n in names if names.count(n) > 1})))
    a0, b0 = a[:, :2] - a[:, :2].mean(axis=0), b[:, :2] - b[:, :2].mean(axis=0)
    spread = np.linalg.svd(a0, compute_uv=False)[-1] / math.sqrt(len(names))
    if spread < MIN_SPREAD:
        raise ValueError(f"the points are too close to one line ({1000 * spread:.0f} mm across it) to fix the turn: "
                         "touch some well off it, like the tags at the sheet's four corners")
    yaw = math.atan2(float(np.sum(a0[:, 0] * b0[:, 1] - a0[:, 1] * b0[:, 0])),
                     float(np.sum(a0[:, 0] * b0[:, 0] + a0[:, 1] * b0[:, 1])))
    R = rot_z(yaw)
    shift = b.mean(axis=0) - R @ a.mean(axis=0)
    turned = a0 @ R[:2, :2].T
    scale = float(np.sum(turned * b0) / np.sum(a0 * a0))
    errors = np.linalg.norm(a @ R.T + shift - b, axis=1) * 1000
    points = [{"name": n, "carrier_mm": [round(float(v), 2) for v in c], "tool_mm": [round(1000 * float(v), 2) for v in t],
               "error_mm": round(float(e), 2)} for n, c, t, e in zip(names, np.asarray(carrier_mm, float), b, errors)]
    return Registration(yaw, shift, scale, points, datetime.now().isoformat(timespec="seconds"))


@dataclass
class Aim:
    q: np.ndarray        # URDF radians in ARM order: the tool point HOVER back from the spot, the gripper on its line
    q_above: np.ndarray  # the same line, ABOVE further back
    pitch: float         # degrees below level the gripper points
    miss: float          # metres the gripper's line passes from the spot
    spot: np.ndarray     # the spot, in the arm's frame, metres


def trouble(chain, joints, q) -> str:
    """Why the driver's move_to() would refuse to end at q, or '' when it wouldn't: a joint's target outside its range
    less RANGE_MARGIN, or the tool point or the elbow and wrist axes under their floors."""
    for i, m in enumerate(ARM):
        lo, hi = (math.degrees(v) for v in joints.limits(m))
        lo, hi, at = lo + RANGE_MARGIN, hi - RANGE_MARGIN, math.degrees(q[i])
        if not lo <= at <= hi:
            return f"the {NAMES[m]} would need {at:+.0f}°, outside {lo:+.0f}° to {hi:+.0f}°"
    tool, axis = lowest(chain, q, FLOOR_JOINTS)
    if tool < TOOL_FLOOR:
        return f"the tool point would be {1000 * tool:.0f} mm above the table, under its {1000 * TOOL_FLOOR:.0f} mm floor"
    if axis < JOINT_FLOOR:
        return f"the elbow or wrist would be {1000 * axis:.0f} mm above the table, under their {1000 * JOINT_FLOOR:.0f} mm"
    return ""


def on_line(chain, q, spot) -> tuple[float, float]:
    """How far the tool point is back from the spot along the gripper's line, and how far that line passes from it,
    metres."""
    T = chain.fk(q)
    d, v = T[:3, 0], np.asarray(spot, float) - T[:3, 3]
    back = float(v @ d)
    return back, float(np.linalg.norm(v - back * d))


def _toward(chain, spot, back: float, pitch: float, q0, roll: float = 0.0):
    """IK for the tool point `back` metres from the spot along a line `pitch` radians below level, the gripper on that
    line and turned `roll` about it: (q, how far its line misses the spot, roll), or None when the solver doesn't get
    there."""
    yaw, q = math.atan2(spot[1], spot[0]), q0
    for _ in range(LINE_TRIES):
        R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
        res = solve_ik(chain, spot - back * R[:, 0], R, q, yaw_task=(yaw, pitch, roll))
        if not res.ok:
            return None
        q, yaw = res.q, res.yaw
        along, miss = on_line(chain, q, spot)
        if miss <= MISS and abs(along - back) <= MISS:
            return q, miss, roll
    return None


def _turns(chain, spot, back: float, pitch: float, q0) -> list:
    """_toward's poses for one line, the wrist roll nearest q0's first. Pointing doesn't care how the jaws are turned
    about the line, and the roll's cable does: from the rest pose on Oct 29 (roll -149.9°), a pose at roll 0 would
    first turn the gripper about 150°. Turning about the line moves the wrist roll joint one way or the other, so both
    are tried, with roll 0 as well."""
    plain = _toward(chain, spot, back, pitch, q0)
    if plain is None:
        return []
    turn = (q0[ROLL] - plain[0][ROLL] + math.pi) % (2 * math.pi) - math.pi
    found = [plain] + [f for f in (_toward(chain, spot, back, pitch, q0, r) for r in (turn, -turn)) if f is not None]
    return sorted(found, key=lambda f: abs(f[0][ROLL] - q0[ROLL]))


def aim(chain, joints, spot, q0, hover: float = HOVER, above: float = ABOVE) -> Aim:
    """The steepest pose in PITCHES that points the gripper at `spot` (arm frame, metres) from `hover` back, with one
    from `hover + above` back on the same line, both somewhere move_to() would go. q0, the arm's joints now, seeds the
    solver. ValueError saying why when no pitch works."""
    spot = np.asarray(spot, float)
    if math.hypot(spot[0], spot[1]) > REACH:
        raise ValueError(f"the spot is {1000 * math.hypot(spot[0], spot[1]):.0f} mm from the base's axis, out of reach")
    q0 = np.asarray(q0, float)
    unsolved = limited = ""  # the steepest pitch's reason of each kind: a pose the driver refuses says more
    for degrees in PITCHES:
        pitch = math.radians(degrees)
        nears = _turns(chain, spot, hover, pitch, q0)
        if not nears:
            unsolved = unsolved or f"no arm pose puts the gripper on that line {1000 * hover:.0f} mm back"
        for near in nears:
            far = _toward(chain, spot, hover + above, pitch, near[0], near[2])
            if far is None:
                unsolved = unsolved or f"no arm pose reaches {1000 * (hover + above):.0f} mm back"
                continue
            problem = trouble(chain, joints, near[0]) or trouble(chain, joints, far[0])
            if problem:
                limited = limited or problem
                continue
            return Aim(near[0], far[0], float(degrees), near[1], spot)
    raise ValueError(f"can't point at it at any pitch from {PITCHES[0]}° to {PITCHES[-1]}° down: "
                     f"{limited or unsolved}")


def point_at(arm, registration: Registration, spot_mm, hover: float = HOVER, above: float = ABOVE) -> Aim:
    """Point the gripper at a carrier point (mm, from point_mm): first to the line's far pose, then in to `hover` back.
    The arm has to be powered on already. Everything move_to() refuses or trips on comes straight through: this never
    powers the arm on or clears a stop."""
    a = aim(arm.chain, arm.joints, registration.to_arm(spot_mm), arm.read().q, hover, above)
    arm.move_to(a.q_above)
    arm.move_to(a.q)
    return a


def registration_path(root) -> Path:
    return Path(root) / ".wbc" / "registration.json"
