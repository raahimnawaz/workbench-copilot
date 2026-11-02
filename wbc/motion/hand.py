"""Moves in mm on top of the arm driver, and the general moves the agent makes with them: picking a thing up and
putting it down.

The agent decides what goes where (wbc/agent/tools.py); this decides how. A target is a point in the arm's frame in mm,
with the gripper's pitch (90° points it straight down) and the line its jaws close along, solved by
kinematics.solve_task from where the arm is. Every limit stays in the driver: a move it refuses is refused here too,
never forced, and nothing here talks to the servos except through it.

pick_up and put_down take a thing as wbc/perception/desk.py's locate found it: its middle, its top's height and the
heading of its long axis. A pick goes straight down over its middle with the jaws closing across that axis: over it,
open, down, grip, and back up. Jaws that close on nothing, or come up empty, are a miss, and they open again. A put
goes over the place with what's held lined up with the place's long axis, down to just over its top, opens, and goes
back up. Every pose a pick or a put needs is solved before the first one moves, so one the arm can't reach is refused
with the arm where it was.

Which way the jaws close: pointing straight down with the wrist roll at config/robot.yaml's hand.across_roll_deg, they
close across the arm, along the wrist flex's axis. That's from a look at the jaws in the rest pose (Oct 29), where the
line between them runs down the arm's middle with the roll reading about 0°. The URDF has them a quarter turn from
there, so the jaws' line comes from the wrist flex's axis, not from the URDF's frame. The four-touch test (Oct 29,
22:34) put the tips 3.7 mm from the roll's axis, too near it to tell which way round they are.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .arm import RANGE_MARGIN, TOOL_FLOOR, Refused
from .joints import ARM, NAMES
from .kinematics import pose_to_task, solve_task

HOVER_MM = 60.0         # over a thing's top: where a pick or a put starts and ends
FLOOR_MARGIN_MM = 3.0   # over the driver's tool floor that a pick goes down to, so its end isn't refused for a scrap
GRIP_DEPTH_MM = 15.0    # under its top that the tips go to on something standing up
PUT_ABOVE_MM = 30.0     # over the place's top that a put opens the jaws
ROUND = 1.3             # a thing whose length is under this many times its width has no long axis to grip across
IK_MISS_MM = 2.0        # an IK answer further than this from its target is refused
TILT_MISS_DEG = 3.0     # and one whose gripper points further than this from the pitch asked for
JAW_MISS_DEG = 5.0      # and one whose jaws close further than this from the line asked for
HOME_CLEAR = RANGE_MARGIN + 1.0  # degrees in from each end of its range that home() takes a joint, as the bench
                                 # scripts' moves back to where they started do
DOWN = 60.0             # degrees of pitch from which the jaws' line is a heading on the table
STEER_ROUNDS = 8        # IK solves a target may take to bring the jaws' line within half a degree


def line(a: float) -> float:
    """A line's heading in degrees, read either way round: -90 (exclusive) to 90."""
    a = (a + 90.0) % 180.0 - 90.0
    return 90.0 if a == -90.0 else a


@dataclass
class Thing:
    """What a pick or a put needs to know about a thing, in the arm's frame: mm and degrees, as
    desk.Found.summary() gives them."""

    name: str
    x: float
    y: float
    top_z: float
    heading_deg: float = 0.0  # its long axis, from the arm's straight ahead (+x) toward its left (+y)
    length_mm: float = 0.0
    width_mm: float = 0.0
    height_mm: float = 0.0

    @classmethod
    def from_summary(cls, s: dict) -> Thing:
        return cls(s["phrase"], s["x_mm"], s["y_mm"], s["top_z_mm"], s["heading_deg"], s["length_mm"], s["width_mm"],
                   s["height_mm"])

    @property
    def jaws_deg(self) -> float | None:
        """The line the jaws close along to hold it across its long axis, or None when it's about round."""
        if self.length_mm < ROUND * max(self.width_mm, 1.0):
            return None
        return line(self.heading_deg + 90.0)


@dataclass
class Step:
    what: str
    xyz: tuple          # mm, the tool point
    jaws_deg: float     # the line the jaws close along
    q: np.ndarray       # the joint angles that do it, radians in ARM order

    def summary(self) -> dict:
        return {"step": self.what, "x_mm": round(self.xyz[0], 1), "y_mm": round(self.xyz[1], 1),
                "z_mm": round(self.xyz[2], 1), "jaws_deg": round(self.jaws_deg, 1)}


class Hand:
    def __init__(self, arm, robot_cfg: dict, say=print):
        """arm: a wbc.motion.arm.Arm. robot_cfg: config/robot.yaml's contents, for its hand settings."""
        self.arm, self.chain, self.say = arm, arm.chain, say
        hand = robot_cfg.get("hand") or {}
        self.across_roll = float(hand.get("across_roll_deg", 0.0))
        self.rest = None   # where power_on found the arm, just inside its ranges: home() goes back there
        self.held = None   # the name of what the jaws hold, from pick_up until put_down
        q = np.zeros(self.chain.n)
        q[self.chain.names.index("wrist_roll")] = math.radians(self.across_roll)
        T, axes = self.chain.frames(q)
        jaw = T[:3, :3].T @ axes[self.chain.names.index("wrist_flex")][1]
        jaw[0] = 0.0  # across the approach axis, the tool frame's x
        self.jaw_axis = jaw / np.linalg.norm(jaw)  # the line the jaws close along, in the tool frame

    def jaws_heading(self, q) -> float:
        """The line the jaws close along at joint angles q, as a heading in degrees on the table."""
        v = self.chain.fk(q)[:3, :3] @ self.jaw_axis
        return line(math.degrees(math.atan2(v[1], v[0])))

    def where(self) -> dict:
        """Where the tool point is (mm), how far the gripper points down, the jaws' line, and what they hold."""
        reading = self.arm.read()
        x, y, z, pitch, _ = pose_to_task(self.chain.fk(reading.q))
        grip = self.arm.jaw()
        return {"x_mm": round(1000 * x, 1), "y_mm": round(1000 * y, 1), "z_mm": round(1000 * z, 1),
                "pitch_deg": round(math.degrees(pitch), 1), "jaws_deg": round(self.jaws_heading(reading.q), 1),
                "opening": round(grip.opening), "holding": self.held if grip.holding else None}

    def power_on(self) -> dict:
        """Bring the motors on (the driver's power_on), and keep where the arm was, just inside its ranges, for
        home()."""
        shift = self.arm.power_on()
        q = np.array(self.arm.read().q, float)
        for i, m in enumerate(ARM):
            lo, hi = self.arm.joints.limits(m)
            q[i] = min(max(q[i], lo + math.radians(HOME_CLEAR)), hi - math.radians(HOME_CLEAR))
        self.rest = q
        return shift

    def _inside(self, q) -> str:
        """'' when every joint of q is where the driver's moves may take it, else which isn't."""
        for i, m in enumerate(ARM):
            lo, hi = (math.degrees(v) for v in self.arm.joints.limits(m))
            if not lo + RANGE_MARGIN <= math.degrees(q[i]) <= hi - RANGE_MARGIN:
                return (f"the {NAMES[m]} would have to go to {math.degrees(q[i]):+.0f}°, outside "
                        f"{lo + RANGE_MARGIN:+.0f}° to {hi - RANGE_MARGIN:+.0f}°")
        return ""

    def solve(self, xyz_mm, pitch_deg: float = 90.0, jaws_deg: float | None = None, q0=None) -> np.ndarray:
        """Joint angles, ARM order in radians, that put the tool point at xyz_mm with the gripper pitch_deg down and the
        jaws closing along jaws_deg (either way round; None keeps the line they have at q0). Starts from q0, by default
        where the arm is, and of the ways round the wrist roll could go takes the one nearest q0's that stays inside the
        driver's ranges. Refused when none does, or the answer misses by more than IK_MISS_MM, TILT_MISS_DEG or
        JAW_MISS_DEG."""
        q0 = np.array(self.arm.read().q if q0 is None else q0, float)
        x, y, z = (float(v) / 1000 for v in xyz_mm)
        steer = pitch_deg >= DOWN  # pointing down, the jaws' line is a heading the roll can turn
        want = line(self.jaws_heading(q0) if jaws_deg is None else jaws_deg)
        roll_i = ARM.index("wrist_roll")
        found, why = [], ""
        for turn in (0.0, math.pi, -math.pi):  # the task roll's ways round: the same line, the jaws either way round
            roll, before = turn, None
            for _ in range(STEER_ROUNDS):
                res = solve_task(self.chain, x, y, z, math.radians(pitch_deg), roll, q0=q0)
                miss = line(want - self.jaws_heading(res.q)) if steer else 0.0
                if abs(miss) < 0.5:
                    break
                # Straight down, the task's roll and the yaw IK may choose both turn about the vertical, so a degree of
                # roll turns the jaws' line by a degree or less: the slope comes from the last two tries
                slope = 1.0
                if before is not None and roll != before[0]:
                    measured = (miss - before[1]) / math.degrees(roll - before[0])
                    slope = measured if abs(measured) > 0.1 else slope
                before = (roll, miss)
                roll -= math.radians(miss / slope)
            pitch = math.degrees(pose_to_task(self.chain.fk(res.q))[3])
            if 1000 * res.pos_err > IK_MISS_MM or abs(pitch - pitch_deg) > TILT_MISS_DEG:
                why = why or (f"the arm can't put the tool point at x {xyz_mm[0]:.0f}, y {xyz_mm[1]:.0f}, z "
                              f"{xyz_mm[2]:.0f} mm pointing {pitch_deg:.0f}° down: the nearest is "
                              f"{1000 * res.pos_err:.0f} mm off")
                continue
            if steer and abs(miss) > JAW_MISS_DEG:
                why = why or f"the jaws can't close along {want:+.0f}° there: the nearest is {miss:+.0f}° off"
                continue
            outside = self._inside(res.q)
            if outside:
                why = why or outside
                continue
            found.append(res.q)
        if not found:
            raise Refused(why or "IK found no pose")
        return min(found, key=lambda q: abs(q[roll_i] - q0[roll_i]))

    def move_to(self, xyz_mm, pitch_deg: float = 90.0, jaws_deg: float | None = None, seconds: float | None = None):
        """Move the tool point to xyz_mm, solved by solve(), through the driver's move_to. Returns its last reading."""
        return self.arm.move_to(self.solve(xyz_mm, pitch_deg, jaws_deg), seconds)

    def home(self):
        """Back to where power_on found the arm, just inside its ranges."""
        if self.rest is None:
            raise Refused("there's no home yet: power the arm on first")
        return self.arm.move_to(self.rest)

    def _floor_z(self) -> float:
        """The lowest the tool point may go, mm in the arm's frame, with FLOOR_MARGIN_MM to spare."""
        return 1000 * (self.arm.table + TOOL_FLOOR) + FLOOR_MARGIN_MM

    def _solved(self, plan: list) -> list:
        """Each (what, xyz, jaws) of a plan solved in turn, from where the arm is: every pose before any moves."""
        q = np.array(self.arm.read().q, float)
        jaws = self.jaws_heading(q)
        steps = []
        for what, xyz, want in plan:
            q = self.solve(xyz, 90.0, jaws if want is None else want, q0=q)
            jaws = self.jaws_heading(q)
            steps.append(Step(what, xyz, jaws, q))
        return steps

    def plan_pick(self, thing: Thing) -> list:
        """The poses a pick goes through, solved: over the thing, down to it, and back up."""
        grip_z = max(self._floor_z(), thing.top_z - (GRIP_DEPTH_MM if thing.height_mm > 0 else 0.0))
        over = (thing.x, thing.y, max(grip_z, thing.top_z) + HOVER_MM)
        jaws = thing.jaws_deg
        return self._solved([("over it", over, jaws), ("down to it", (thing.x, thing.y, grip_z), jaws),
                             ("back up", over, jaws)])

    def plan_put(self, place: Thing) -> list:
        """The poses a put goes through, solved: over the place, down to just over its top, and back up."""
        open_z = max(self._floor_z(), place.top_z + PUT_ABOVE_MM)
        over = (place.x, place.y, max(open_z, place.top_z) + HOVER_MM)
        jaws = place.jaws_deg
        return self._solved([("over it", over, jaws), ("down to it", (place.x, place.y, open_z), jaws),
                             ("back up", over, jaws)])

    def pick_up(self, thing: Thing, confirm=None) -> dict:
        """Pick `thing` up: over it, open, down, grip, back up. confirm(prompt), when given, is asked over the thing
        before the jaws open and go down; False stops there. Refused (the driver's or IK's) before anything moves when a
        pose can't be reached; Tripped, with the motors cut, as the driver says."""
        if self.held:
            raise Refused(f"the jaws already hold the {self.held}: put it down first")
        over, down, up = self.plan_pick(thing)
        self.arm.move_to(over.q)
        if confirm is not None and not confirm(f"Over the {thing.name}, the jaws along {over.jaws_deg:+.0f}°. Go down "
                                               "and grip it?"):
            return {"ok": False, "stopped": f"over the {thing.name}, as you asked", "steps": [over.summary()]}
        self.arm.release(100.0)
        self.arm.move_to(down.q)
        grip = self.arm.grip()
        if grip.holding:
            self.arm.move_to(up.q)
            grip = self.arm.jaw()
        if not grip.holding:
            self.arm.release(100.0)
            self.arm.move_to(up.q)
            return {"ok": False, "missed": True, "error": f"the jaws closed on nothing where the {thing.name} was",
                    "steps": [s.summary() for s in (over, down, up)]}
        self.held = thing.name
        return {"ok": True, "holding": thing.name, "jaws_open_percent": round(grip.opening),
                "steps": [s.summary() for s in (over, down, up)]}

    def put_down(self, place: Thing, confirm=None) -> dict:
        """Put what the jaws hold down on or in `place`: over it, down to just over its top, open, back up. confirm as
        for pick_up, asked over the place before it goes down."""
        if not self.held and not self.arm.jaw().holding:
            raise Refused("the jaws hold nothing to put down")
        over, down, up = self.plan_put(place)
        self.arm.move_to(over.q)
        if confirm is not None and not confirm(f"Over the {place.name}. Go down and let go?"):
            return {"ok": False, "stopped": f"over the {place.name}, as you asked", "steps": [over.summary()]}
        self.arm.move_to(down.q)
        self.arm.release(100.0)
        self.arm.move_to(up.q)
        held, self.held = self.held, None
        return {"ok": True, "put": held, "on": place.name, "steps": [s.summary() for s in (over, down, up)]}
