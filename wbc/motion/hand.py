"""Moves in mm on top of the arm driver, and the general moves the agent makes with them: picking a thing up and
putting it down.

The agent decides what goes where (wbc/agent/tools.py); this decides how. A target is a point in the arm's frame in mm,
with the gripper's pitch (90° points it straight down) and the line its jaws close along, solved from where the arm
is: straight down by kinematics.solve_ik, with the jaws' heading part of the rotation it asks for, and tilted by
kinematics.solve_task. Every limit stays in the driver: a move it refuses is refused here too, never forced, and
nothing here talks to the servos except through it.

pick_up and put_down take a thing as wbc/perception/desk.py's locate found it: its middle, its size, the heading of
its long axis and, for something standing up, its height. A pick goes straight down beside its middle with the jaws
closing across that axis: over it, open as far as its width needs, just over its top, down, grip, and back up. A put
goes over the place with what's held lined up with the place's long axis, down to just over its top, opens as far as
the pick did, goes back up and closes the jaws. Only the moves down to the thing (just over it, then down) and down
to the place re-aim; over it and back up, a few mm short is as good, and the re-aims' settling would only add stops.
Every pose a pick or a put needs is solved before the first one moves, so one the arm can't reach is refused with the
arm where it was.

Heights come from the table the driver's floors use (table.py), the desk as the arm's touches measured it, not from
locate's camera, whose table is one height for the whole desk. Something lying flat is taken to be as thick as it is
wide, like a handle, and gripped GRIP_SHARE of the way up, under its middle, so the tips close under a round handle's
widest point rather than over it, where they'd push it down and out from under them; something standing up
GRIP_DEPTH_MM under its top. Neither goes under the floor the driver gives the way down to a grip (GRIP_FLOOR, 1 mm
over the table's 2 mm margin), so with FLOOR_MARGIN_MM the tips go no lower than about 4 mm over the desk, and anything
they'd grip above THIN of its height is refused, since they'd close near its top and slip off. That's judged where the
tips come down beside it, whose floor can be higher, against the desk under it as the touches measured it. So is a
pick where no touch measured the desk (table.py holds it high there).

Jaws that close all the way may still hold the thing. These close a handle up into the gap above their tips, and hold
it there reading as close to shut as empty jaws do (15-25 ticks from closed holding a screwdriver, 19-24 empty), so a
pick goes back up with them shut either way and says when it can't tell (unconfirmed): only a look at where the thing
was can.

The arm stops short of each target by its joints' load over their P gains. The driver's re-aim makes that up at the
end of a move, but not during the next one, so going down to the grip the arm reads lower than its plan by about as
much as it stopped short just over the thing. When that would bring it within SAG_SPARE_MM of the driver's cut for
sagging under a floor, the pick stops just over the thing rather than trip it.

One jaw is fixed and only the other moves, so a pick doesn't centre the jaws on the thing: the tool point, between the
closed tips, sits CLOSED_GAP_MM / 2 from the fixed jaw's face, which would come down on the thing's middle. It goes down
with the fixed jaw JAW_CLEAR_MM beside the thing instead, and the moving jaw closes it against the fixed one. Which side
the moving jaw is on comes from config/robot.yaml's hand.moving_jaw, looking out from the base with the gripper
pointing straight down at the rest roll.

Pointing straight down, the jaws close along their line either way round, half a turn of the wrist roll apart, and
the way round decides which side of the thing the fixed jaw comes down. A pick or a put picks the way round once,
before anything moves (whichever needs the wrist roll nearest where it is), and every pose it goes through keeps it.

Which way the jaws close: pointing straight down with the wrist roll at config/robot.yaml's hand.across_roll_deg, they
close across the arm, along the wrist flex's axis. In the rest pose the line between them runs down the arm's middle
with the roll reading about 0°. The URDF has them a quarter turn from there, so the jaws' line comes from the wrist
flex's axis, not from the URDF's frame. The tips sit 3.7 mm from the roll's axis, by a four-touch test, too near it to
tell which way round they are.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import numpy as np

from .arm import (FLOOR_JOINTS, GRIP_FLOOR, MAX_JOINT_SPEED, MAX_TIP_SPEED, RANGE_MARGIN, SAG, TICKS_PER_DEGREE, Halted,
                  Refused)
from .joints import ARM, NAMES
from .kinematics import pose_to_task, solve_ik, solve_task
from .table import PAN_AXIS
from .trajectory import lowest, min_seconds

HOVER_MM = (60.0, 45.0, 30.0)  # over a thing's top, where a pick or a put starts and ends: the first the arm can
                               # reach straight down, since nearer the base or further out it can't point down as high
FLOOR_MARGIN_MM = 1.0   # over the driver's tool floor that a pick goes down to, so its end isn't refused for a scrap
THICK_MM = 40.0         # the thickest something lying flat is taken to be, whatever its width
GRIP_DEPTH_MM = 15.0    # under its top that the tips go to on something standing up
GRIP_SHARE = 0.25       # of its thickness that the tips go up something lying flat: well under a handle's middle, and
                        # for one under 16 mm, down to the grip floor, so they close under its widest point
CLOSED_GAP_MM = 5.0     # the closed tips' gap: the tool point sits half of it from the fixed jaw's face
JAW_CLEAR_MM = 5.0      # between the fixed jaw and the thing's side at the grip: the four-touch fit puts the closed
                        # tips' middle within about 2.4 mm (one sd) along the jaws' line
JUST_OVER_MM = 5.0      # over the thing's top that a pick's last way down starts from: a short move, so its line in
                        # joint space can't swing the jaws into the thing (5.5 mm off the vertical over a 60 mm one)
SAG_SPARE_MM = 1.0      # that a pick keeps between where its grip would read, sagging as it did just over the thing,
                        # and the driver's sag cut
THIN = 2 / 3            # of its height over the desk that the tips have to get under for a pick to grip a thing
HELD_MM = 2.0           # a pick is refused where the table is held more than this over its touched line: the desk's
                        # height there isn't known well enough to grip near a thing's middle
ROLL_STEP_DEG = 30.0    # the most the wrist roll may turn from one step of a plan to the next, the jaws one way round
OPEN_ENOUGH = 90.0      # percent of the opening asked for that the jaws have to reach before a pick goes down
JAW_LEVER_MM = 70.0     # how far the moving jaw's tip goes across per radian it opens: the URDF puts the gripper's
                        # pivot 75 mm up the jaws and 28 mm across from the point between the tips, and a ~10 mm handle
                        # stopped the closing jaws 43 ticks past where empty ones stop, 76 mm a radian. Unmeasured at
                        # the tips themselves, so a little under both
MOVING_CLEAR_MM = 15.0  # past the thing's far side that the open moving jaw comes down: the desk map places a touched
                        # tag 1-9 mm from where the others put it
OPEN_SPARE = 1.25       # times the angle that lever says a pick needs, that it opens the jaws
MIN_OPEN = 25.0         # percent: the least a pick opens the jaws, a quarter of their range, about 34°
CLOSED_OPENING = 2.0    # percent: where a put or a miss leaves the jaws once it's done, closed but 31 ticks from their
                        # closed end, short of the 19-24 where empty ones stop at P 32, so they don't keep pushing
PUT_ABOVE_MM = 30.0     # over the place's top that a put opens the jaws
LIFT_MM = 95.0          # over the desk that a put into a container lifts what's held to first, the jaws turned across
                        # the arm so it lies along it: turning the gripper up from straight down then stands it on end
ON_END_TILTS = (0.0, 15.0, 30.0)  # degrees down from level the gripper may point to hold it on end over a container:
                        # level stands it straight, which the arm's model reaches only from about 28 cm out
ON_END_ABOVE_MM = 60.0  # over a container's rim that what's held, on end, swings in at, its lower end clear of the rim
ON_END_CLEAR_MM = 15.0  # over the rim its lower end comes down to before it goes in
INSERT_MM = 15.0        # under the rim its lower end goes before the jaws let go of it: the gripper's own body, just
                        # over the rim beside it then, has to stay clear of it
CAGE_MM = 8.0           # back from the tips, toward the palm, that a loosely held handle rides, in the gap above them:
                        # an estimate, not measured
ON_END_SLOW = 2.0       # times as long as the speed caps allow that the turns on end take, so a loose hold keeps it:
                        # at this, a screwdriver turned to level on end stayed in the jaws
ROUND = 1.3             # a thing whose length is under this many times its width has no long axis to grip across
THIN_STANDING_MM = 6.0  # standing up, a thing thinner than this is under the closed tips' gap, so a pick grips it across
                        # its length instead: a screen stood on its long edge, by its ends (Nov 3)
IK_MISS_MM = 2.0        # an IK answer further than this from its target is refused
TILT_MISS_DEG = 3.0     # and one whose gripper points further than this from the pitch asked for
JAW_MISS_DEG = 5.0      # and one whose jaws close further than this from the line asked for
HOME_CLEAR = RANGE_MARGIN + 1.0  # degrees in from each end of its range that home() takes a joint, as the bench
                                 # scripts' moves back to where they started do
DOWN = 60.0             # degrees of pitch from which the jaws' line is a heading on the table
STEER_ROUNDS = 8        # IK solves a tilted target may take to bring the jaws' line within half a degree
REACHING = np.radians([0.0, 50.0, -50.0, 90.0, 0.0])  # IK's start straight down when one from where the arm is
                                                       # stalls: reaching out, with the base and roll set per target
ROLL = ARM.index("wrist_roll")


def line(a: float) -> float:
    """A line's heading in degrees, read either way round: -90 (exclusive) to 90."""
    a = (a + 90.0) % 180.0 - 90.0
    return 90.0 if a == -90.0 else a


def turn(a: float) -> float:
    """An angle in degrees as the shortest turn, -180 to 180."""
    return (a + 180.0) % 360.0 - 180.0


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
    grip_x: float | None = None  # where a pick grips it, when locate found an even stretch to grip off its middle
    grip_y: float | None = None
    grip_width_mm: float | None = None  # how wide it is there
    by_ends: bool = False  # gripped across its length rather than its width (THIN_STANDING_MM)

    @classmethod
    def from_summary(cls, s: dict) -> Thing:
        return cls(s["phrase"], s["x_mm"], s["y_mm"], s["top_z_mm"], s["heading_deg"], s["length_mm"], s["width_mm"],
                   s["height_mm"], s.get("grip_x_mm"), s.get("grip_y_mm"), s.get("grip_width_mm"))

    def gripped(self) -> Thing:
        """Itself as a pick takes it: centred where locate said to grip it, as wide as it is there. Something standing
        up thinner than THIN_STANDING_MM is taken by its ends, as wide as it's long."""
        if self.height_mm > 0 and self.width_mm < THIN_STANDING_MM and not self.by_ends:
            return replace(self, width_mm=self.length_mm, by_ends=True)
        if self.grip_x is None:
            return self
        return replace(self, x=self.grip_x, y=self.grip_y, width_mm=self.grip_width_mm or self.width_mm)

    @property
    def jaws_deg(self) -> float | None:
        """The line the jaws close along to hold it across its long axis (along it when gripped by its ends), or None
        when it's about round."""
        if self.by_ends:
            return line(self.heading_deg)
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
        idle = hand.get("idle_deg")
        self.roll_on_the_way = bool(hand.get("roll_on_the_way", False))  # see put_in
        self.idle_q = np.radians(np.asarray(idle, float)) if idle is not None else None  # see idle()
        self.held = None   # the name of what the jaws hold, from pick_up until put_down
        self.held_thing = None  # it, as locate found it, for a put into a container: its length and narrow end
        self.held_tip = None    # where its narrow end points, degrees from the jaw axis; None when it has none
        self.opened = None  # percent the pick that's holding it opened the jaws to: put_down opens them that far
        self.log = None    # a list while the agent records moves: each move and jaw action, with its samples
        q = np.zeros(self.chain.n)
        q[self.chain.names.index("wrist_roll")] = math.radians(self.across_roll)
        T, axes = self.chain.frames(q)
        flex = axes[self.chain.names.index("wrist_flex")][1]
        jaw = T[:3, :3].T @ flex
        jaw[0] = 0.0  # across the approach axis, the tool frame's x
        self.jaw_axis = jaw / np.linalg.norm(jaw)  # the line the jaws close along, in the tool frame
        # From the fixed jaw toward the moving one, in the tool frame. At q, the arm points along +x and the flex's axis
        # lies across it, so "left" looking out from the base is +y.
        side = hand.get("moving_jaw")
        if side not in (None, "left", "right"):
            raise ValueError(f"robot.yaml's hand.moving_jaw is left or right, not {side!r}")
        # 1 when the moving jaw is the way the jaw axis points, -1 when it's the other way, None when it isn't set
        self.moving_sign = None if side is None else (1.0 if (flex[1] > 0) == (side == "left") else -1.0)
        self.moving_axis = None if side is None else self.moving_sign * self.jaw_axis

    def jaws_heading(self, q) -> float:
        """The line the jaws close along at joint angles q, as a heading in degrees on the table."""
        return line(self.jaws_toward(q))

    def jaws_toward(self, q) -> float:
        """Which way the jaw axis points at joint angles q, a heading in degrees on the table: the jaws' line, one way
        round."""
        v = self.chain.fk(q)[:3, :3] @ self.jaw_axis
        return math.degrees(math.atan2(v[1], v[0]))

    def _down(self, toward: float) -> np.ndarray:
        """The tool's rotation pointing straight down with its jaw axis toward `toward`, a heading in degrees."""
        t = math.radians(toward)
        a, j, x = np.array([0.0, 0.0, -1.0]), np.array([math.cos(t), math.sin(t), 0.0]), np.array([1.0, 0.0, 0.0])
        return np.column_stack([a, j, np.cross(a, j)]) @ \
            np.column_stack([x, self.jaw_axis, np.cross(x, self.jaw_axis)]).T

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
        shift = self.arm.power_on(samples=self._samples("power on"))
        q = np.array(self.arm.read().q, float)
        for i, m in enumerate(ARM):
            lo, hi = self.arm.joints.limits(m)
            q[i] = min(max(q[i], lo + math.radians(HOME_CLEAR)), hi - math.radians(HOME_CLEAR))
        self.rest = q
        return shift

    def _samples(self, what: str) -> list | None:
        """While self.log is a list, a new entry in it for one move or jaw action, `what`, with when it started, and the
        list the driver fills with its samples; None otherwise."""
        if self.log is None:
            return None
        self.log.append({"what": what, "started": round(time.time(), 3), "samples": []})
        return self.log[-1]["samples"]

    def _move(self, step: Step, aim: bool = True, grip: bool = False, slow: float | None = None):
        """The driver's move_to to a step's pose, recorded as the step; with `slow`, taking that many times as long as
        the speed caps allow."""
        seconds = None
        if slow is not None:
            seconds = slow * min_seconds(self.arm.read().q, step.q, math.radians(MAX_JOINT_SPEED), self.chain,
                                         MAX_TIP_SPEED)
        return self.arm.move_to(step.q, seconds=seconds, samples=self._samples(step.what), aim=aim, grip=grip)

    def opening_for(self, thing: Thing) -> float:
        """How far a pick opens the jaws for `thing`, in percent: far enough for the moving jaw to come down
        MOVING_CLEAR_MM past its far side with the fixed jaw JAW_CLEAR_MM off its near one, by JAW_LEVER_MM, times
        OPEN_SPARE; MIN_OPEN at least, and fully open for anything wider than that lever reaches."""
        across = thing.width_mm + JAW_CLEAR_MM + MOVING_CLEAR_MM - CLOSED_GAP_MM  # how far the moving tip goes
        if across >= JAW_LEVER_MM:
            return 100.0
        s = self.arm.joints.servos["gripper"]
        span = (s.range_max - s.range_min) / TICKS_PER_DEGREE  # degrees from closed to fully open
        return min(100.0, max(MIN_OPEN, 100 * OPEN_SPARE * math.degrees(math.asin(across / JAW_LEVER_MM)) / span))

    def _close(self):
        """The jaws closed again, to CLOSED_OPENING, once a put or a miss is done."""
        self.arm.release(CLOSED_OPENING, samples=self._samples("close"))

    def _inside(self, q) -> str:
        """'' when every joint of q is where the driver's moves may take it, else which isn't."""
        for i, m in enumerate(ARM):
            lo, hi = (math.degrees(v) for v in self.arm.joints.limits(m))
            if not lo + RANGE_MARGIN <= math.degrees(q[i]) <= hi - RANGE_MARGIN:
                return (f"the {NAMES[m]} would have to go to {math.degrees(q[i]):+.0f}°, outside "
                        f"{lo + RANGE_MARGIN:+.0f}° to {hi - RANGE_MARGIN:+.0f}°")
        return ""

    def solve(self, xyz_mm, pitch_deg: float = 90.0, jaws_deg: float | None = None, q0=None,
              toward: float | None = None, loose: bool = False) -> np.ndarray:
        """Joint angles, ARM order in radians, that put the tool point at xyz_mm with the gripper pitch_deg down and the
        jaws closing along jaws_deg (None keeps the line they have at q0). Starts from q0, by default where the arm
        is. Straight down, of the two ways round the jaws could close along the line, takes the one whose wrist roll is
        nearest q0's and stays inside the driver's ranges, or with `toward` (a heading in degrees) the one whose jaw
        axis points within 90° of it. Refused when none does, or the answer misses by more than IK_MISS_MM,
        TILT_MISS_DEG or JAW_MISS_DEG. Where IK from q0 stalls, it tries again (_again), keeping a try that never quite
        converged only when `loose`."""
        q0 = np.array(self.arm.read().q if q0 is None else q0, float)
        want = line(self.jaws_heading(q0) if jaws_deg is None else jaws_deg)
        if pitch_deg != 90.0:
            return self._tilted(xyz_mm, pitch_deg, want, q0)
        p = np.array([float(v) / 1000 for v in xyz_mm])
        ways = (want, want + 180.0)
        if toward is not None:
            ways = [min(ways, key=lambda d: abs(turn(d - toward)))]
        found, why = [], ""
        for way in ways:
            # Straight down, the jaws' heading is part of the rotation asked for, so IK solves it exactly: a target's
            # yaw and roll turn about the same vertical there, which steering the roll through solve_task can't tell
            # apart
            R = self._down(way)
            tries = [solve_ik(self.chain, p, R, q0=q0)]
            if not tries[0].ok:  # far out, IK from where the arm is can stall with the wrist flex on its URDF limit
                tries += [r for r in self._again(p, R, way, tries[0].q) if loose or r.ok]
            tries.sort(key=lambda r: (not r.ok, r.pos_err))  # converged first, then the nearest
            problems = [self._unusable(r, xyz_mm, way, want) for r in tries]
            if "" in problems:
                found.append(tries[problems.index("")].q)
            else:
                why = why or problems[0]
        if not found:
            raise Refused(why or "IK found no pose")
        return min(found, key=lambda q: abs(q[ROLL] - q0[ROLL]))

    def _unusable(self, res, xyz_mm, way: float, want: float) -> str:
        """'' when IK's answer puts the tool point within IK_MISS_MM of xyz_mm, pointing within TILT_MISS_DEG of
        straight down, with its jaw axis within JAW_MISS_DEG of `way` and every joint inside the driver's ranges; else
        why not."""
        pitch = math.degrees(pose_to_task(self.chain.fk(res.q))[3])
        if 1000 * res.pos_err > IK_MISS_MM or abs(pitch - 90.0) > TILT_MISS_DEG:
            above = xyz_mm[2] - self._desk_z(xyz_mm[0], xyz_mm[1])
            return (f"the arm can't put the tool point at x {xyz_mm[0]:.0f}, y {xyz_mm[1]:.0f}, z {xyz_mm[2]:.0f} mm "
                    f"({above:.0f} mm over the desk) pointing 90° down: the nearest IK found is "
                    f"{1000 * res.pos_err:.0f} mm off, pointing {pitch:.0f}° down")
        miss = turn(way - self.jaws_toward(res.q))
        if abs(miss) > JAW_MISS_DEG:
            return f"the jaws can't close along {want:+.0f}° there: the nearest is {miss:+.0f}° off"
        return self._inside(res.q)

    def _again(self, p, R, way: float, stuck) -> list:
        """IK for p and R once more, after it stalled from where the arm is at `stuck` (joint angles): from the arm
        reaching out with the base facing p and the jaw axis already toward `way`, then position first from each
        stall. Every try, up to the first that converges."""
        q = REACHING.copy()
        q[0] = -math.atan2(p[1] - PAN_AXIS[1], p[0] - PAN_AXIS[0])  # the pan turns clockwise, seen from above
        at = self.jaws_toward(q)
        q[ROLL] = math.radians(10.0)
        sign = 1.0 if turn(self.jaws_toward(q) - at) > 0 else -1.0  # which way turning the roll turns the jaw axis
        q[ROLL] = math.radians(turn(sign * turn(way - at)))
        tries = [solve_ik(self.chain, p, R, q0=q, seeds=1)]
        for start in (stuck, tries[0].q):
            if tries[-1].ok:
                break
            tries.append(solve_ik(self.chain, p, R, q0=start, seeds=1, w_rot=0.02))
        return tries

    def _tilted(self, xyz_mm, pitch_deg: float, want: float, q0) -> np.ndarray:
        """solve() for a gripper that isn't pointing straight down: through solve_task, steering its roll until the
        jaws' line is within half a degree of `want` (from DOWN degrees of pitch, where it's a heading on the
        table)."""
        x, y, z = (float(v) / 1000 for v in xyz_mm)
        steer = pitch_deg >= DOWN  # pointing down, the jaws' line is a heading the roll can turn
        found, why = [], ""
        for turned in (0.0, math.pi, -math.pi):  # the task roll's ways round: the same line, the jaws either way round
            roll, before = turned, None
            for _ in range(STEER_ROUNDS):
                res = solve_task(self.chain, x, y, z, math.radians(pitch_deg), roll, q0=q0)
                miss = line(want - self.jaws_heading(res.q)) if steer else 0.0
                if abs(miss) < 0.5:
                    break
                # The task's roll and the yaw IK may choose both turn the jaws, so a degree of roll turns their line by
                # a degree or less: the slope comes from the last two tries
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
        return min(found, key=lambda q: abs(q[ROLL] - q0[ROLL]))

    def move_to(self, xyz_mm, pitch_deg: float = 90.0, jaws_deg: float | None = None, seconds: float | None = None):
        """Move the tool point to xyz_mm, solved by solve(), through the driver's move_to. Returns its last reading."""
        return self.arm.move_to(self.solve(xyz_mm, pitch_deg, jaws_deg), seconds)

    def idle(self):
        """Wait in the idle pose, robot.yaml's hand.idle_deg (the L pose with the gripper pointing down), with the
        motors on. Only empty-handed: holding something, the arm waits where the pick left it. None without an idle
        pose. Unlike rest, the arm can't stay here with its motors off, so the end of a session parks it at rest."""
        if self.idle_q is None:
            return None
        if self.held:
            raise Refused(f"it holds the {self.held}, so it waits where it is instead of in the idle pose")
        return self.arm.move_to(self.idle_q, samples=self._samples("idle"))

    def home(self):
        """Back to where power_on found the arm, just inside its ranges."""
        if self.rest is None:
            raise Refused("there's no home yet: power the arm on first")
        return self.arm.move_to(self.rest, samples=self._samples("home"))

    def _desk_z(self, x: float, y: float) -> float:
        """The desk's height under (x, y) mm, mm in the arm's frame, as the driver's table has it."""
        return 1000 * self.arm.table.desk(x / 1000, y / 1000)

    def _floor_z(self, x: float, y: float, grip: bool = False) -> float:
        """The lowest the tool point may go over (x, y) mm, mm in the arm's frame, with FLOOR_MARGIN_MM to spare: over
        the driver's tool floor, or with `grip`, on the way down to a grip, over its GRIP_FLOOR."""
        floor = GRIP_FLOOR if grip else self.arm.tool_floor
        return 1000 * (self.arm.table.z(x / 1000, y / 1000) + floor) + FLOOR_MARGIN_MM

    def _top_z(self, thing: Thing) -> float:
        """Its top, mm in the arm's frame: its height over the desk, or for something lying flat its width, up to
        THICK_MM."""
        tall = thing.height_mm if thing.height_mm > 0 else min(thing.width_mm, THICK_MM)
        return self._desk_z(thing.x, thing.y) + tall

    def _solved(self, plan: list, toward: float | None = None, loose: bool = False, start=None) -> list:
        """Each (what, xyz, jaws) of a plan solved in turn, from where the arm is (or joint angles `start`), with the
        jaws one way round throughout: `toward` (a heading the jaw axis points within 90° of), or whichever the first
        step takes. Every pose before any moves. `loose` as for solve()."""
        q = np.array(self.arm.read().q if start is None else start, float)
        jaws = self.jaws_heading(q)
        steps = []
        for what, xyz, want in plan:
            q = self.solve(xyz, 90.0, jaws if want is None else want, q0=q, toward=toward, loose=loose)
            if steps and abs(math.degrees(q[ROLL] - steps[-1].q[ROLL])) > ROLL_STEP_DEG:
                raise Refused(f"the wrist roll would turn {math.degrees(q[ROLL] - steps[-1].q[ROLL]):+.0f}° from "
                              f"{steps[-1].what} to {what}, the jaws the same way round: IK went the long way")
            toward, jaws = self.jaws_toward(q), self.jaws_heading(q)
            steps.append(Step(what, xyz, jaws, q))
        return steps

    def toward_moving_jaw(self, q) -> np.ndarray:
        """The way from the fixed jaw to the moving one at joint angles q, a unit vector on the table."""
        v = (self.chain.fk(q)[:3, :3] @ self.moving_axis)[:2]
        return v / np.linalg.norm(v)

    def plan_pick(self, thing: Thing) -> list:
        """The poses a pick goes through, solved: over the thing, just over its top, down to it, and back up, with the
        fixed jaw coming down JAW_CLEAR_MM beside it. Of the two ways round the jaws could close across it, the one
        whose wrist roll is nearest where the arm is, of those whose tips get low enough on it. Refused when the moving
        jaw's side isn't set, the desk under it wasn't touched, or neither way round gets low enough and is in
        reach. Where locate gave a place to grip it (Thing.gripped), it's there, as wide as it is there."""
        thing = thing.gripped()
        if self.moving_sign is None:
            raise Refused("which side the moving jaw is on isn't set, so a pick can't put the fixed jaw beside the "
                          f"{thing.name}: set hand.moving_jaw in config/robot.yaml to left or right, looking out from "
                          "the base with the gripper pointing straight down")
        held = 1000 * self.arm.table.held(thing.x / 1000, thing.y / 1000)
        if held > HELD_MM:
            raise Refused(f"no touch measured the desk near the {thing.name}, so the floors take it up to "
                          f"{held:.0f} mm higher than it may be there, too rough to grip it near its middle: move it "
                          "nearer the touched desk tags, or touch one near it (desk_camera.py touch)")
        desk, top = self._desk_z(thing.x, thing.y), self._top_z(thing)
        grip_z = top - GRIP_DEPTH_MM if thing.height_mm > 0 else desk + GRIP_SHARE * (top - desk)  # or low on it
        under = desk - held  # the desk under it as the touches measured it: the lowest it may sit
        aside = max(0.0, thing.width_mm / 2 + JAW_CLEAR_MM - CLOSED_GAP_MM / 2)
        now = self.arm.read().q
        jaws = thing.jaws_deg if thing.jaws_deg is not None else self.jaws_heading(now)
        plans, why, thin = [], None, None
        for toward in (jaws, jaws + 180.0):  # the jaw axis pointing one way along the line, then the other
            t = math.radians(toward)
            u = self.moving_sign * np.array([math.cos(t), math.sin(t)])  # from the fixed jaw toward the moving one
            x, y = thing.x - aside * u[0], thing.y - aside * u[1]
            z = max(self._floor_z(x, y, grip=True), grip_z)
            if z > grip_z and z - under > THIN * (top - desk):  # the floor where the tips come down keeps them high
                beside = 1000 * self.arm.table.held(x / 1000, y / 1000) > 0.5
                thin = thin or Refused(
                    f"the {thing.name} is about {top - desk:.0f} mm {'tall' if thing.height_mm > 0 else 'thick'}, and "
                    f"the jaw tips go no lower than {z - under:.0f} mm over the desk there (robot.yaml's tool_floor_mm "
                    "and the table's margin" + (", held higher beside it, where no touch measured the desk" if beside
                                                else "") + "), so they'd close near its top and slip off")
                continue
            try:
                plans.append(self._hovering(x, y, max(z, top), lambda over: [
                    ("over it", over, jaws), ("just over it", (over[0], over[1], max(z, top) + JUST_OVER_MM), jaws),
                    ("down to it", (over[0], over[1], z), jaws)], toward=toward))
            except Refused as e:
                why = why or e
        if not plans:
            raise thin or why  # one way round too high on the thing says why, not the other's reach
        over, just, down = min(plans, key=lambda steps: abs(steps[0].q[ROLL] - now[ROLL]))
        return [over, just, down, Step("back up", over.xyz, over.jaws_deg, over.q)]  # the way it came down

    def plan_put(self, place: Thing, start=None) -> list:
        """The poses a put goes through, solved: over the place, down to just over its top, and back up. Of the two ways
        round the jaws could close along the place's line (or the line they hold the thing along, for a round place),
        each at its own highest hover, the one whose wrist roll is nearest where the arm is: a higher hover isn't worth
        turning what's held half a turn. `start`, joint angles, plans it from there instead, as for a put planned
        before the pick that comes first."""
        top = self._desk_z(place.x, place.y) + place.height_mm
        open_z = max(self._floor_z(place.x, place.y), top + PUT_ABOVE_MM)
        now = np.array(self.arm.read().q if start is None else start, float)
        jaws = place.jaws_deg if place.jaws_deg is not None else self.jaws_heading(now)
        plans, why = [], None
        for toward in (jaws, jaws + 180.0):
            try:
                plans.append(self._hovering(place.x, place.y, top, lambda over: [
                    ("over it", over, jaws), ("down to it", (place.x, place.y, open_z), jaws)], toward=toward,
                    start=now))
            except Refused as e:
                why = why or e
        if not plans:
            raise why
        over, down = min(plans, key=lambda steps: abs(steps[0].q[ROLL] - now[ROLL]))
        return [over, down, Step("back up", over.xyz, over.jaws_deg, over.q)]

    def tip_from_jaws(self, thing: Thing, q) -> float | None:
        """Where `thing`'s narrow end points (locate grips it off its middle toward that end), as degrees from the jaw
        axis at joint angles q; None when locate gripped it at its middle."""
        if thing.grip_x is None:
            return None
        end = math.degrees(math.atan2(thing.grip_y - thing.y, thing.grip_x - thing.x))
        return turn(end - self.jaws_toward(q))

    def plan_insert(self, place: Thing, held: Thing, start=None, tip: float | None = None) -> list:
        """The poses a put into a container goes through, solved: lift what's held LIFT_MM over the desk where it is,
        the jaws turned across the arm toward the container so it lies along the arm with its narrow end (`tip`,
        degrees from the jaw axis, as tip_from_jaws gave it at the pick) toward the base, or its wide end where the
        wrist roll's stop rules that out; turn the gripper up until it stands on end, that end down, over the
        container's opening, ON_END_ABOVE_MM over the rim; down until its lower end is ON_END_CLEAR_MM over the rim,
        then INSERT_MM under it; and back up. Level where the arm reaches that low, else up to the last of
        ON_END_TILTS down. `held` is the thing as locate found it, for its length and its
        grip; `start`, joint angles, plans from there instead of where the arm is. Refused, with nothing moved, when
        the arm can't stand it on end over the opening."""
        now = np.array(self.arm.read().q if start is None else start, float)
        b = math.atan2(place.y, place.x)
        out = np.array([math.cos(b), math.sin(b)])  # from the base through the container
        across = line(math.degrees(b) + 90.0)
        x, y, _, _, _ = pose_to_task(self.chain.fk(now))
        at = (1000 * x, 1000 * y, self._desk_z(1000 * x, 1000 * y) + LIFT_MM)
        off = 0.0 if held.grip_x is None else math.hypot(held.grip_x - held.x, held.grip_y - held.y)
        rim = self._desk_z(place.x, place.y) + place.height_mm
        # Which end goes in first: its narrow end, turned toward the base before the gripper turns up, unless the
        # wrist roll can't turn the jaws that way round (its stop); then its wide end. With no narrow end, either.
        ends = [("", None, held.length_mm / 2)] if tip is None else [
            (", its narrow end to go in first", math.degrees(b) + 180.0 - tip, held.length_mm / 2 - off),
            (", its wide end to go in first", math.degrees(b) - tip, held.length_mm / 2 + off)]
        why = None
        for end, toward, low in ends:  # low: from where the jaws hold it to the end that goes in
            try:
                lift = Step("lift, the jaws across the arm" + end, at, across,
                            self.solve(at, jaws_deg=across, q0=now, toward=toward))
            except Refused as e:
                why = why or e
                continue
            for tilt in ON_END_TILTS:
                p = math.radians(tilt)
                approach = np.array([math.cos(p) * out[0], math.cos(p) * out[1], -math.sin(p)])
                lower = np.array([-math.sin(p) * out[0], -math.sin(p) * out[1], -math.cos(p)])  # along it, to that end
                q, steps = lift.q, []
                try:
                    for what, end_z in (("on end, over it", rim + ON_END_ABOVE_MM),
                                        ("down over it", rim + ON_END_CLEAR_MM), ("into it", rim - INSERT_MM)):
                        # the tool point that puts that end over the opening's middle at end_z
                        xyz = tuple(np.array([place.x, place.y, end_z]) - low * lower + CAGE_MM * approach)
                        q = self.solve(xyz, pitch_deg=tilt, q0=q)
                        steps.append(Step(what, xyz, across, q))
                except Refused as e:
                    why = why or e
                    continue
                return [lift] + steps + [Step("back up", steps[0].xyz, across, steps[0].q)]
        raise Refused(f"the arm can't stand the {held.name} on end over the {place.name} at x {place.x:.0f}, "
                      f"y {place.y:.0f} mm ({why}): it can from about 28 cm out from its base, nearer straight out")

    def put_in(self, place: Thing, confirm=None, steps: list | None = None) -> dict:
        """Put what the jaws hold into a container, on end: through plan_insert's steps (planned now unless given),
        letting go once its lower end is under the rim, then back up and the jaws closed. The turns on end take
        ON_END_SLOW times as long as the caps allow. The jaws turn across the arm in a stop of their own, before it turns
        on end, unless robot.yaml's hand.roll_on_the_way turns them on the way over: the one try of that (Nov 3, 20:01)
        dropped the screwdriver as the gripper turned up, from a grip that may not have held it. The lift step is
        planned and checked either way. confirm as for pick_up, asked on end over the container."""
        if not self.held and not self.arm.jaw().holding:
            raise Refused("the jaws hold nothing to put in")
        if steps is None:
            if self.held_thing is None:
                raise Refused("what the jaws hold isn't known well enough to stand it on end: pick it up again first")
            steps = self.plan_insert(place, self.held_thing, tip=self.held_tip)
        lift, high, over, into, up = steps
        try:
            if not self.roll_on_the_way:
                self._move(lift, aim=False, slow=ON_END_SLOW)
            self._move(high, aim=False, slow=ON_END_SLOW)
            if confirm is not None:
                answer = confirm(f"On end over the {place.name}. Put it in?")
                if not answer:
                    return {"ok": False, "stopped": f"over the {place.name}, "
                            + ("as you asked" if answer is False else "since the session is ending"),
                            "steps": [s.summary() for s in (lift, high)]}
            self._move(over)
            self._move(into)
            self.arm.release(self.opened or 100.0, samples=self._samples("let go"))
        except Halted:
            if not self.arm.jaw().holding:
                self.held = None
            raise
        held, self.held, self.opened, self.held_thing, self.held_tip = self.held, None, None, None, None
        self._move(up, aim=False)
        self._close()
        return {"ok": True, "put": held, "on": place.name, "on_end": True, "steps": [s.summary() for s in steps]}

    def _hovering(self, x: float, y: float, top: float, plan, toward: float | None = None, start=None) -> list:
        """plan(over) solved, the jaws one way round as _solved has them (from `start`, as it takes it), for the highest
        of HOVER_MM over `top` at (x, y) mm that the arm can reach straight down. A put opens PUT_ABOVE_MM over the
        top, so its lowest hover goes no lower than that. Only when no hover can with poses IK converged on does it
        take one IK got within solve()'s checks of (loose), far out, rather than refuse. Refused, as the lowest hover
        was, when none can: something too tall to come down over straight down there says so in its height over the
        desk."""
        why = None
        for loose in (False, True):
            for hover in HOVER_MM:
                try:
                    return self._solved(plan((x, y, top + hover)), toward, loose, start)
                except Refused as e:
                    why = e
        raise why

    def _sagging(self, down: Step, thing: Thing) -> str:
        """'' when the arm, stopping as short of `down` as it stopped short of its last target (just over the thing,
        with about the same load), would still read SAG_SPARE_MM over the driver's cut for sagging under the floor of
        the way down to a grip (GRIP_FLOOR); else why it shouldn't go down. Only the joints whose shortfall lowers the
        tips count: the re-aim at the end of the way down takes up at once one that held them up, while one that sagged
        can stay put."""
        short = self.arm.stopped_short
        if short is None:
            return ""
        table = self.arm.table
        planned = lowest(self.chain, down.q, FLOOR_JOINTS, table)[0]
        q = np.array(down.q, float)
        for i, s in enumerate(short):
            one = np.array(down.q, float)
            one[i] -= s
            if lowest(self.chain, one, FLOOR_JOINTS, table)[0] < planned:
                q[i] -= s
        sagged = lowest(self.chain, q, FLOOR_JOINTS, table)[0]
        room = 1000 * (sagged - (GRIP_FLOOR - SAG))  # the cut on the way down to a grip
        if room >= SAG_SPARE_MM:
            return ""
        return (f"just over the {thing.name} the arm sagged enough that down at it, it would stop about "
                f"{1000 * (planned - sagged):.0f} mm under its plan, "
                + (f"only {room:.1f} mm over" if room >= 0 else f"{-room:.1f} mm under")
                + " where the driver cuts every motor for sagging, so it stopped there: nearer the base, the arm sags "
                  "less")

    def pick_up(self, thing: Thing, confirm=None, steps: list | None = None) -> dict:
        """Pick `thing` up: over it, open, just over its top, down, grip, back up, through `steps` if plan_pick has
        already given them. confirm(prompt), when given, is asked over the thing once the jaws are open, before they go
        down; False stops there, and so does None (the session is ending). Back up, it holds the thing as far as the
        jaws can tell: with "unconfirmed" when they closed all the way, which they do on nothing and on a loosely held
        handle alike.
        Refused (the driver's or IK's) before anything moves when a pose can't be reached, but a refused climb back up
        from the thing comes back as an error that says the arm is down there; Tripped, with the motors cut, as the
        driver says; Halted by the spoken stop, with `held` set if the jaws hold the thing by then."""
        if self.held:
            raise Refused(f"the jaws already hold the {self.held}: put it down first")
        steps = steps or self.plan_pick(thing)
        self.held_thing, self.held_tip = thing, self.tip_from_jaws(thing, steps[2].q)
        try:
            return self._pick(thing.gripped(), steps, confirm)
        except Halted:
            if self.arm.jaw().holding:
                self.held = thing.name
            raise

    def _pick(self, thing: Thing, steps: list, confirm) -> dict:
        over, just, down, up = steps
        want = self.opening_for(thing)
        self._move(over, aim=False)  # the way down to just over it re-aims instead
        opened = self.arm.release(want, samples=self._samples("open"))
        if opened.opening < OPEN_ENOUGH / 100 * want:  # jaws shut hard can stick as they start (Nov 4, 12:15): again
            opened = self.arm.release(want, samples=self._samples("open again"))
        if opened.opening < OPEN_ENOUGH / 100 * want:
            return {"ok": False, "error": f"the jaws stopped {opened.opening:.0f}% open over the {thing.name}, short "
                                          f"of the {want:.0f}% it needs: something's in their way, so they didn't go "
                                          "down", "steps": [over.summary()]}
        if confirm is not None:  # open, so whoever answers can see the fixed jaw is beside it, not over it
            answer = confirm(f"Over the {thing.name} with the jaws open: the one that didn't move should be beside it, "
                             "not over it. Go down and grip it?")
            if not answer:
                return {"ok": False, "stopped": f"over the {thing.name} with the jaws open, "
                        + ("as you asked" if answer is False else "since the session is ending"),
                        "steps": [over.summary()]}
        self._move(just)
        sagging = self._sagging(down, thing)
        if sagging:
            return {"ok": False, "error": sagging, "steps": [over.summary(), just.summary()]}
        self._move(down, grip=True)
        self.arm.grip(samples=self._samples("grip"))
        self.held, self.opened = thing.name, want  # shut all the way or not, they may hold it: see the module's notes
        try:
            self._move(up, aim=False)  # up from the thing, a few mm short is as good
        except Refused as e:  # down at the desk with the jaws shut: say so, not that nothing moved
            return {"ok": False, "holding": thing.name, "error": f"the jaws closed at the {thing.name} down at the "
                    f"desk, and the arm couldn't go back up ({e})", "steps": [s.summary() for s in (over, just, down)]}
        grip = self.arm.jaw()
        result = {"ok": True, "holding": thing.name, "jaws_open_percent": round(grip.opening),
                  "steps": [s.summary() for s in steps]}
        if not grip.holding:
            result["unconfirmed"] = (f"the jaws closed all the way, which they do on nothing and holding the "
                                     f"{thing.name} loosely alike: only a look at where it was can tell")
        return result

    def put_down(self, place: Thing, confirm=None, steps: list | None = None) -> dict:
        """Put what the jaws hold down on or in `place`: over it, down to just over its top, open, back up, through
        `steps` if plan_put has already given them. confirm as for pick_up, asked over the place before it goes down.
        Halted by the spoken stop, with `held` cleared if the jaws have let go by then."""
        if not self.held and not self.arm.jaw().holding:
            raise Refused("the jaws hold nothing to put down")
        try:
            return self._put(place, *(steps or self.plan_put(place)), confirm)
        except Halted:
            if not self.arm.jaw().holding:
                self.held = None
            raise

    def _put(self, place: Thing, over: Step, down: Step, up: Step, confirm) -> dict:
        self._move(over, aim=False)  # the way down to the place re-aims instead
        if confirm is not None:
            answer = confirm(f"Over the {place.name}. Go down and let go?")
            if not answer:
                return {"ok": False, "stopped": f"over the {place.name}, "
                        + ("as you asked" if answer is False else "since the session is ending"),
                        "steps": [over.summary()]}
        self._move(down)
        self.arm.release(self.opened or 100.0, samples=self._samples("let go"))  # as far as the pick opened them
        held, self.held, self.opened, self.held_thing, self.held_tip = self.held, None, None, None, None
        self._move(up, aim=False)
        self._close()
        return {"ok": True, "put": held, "on": place.name, "steps": [s.summary() for s in (over, down, up)]}
