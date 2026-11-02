"""Moves in mm on top of the arm driver, and the general moves the agent makes with them: picking a thing up and
putting it down.

The agent decides what goes where (wbc/agent/tools.py); this decides how. A target is a point in the arm's frame in mm,
with the gripper's pitch (90° points it straight down) and the line its jaws close along, solved from where the arm
is: straight down by kinematics.solve_ik, with the jaws' heading part of the rotation it asks for, and tilted by
kinematics.solve_task. Every limit stays in the driver: a move it refuses is refused here too, never forced, and
nothing here talks to the servos except through it.

pick_up and put_down take a thing as wbc/perception/desk.py's locate found it: its middle, its size, the heading of
its long axis and, for something standing up, its height. A pick goes straight down beside its middle with the jaws
closing across that axis: over it, open, just over its top, down, grip, and back up. Jaws that close on nothing, or come
up empty, are a miss, and they open again. A put goes over the place with what's held lined up with the place's long
axis, down to just over its top, opens, and goes back up. Every pose a pick or a put needs is solved before the first
one moves, so one the arm can't reach is refused with the arm where it was.

Heights come from the table the driver's floors use (table.py), the desk as the arm's touches measured it, not from
locate's camera, whose table is one height for the whole desk. Something lying flat is taken to be as thick as it is
wide, like a handle, and gripped halfway up; something standing up GRIP_DEPTH_MM under its top. Neither goes under the
driver's floor for the tool point, so with robot.yaml's 5 mm tool_floor_mm and the table's 2 mm margin the tips go no
lower than about 8 mm over the desk: a 16 mm handle is gripped at its middle, and anything they'd grip above THIN of
its height is refused, since they'd close near its top and slip off. That's judged where the tips come down beside it,
whose floor can be higher, against the desk under it as the touches measured it. So is a pick where no touch measured
the desk (table.py holds it high there).

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
close across the arm, along the wrist flex's axis. That's from a look at the jaws in the rest pose (Oct 29), where the
line between them runs down the arm's middle with the roll reading about 0°. The URDF has them a quarter turn from
there, so the jaws' line comes from the wrist flex's axis, not from the URDF's frame. The four-touch test (Oct 29,
22:34) put the tips 3.7 mm from the roll's axis, too near it to tell which way round they are.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .arm import FLOOR_JOINTS, RANGE_MARGIN, SAG, Halted, Refused
from .joints import ARM, NAMES
from .kinematics import pose_to_task, solve_ik, solve_task
from .table import PAN_AXIS
from .trajectory import lowest

HOVER_MM = (60.0, 45.0, 30.0)  # over a thing's top, where a pick or a put starts and ends: the first the arm can
                               # reach straight down, since nearer the base or further out it can't point down as high
FLOOR_MARGIN_MM = 1.0   # over the driver's tool floor that a pick goes down to, so its end isn't refused for a scrap
THICK_MM = 40.0         # the thickest something lying flat is taken to be, whatever its width
GRIP_DEPTH_MM = 15.0    # under its top that the tips go to on something standing up
CLOSED_GAP_MM = 5.0     # the closed tips' gap (Oct 28): the tool point sits half of it from the fixed jaw's face
JAW_CLEAR_MM = 5.0      # between the fixed jaw and the thing's side at the grip: the four-touch fit puts the closed
                        # tips' middle within about 2.4 mm (one sd) along the jaws' line (review, Oct 30)
JUST_OVER_MM = 5.0      # over the thing's top that a pick's last way down starts from: a short move, so its line in
                        # joint space can't swing the jaws into the thing (5.5 mm off the vertical over a 60 mm one)
SAG_SPARE_MM = 1.0      # that a pick keeps between where its grip would read, sagging as it did just over the thing,
                        # and the driver's sag cut
THIN = 2 / 3            # of its height over the desk that the tips have to get under for a pick to grip a thing
HELD_MM = 2.0           # a pick is refused where the table is held more than this over its touched line: the desk's
                        # height there isn't known well enough to grip near a thing's middle
ROLL_STEP_DEG = 30.0    # the most the wrist roll may turn from one step of a plan to the next, the jaws one way round
OPEN_ENOUGH = 90.0      # percent open the jaws have to reach before a pick goes down
PUT_ABOVE_MM = 30.0     # over the place's top that a put opens the jaws
ROUND = 1.3             # a thing whose length is under this many times its width has no long axis to grip across
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

    def home(self):
        """Back to where power_on found the arm, just inside its ranges."""
        if self.rest is None:
            raise Refused("there's no home yet: power the arm on first")
        return self.arm.move_to(self.rest)

    def _desk_z(self, x: float, y: float) -> float:
        """The desk's height under (x, y) mm, mm in the arm's frame, as the driver's table has it."""
        return 1000 * self.arm.table.desk(x / 1000, y / 1000)

    def _floor_z(self, x: float, y: float) -> float:
        """The lowest the tool point may go over (x, y) mm, mm in the arm's frame, with FLOOR_MARGIN_MM to spare."""
        return 1000 * (self.arm.table.z(x / 1000, y / 1000) + self.arm.tool_floor) + FLOOR_MARGIN_MM

    def _top_z(self, thing: Thing) -> float:
        """Its top, mm in the arm's frame: its height over the desk, or for something lying flat its width, up to
        THICK_MM."""
        tall = thing.height_mm if thing.height_mm > 0 else min(thing.width_mm, THICK_MM)
        return self._desk_z(thing.x, thing.y) + tall

    def _solved(self, plan: list, toward: float | None = None, loose: bool = False) -> list:
        """Each (what, xyz, jaws) of a plan solved in turn, from where the arm is, with the jaws one way round
        throughout: `toward` (a heading the jaw axis points within 90° of), or whichever the first step takes. Every
        pose before any moves. `loose` as for solve()."""
        q = np.array(self.arm.read().q, float)
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
        reach."""
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
        grip_z = top - GRIP_DEPTH_MM if thing.height_mm > 0 else (top + desk) / 2  # or halfway up
        under = desk - held  # the desk under it as the touches measured it: the lowest it may sit
        aside = max(0.0, thing.width_mm / 2 + JAW_CLEAR_MM - CLOSED_GAP_MM / 2)
        now = self.arm.read().q
        jaws = thing.jaws_deg if thing.jaws_deg is not None else self.jaws_heading(now)
        plans, why, thin = [], None, None
        for toward in (jaws, jaws + 180.0):  # the jaw axis pointing one way along the line, then the other
            t = math.radians(toward)
            u = self.moving_sign * np.array([math.cos(t), math.sin(t)])  # from the fixed jaw toward the moving one
            x, y = thing.x - aside * u[0], thing.y - aside * u[1]
            z = max(self._floor_z(x, y), grip_z)
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

    def plan_put(self, place: Thing) -> list:
        """The poses a put goes through, solved: over the place, down to just over its top, and back up. Of the two ways
        round the jaws could close along the place's line (or the line they hold the thing along, for a round place),
        each at its own highest hover, the one whose wrist roll is nearest where the arm is: a higher hover isn't worth
        turning what's held half a turn."""
        top = self._desk_z(place.x, place.y) + place.height_mm
        open_z = max(self._floor_z(place.x, place.y), top + PUT_ABOVE_MM)
        now = self.arm.read().q
        jaws = place.jaws_deg if place.jaws_deg is not None else self.jaws_heading(now)
        plans, why = [], None
        for toward in (jaws, jaws + 180.0):
            try:
                plans.append(self._hovering(place.x, place.y, top, lambda over: [
                    ("over it", over, jaws), ("down to it", (place.x, place.y, open_z), jaws)], toward=toward))
            except Refused as e:
                why = why or e
        if not plans:
            raise why
        over, down = min(plans, key=lambda steps: abs(steps[0].q[ROLL] - now[ROLL]))
        return [over, down, Step("back up", over.xyz, over.jaws_deg, over.q)]

    def _hovering(self, x: float, y: float, top: float, plan, toward: float | None = None) -> list:
        """plan(over) solved, the jaws one way round as _solved has them, for the highest of HOVER_MM over `top` at
        (x, y) mm that the arm can reach straight down. A put opens PUT_ABOVE_MM over the top, so its lowest hover goes
        no lower than that. Only when no hover can with poses IK converged on does it take one IK got within solve()'s
        checks of (loose), far out, rather than refuse. Refused, as the lowest hover was, when none can: something too
        tall to come down over straight down there says so in its height over the desk."""
        why = None
        for loose in (False, True):
            for hover in HOVER_MM:
                try:
                    return self._solved(plan((x, y, top + hover)), toward, loose)
                except Refused as e:
                    why = e
        raise why

    def _sagging(self, down: Step, thing: Thing) -> str:
        """'' when the arm, stopping as short of `down` as it stopped short of its last target (just over the thing,
        with about the same load), would still read SAG_SPARE_MM over the driver's cut for sagging under the tool
        point's floor; else why it shouldn't go down. Only the joints whose shortfall lowers the tips count: the re-aim
        at the end of the way down takes up at once one that held them up, while one that sagged can stay put."""
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
        room = 1000 * (sagged - (self.arm.tool_floor - SAG))
        if room >= SAG_SPARE_MM:
            return ""
        return (f"just over the {thing.name} the arm sagged enough that down at it, it would stop about "
                f"{1000 * (planned - sagged):.0f} mm under its plan, "
                + (f"only {room:.1f} mm over" if room >= 0 else f"{-room:.1f} mm under")
                + " where the driver cuts every motor for sagging, so it stopped there: nearer the base, the arm sags "
                  "less")

    def pick_up(self, thing: Thing, confirm=None) -> dict:
        """Pick `thing` up: over it, open, just over its top, down, grip, back up. confirm(prompt), when given, is asked
        over the thing once the jaws are open, before they go down; False stops there, and so does None (the session
        is ending).
        Refused (the driver's or IK's) before anything moves when a pose can't be reached, but a refused climb back up
        from the thing comes back as an error that says the arm is down there; Tripped, with the motors cut, as the
        driver says; Halted by the spoken stop, with `held` set if the jaws hold the thing by then."""
        if self.held:
            raise Refused(f"the jaws already hold the {self.held}: put it down first")
        steps = self.plan_pick(thing)
        try:
            return self._pick(thing, steps, confirm)
        except Halted:
            if self.arm.jaw().holding:
                self.held = thing.name
            raise

    def _pick(self, thing: Thing, steps: list, confirm) -> dict:
        over, just, down, up = steps
        self.arm.move_to(over.q)
        opened = self.arm.release(100.0)
        if opened.opening < OPEN_ENOUGH:
            return {"ok": False, "error": f"the jaws stopped {opened.opening:.0f}% open over the {thing.name}: "
                                          "something's in their way, so they didn't go down", "steps": [over.summary()]}
        if confirm is not None:  # open, so whoever answers can see the fixed jaw is beside it, not over it
            answer = confirm(f"Over the {thing.name} with the jaws open: the one that didn't move should be beside it, "
                             "not over it. Go down and grip it?")
            if not answer:
                return {"ok": False, "stopped": f"over the {thing.name} with the jaws open, "
                        + ("as you asked" if answer is False else "since the session is ending"),
                        "steps": [over.summary()]}
        self.arm.move_to(just.q)
        sagging = self._sagging(down, thing)
        if sagging:
            return {"ok": False, "error": sagging, "steps": [over.summary(), just.summary()]}
        self.arm.move_to(down.q)
        grip = self.arm.grip()
        if grip.holding:
            try:
                self.arm.move_to(up.q)
            except Refused as e:  # it's in the jaws down at the desk: say so, not that nothing moved
                self.held = thing.name
                return {"ok": False, "holding": thing.name, "error": f"the jaws hold the {thing.name} down at the "
                        f"desk: the arm couldn't go back up ({e})", "steps": [s.summary() for s in (over, just, down)]}
            grip = self.arm.jaw()
        if not grip.holding:
            try:
                self.arm.release(100.0)
                self.arm.move_to(up.q)
            except Refused as e:  # down at the desk with nothing in the jaws: say so, not that nothing moved
                return {"ok": False, "missed": True, "error": f"the jaws closed on nothing where the {thing.name} was, "
                        f"and the arm couldn't go back up ({e})", "steps": [s.summary() for s in (over, just, down)]}
            return {"ok": False, "missed": True, "error": f"the jaws closed on nothing where the {thing.name} was",
                    "steps": [s.summary() for s in steps]}
        self.held = thing.name
        return {"ok": True, "holding": thing.name, "jaws_open_percent": round(grip.opening),
                "steps": [s.summary() for s in steps]}

    def put_down(self, place: Thing, confirm=None) -> dict:
        """Put what the jaws hold down on or in `place`: over it, down to just over its top, open, back up. confirm as
        for pick_up, asked over the place before it goes down. Halted by the spoken stop, with `held` cleared if the
        jaws have let go by then."""
        if not self.held and not self.arm.jaw().holding:
            raise Refused("the jaws hold nothing to put down")
        try:
            return self._put(place, *self.plan_put(place), confirm)
        except Halted:
            if not self.arm.jaw().holding:
                self.held = None
            raise

    def _put(self, place: Thing, over: Step, down: Step, up: Step, confirm) -> dict:
        self.arm.move_to(over.q)
        if confirm is not None:
            answer = confirm(f"Over the {place.name}. Go down and let go?")
            if not answer:
                return {"ok": False, "stopped": f"over the {place.name}, "
                        + ("as you asked" if answer is False else "since the session is ending"),
                        "steps": [over.summary()]}
        self.arm.move_to(down.q)
        self.arm.release(100.0)
        self.arm.move_to(up.q)
        held, self.held = self.held, None
        return {"ok": True, "put": held, "on": place.name, "steps": [s.summary() for s in (over, down, up)]}
