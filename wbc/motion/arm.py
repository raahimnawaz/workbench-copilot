"""The arm driver: the one owner of the SO-101's servo bus, moving the five arm joints together in URDF angles.

The plan gives the bus one owner. Each tick of a move sends all five arm joints their goals in one sync write and
reads every joint's position and load back by sync read, so the joints move together and nothing else talks on the
half-duplex line in between. A lock serializes the bus, because the agent runs its tools on a thread pool. Angles are
the URDF's, through joints.py's map, so FK and IK see the arm as it is. Keep to one Arm per port: LeRobot opens the
port without locking it, so a second Arm, or another program on the port such as scripts/check_joint_map.py, would
interleave its packets with this one's.

It powers the arm on the way scripts/first_moves.py does, which brought all six joints up without a jump on Oct 27:
LeRobot's configure() with its final torque-on held back, this arm's P gains (the elbow at 32) and a torque cap on every
joint, then the joints one at a time from the base out, each watched for 0.3 s. Writing a joint's goal with its torque
off turns the torque on (Oct 29), so each goal is set to where its joint is, from a fresh reading, only as that joint's
turn comes: a hand or a sag can move it while the joints before it come up. A joint that moves more than 3° as its
torque comes on, or strays more than 5° from where it should be during a move, cuts every motor at once with one
broadcast write, which a servo flagging overload can't refuse. Those thresholds caught the elbow's runaway on Oct 27,
126 ms in, at -3.2°. The wrist and gripper servos' voltage readings glitch under their 4.0 V limit for single replies
while the supply holds 5.3 V (Oct 28-29), and LeRobot fails any reply that flags it, so a write or setting read whose
reply flags the input voltage is sent again.

Refusals come before anything is written. power_on refuses a joint past an end of its range or a servo at 65 °C (heat
is checked only there). move_to refuses a start past an end of a range, a target too near one, and a path that takes
the tool point or the elbow and wrist axes too near the table. A move that starts under one of those floors has to
climb back over it without going lower on the way, or only turn the base, so moves can't walk the arm down a little at
a time. Moves are quintic and peak at 30°/s on any joint and 80 mm/s at the tool point, and no goal goes more than 5°
from where its joint is. The floors hold for the plan, and joints within the 5° stray can put the tool point 17-41 mm
under it, so a move also cuts every motor when the arm itself goes 5 mm under a floor. After a power cut the servos
come back limp, with every goal 0 and no torque cap (Oct 28), so each move first checks that every powered joint still
has its torque on at its cap.

Every cut latches, whether it came from cut() on any thread, from a signal handler or from a trip: a move or power-on
running elsewhere stops at its next reading, and power_on refuses until clear_stop(), so a power-on queued behind a
stop can't turn the motors back on. Each cut is read back, and when it can't be confirmed the message says so, and to
switch the strip off. Ctrl-C while LeRobot awaits a reply leaves scservo_sdk's port marked busy, which fails every later
request, the cut included, so the driver clears that mark before it talks to the bus, as LeRobot's own disconnect does.
Python delivers Ctrl-C only to its main thread: a move there holds the arm where it is, and one on another thread stops
by cut().

LeRobot's robot.connect() is never called. With no calibration file under this macOS account it runs a full
recalibration that rewrites every servo's homing offset and limits, and it always ends by turning torque on with
whatever goals the servos hold (0 after a power-up). Only robot.bus.connect() runs, which opens the port and pings the
servos, with the calibration read from calibration_dir: from another account, the folder of the one that ran
lerobot-calibrate.

The wrist roll has no stop, and its reading wraps at ±180°, so it can't show how far its cable has wound. Moves keep it
inside the URDF's limits, at least 17° clear of the wrap, and power_on and move_to refuse a roll that starts outside
them. A gripper turned a whole turn by hand reads as if it hadn't been: turn it back before powering on.

Each joint stops short of its goal and holds there, by its load over its P gain: gravity and gearbox friction, 1-4° on
the first powered run (Oct 29), which reaching out puts at 11-16 mm at the tool point. So once a move's path is done,
move_to re-aims: it sets each goal past its target by how far its joint stopped short, ramped under the same speed caps
and with every check a move has, until each joint is within AIM of its target, for up to AIM_ROUNDS rounds.

The gripper moves on its own goal. release() opens it to a share of its range, and grip() closes it on what's between
its jaws and keeps squeezing, under its 24% cap, saying whether the jaws stopped on something. While it grips, jaws
closing further because what they held slipped out are following their goal, not straying, so a move doesn't cut the
arm for it, and hold() keeps the squeeze.

Not here yet: compliant holds, and the skills on top (the probe hold, handing things over).
"""
from __future__ import annotations

import math
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .joints import ARM, CALIBRATION_ID, JOINTS, NAMES, TICKS_PER_TURN, JointMap
from .kinematics import load_chain
from .trajectory import PEAK, joint_path, lowest, min_seconds

HZ = 50.0               # goal writes per second during a move
TORQUE_CAP = 50         # percent of full torque on the arm joints (each servo's Torque_Limit), as first_moves.py ran
GRIPPER_CAP = 24        # percent on the gripper: under the 25% where the overload protection configure() sets starts
MAX_JUMP = 3.0          # degrees a joint may move on its own as its torque comes on before every motor is cut
MAX_STRAY = 5.0         # degrees a joint may be from where it should be during a move before every motor is cut
MAX_STEP = 5.0          # degrees: no goal goes further than this from where its joint is (LeRobot's max_relative_target)
MAX_JOINT_SPEED = 30.0  # degrees per second any joint may reach at a move's peak
MAX_TIP_SPEED = 0.08    # metres per second the tool point may reach at a move's peak: the plan's 80 mm/s near the bench
RANGE_MARGIN = 2.0      # degrees inside each end of a joint's range that a move may aim for
TOOL_FLOOR = 0.010      # metres above the table the tool point has to stay
JOINT_FLOOR = 0.030     # metres above the table the elbow, wrist flex and wrist roll axes have to stay: their servos
                        # are about 25 mm thick
FLOOR_JOINTS = ("elbow_flex", "wrist_flex", "wrist_roll")
LEVEL = 1e-6            # metres two heights may differ and still count as level: the URDF's base axis isn't quite
                        # vertical, so turning the base alone moves the tool point up to about 0.4 µm
SAG = 0.005             # metres the arm itself may go under a floor during a move before every motor is cut: joints
                        # within MAX_STRAY can put the tool point 17-41 mm under the plan (review, Oct 29)
TEMP_WARN = 55          # °C, the median of TEMP_READS readings: power_on says so
TEMP_STOP = 65          # °C: power_on refuses. The shoulder lift's own limit is 70 °C, the others' 80
TEMP_READS = 5          # readings per median: single temperature readings glitch high on this arm (Oct 28)
WATCH = 0.3             # seconds each joint is watched after its torque comes on
SETTLE = 0.3            # seconds a move holds its last goal, still checking
AIM = 0.2               # degrees from its target each joint should end a move within: move_to re-aims until it does
AIM_ROUNDS = 3          # re-aims a move may take before it settles for where its joints stopped
MAX_OFFSET = 4.0        # degrees past its target a re-aim may set a goal: under MAX_STRAY, so a joint held that far
                        # short still sits clear of the stray cut
GRIPPER_SPEED = 60.0    # degrees per second the gripper's goal may reach at a release's peak: fully open in about 4 s.
                        # The jaw pushes about 2.4% of drive per degree it trails its goal, so under its 24% cap it
                        # tops out about 110°/s, 10° behind, where GRIP_LAG stops it; at 60°/s it trailed 4-6° (Oct 29)
GRIP_SPEED = 60.0       # and at a grip's, so the jaws meet what they close on without a knock
GRIP_LAG = 10.0         # degrees the jaw may trail its goal before release() stops it where it is: something's in the way
GRIP_CLEAR = 30         # raw ticks from its closed end the jaw has to start a grip at, as gripper_probe's squeeze (Oct 28)
EMPTY = 20              # raw ticks: a grip that ends nearer its closed end than this holds nothing. Closed on nothing, the
                        # jaws stop 0-12 ticks past it (1277-1290 against 1278, Oct 28)
MAX_SECONDS = 60.0      # the slowest move this takes: a longer one is more likely a mistake in units
GAIN_RANGE = (8, 64)    # P gains it takes. LeRobot's configure() sets every joint to 16; the elbow needs 32
CUT_TRIES = 3           # times a cut is sent and every servo's torque read back before it counts as unconfirmed
VOLTAGE_TRIES = 3       # times a write or setting read goes out while its reply flags the input voltage: single readings
                        # on the wrist and gripper servos dip under their 4.0 V limit, the gripper's about every 4 s at
                        # rest, while the supply holds 5.3 V (Oct 29, 15:36)
STALE = 0.05            # seconds the reply to an interrupted request may take to arrive: three of the 16 ms USB latency
                        # scservo_sdk allows for (LATENCY_TIMER)
TICKS_PER_DEGREE = TICKS_PER_TURN / 360
STEP_TICKS = int(MAX_STEP * TICKS_PER_DEGREE)


class Tripped(Exception):
    """Something went wrong with the arm powered, a joint moving when or where it shouldn't or the bus failing, and
    every motor was cut. The message says so when the cut couldn't be confirmed."""


class Refused(ValueError):
    """Refused before anything was written or moved."""


@dataclass
class Reading:
    t: float        # seconds, on the arm's clock
    raw: dict       # {joint: ticks} for all six
    q: np.ndarray   # the five arm joints' URDF angles in radians, in ARM order
    load: dict      # {joint: per mille of full drive, signed by which way it pushes} for all six


@dataclass
class Grip:
    opening: float  # percent of the jaws' calibrated range: 0 at their closed end, 100 fully open
    gap: int        # raw ticks the jaw is from its closed end
    load: int       # per mille of full drive the gripper pushes with, signed as the servo reports it
    holding: bool   # it's gripping, and the jaws stopped EMPTY or more short of closed: on something


def listed(joints) -> str:
    return ", ".join(NAMES[m] for m in joints)


def failure(e: Exception) -> str:
    """What went wrong, for a message: the bus, or anything else by its type."""
    return "the bus failed" if isinstance(e, (ConnectionError, OSError)) else f"the driver hit a {type(e).__name__}"


def glitch(e: Exception) -> bool:
    """LeRobot's error for a reply whose status flags the input voltage. scservo_sdk names only the first flag it
    finds, and voltage first, so another flag set with it shows on the next reply."""
    return isinstance(e, RuntimeError) and "Input voltage error" in str(e)


def follower(port: str, robot_id: str = CALIBRATION_ID, calibration_dir=None):
    """LeRobot's SO-101 follower on `port`, built but not connected, with its calibration from calibration_dir
    (LeRobot's own folder for this macOS account when None). Its torque-off on disconnect is off: it would also unlock
    every servo's EEPROM, so the Arm cuts the motors itself. RuntimeError if LeRobot isn't installed."""
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError:
        raise RuntimeError("LeRobot is missing: uv pip install -e '.[lerobot]'") from None
    folder = None if calibration_dir is None else Path(calibration_dir).expanduser()
    return SO101Follower(SO101FollowerConfig(port=port, id=robot_id, calibration_dir=folder,
                                             disable_torque_on_disconnect=False))


def free_port(bus) -> bool:
    """Clear the busy mark an interrupted request leaves on LeRobot's port. scservo_sdk marks the port busy as a
    request goes out and clears the mark only once the reply is in (protocol_packet_handler.py), so Ctrl-C or a serial
    error while a reply is awaited makes every later request fail with "Port is in use!", the cut included. LeRobot's
    own disconnect clears it the same way. True if it was set: that reply may still be on its way, so drain() before
    the next read. Only with the bus lock held, when no request is in flight."""
    port = getattr(bus, "port_handler", None)
    if port is None or not getattr(port, "is_using", False):
        return False
    port.is_using = False
    return True


def drain(bus, sleep=time.sleep) -> None:
    """Let the reply to an interrupted request arrive, then throw away whatever came in, so the next read can't take it
    for its own: a late Present_Load reply read as positions would send the arm to them."""
    sleep(STALE)
    try:
        bus.port_handler.ser.reset_input_buffer()
    except Exception:  # noqa: BLE001 - no open serial port to flush
        pass


def cut_torque(robot) -> bool:
    """Every motor off at once. This is one broadcast write that servos don't answer, so a servo flagging overload
    can't refuse it the way it makes LeRobot's one-servo writes fail. Sent twice in case one is lost, after freeing a
    port that an interrupted request left busy. True if it was busy: drain() before reading."""
    busy = free_port(robot.bus)
    for _ in range(2):
        try:
            robot.bus.sync_write("Torque_Enable", {m: 0 for m in JOINTS}, normalize=False)
        except Exception:  # noqa: BLE001 - nothing more to try over the bus
            pass
    return busy


class Arm:
    def __init__(self, robot, joints: JointMap, chain, gains: dict | None = None, cap: int = TORQUE_CAP, say=print,
                 clock=time.monotonic, sleep=time.sleep):
        """robot: LeRobot's follower with its bus connected (open() builds one). joints and chain: the joint map and
        the IK chain. gains: P gains written over LeRobot's 16 at each power-on, {joint: P}. cap: the arm joints'
        torque cap, 10-100 percent; the gripper's stays at most GRIPPER_CAP."""
        if not 10 <= cap <= 100:
            raise ValueError(f"the torque cap has to be from 10 to 100 percent, not {cap}")
        gains = dict(gains or {})
        for m, p in gains.items():
            if m not in JOINTS or not isinstance(p, int) or not GAIN_RANGE[0] <= p <= GAIN_RANGE[1]:
                raise ValueError(f"a P gain needs a joint from {', '.join(JOINTS)} and a whole number from "
                                 f"{GAIN_RANGE[0]} to {GAIN_RANGE[1]}, not {m}: {p!r}")
        self.robot, self.joints, self.chain = robot, joints, chain
        self.gains, self.cap = gains, cap
        self.caps = {m: 10 * (min(cap, GRIPPER_CAP) if m == "gripper" else cap) for m in JOINTS}  # Torque_Limit
        self.say, self.clock, self.sleep = say, clock, sleep
        self.powered = set()
        self._jaw_goal, self._gripping = None, False  # the gripper's goal in raw ticks, and whether grip() set it
        self._limits = None  # each servo's Torque_Limit before the first power-on, for close() to put back
        self._lock = threading.RLock()
        self._stopped = threading.Event()  # set by every cut, cleared only by clear_stop()
        self._closed = False

    @classmethod
    def open(cls, port: str, robot_cfg: dict, root, robot_id: str = CALIBRATION_ID, calibration_dir=None,
             make_robot=None, **kwargs) -> Arm:
        """Connect to the arm on `port` without moving it or writing anything: LeRobot's follower with its bus
        connected and nothing else, the calibration from calibration_dir, and the joint map, chain and gains from
        config/robot.yaml's contents. Refused if there's no calibration file or the servos don't match it, and the
        port is closed again on any failure. kwargs go to Arm()."""
        robot = (make_robot or follower)(port, robot_id, calibration_dir)
        try:
            robot.bus.connect()
        except BaseException:
            try:  # LeRobot leaves the port open after a failed handshake
                robot.bus.port_handler.closePort()
            except Exception:  # noqa: BLE001
                pass
            raise
        try:
            if not robot.calibration:
                where = getattr(robot, "calibration_fpath", None) or f"{robot_id}.json"
                raise Refused(f"there's no calibration file at {where}. lerobot-calibrate saves it under the macOS "
                              "account that ran it, so from another account pass that account's folder as "
                              "calibration_dir")
            if not robot.bus.is_calibrated:
                raise Refused("the servos' calibration doesn't match the file. Check that the file is this arm's, "
                              "and run lerobot-calibrate again if it is")
            chain = load_chain(robot_cfg, Path(root))
            joints = JointMap.from_config(robot_cfg, robot.calibration, chain)
            return cls(robot, joints, chain, gains=robot_cfg.get("gains"), **kwargs)
        except BaseException:
            try:
                robot.bus.disconnect(disable_torque=False)
            except Exception:  # noqa: BLE001
                pass
            raise

    def read(self) -> Reading:
        """Where every joint is and how hard it pushes: one sync read of Present_Position and one of Present_Load,
        raw."""
        with self._lock:
            self._free()
            return self._read()

    def check(self, joints=JOINTS) -> None:
        """Refused if any of `joints` is past where moves may take it (an end of its calibrated range, or the URDF's
        limits for the wrist roll), or a servo's median temperature over TEMP_READS readings is TEMP_STOP or more.
        Says so at TEMP_WARN. power_on runs it; heat isn't checked again during moves."""
        with self._lock:
            bus = self.robot.bus
            self._free()
            try:
                raw = bus.sync_read("Present_Position", normalize=False, num_retry=2)
                temps = [bus.sync_read("Present_Temperature", normalize=False, num_retry=2)
                         for _ in range(TEMP_READS)]
            except ConnectionError as e:
                raise Refused(f"the servos didn't answer ({e}). Is their power on?") from e
        where, turn = self._past(raw, joints)
        if where:
            raise Refused(f"past the end of its range: {where}. Move it in a little, toward the middle"
                          + (f". {turn}" if turn else ""))
        heat = {m: statistics.median(t[m] for t in temps) for m in JOINTS if m in joints}
        hot = {m: c for m, c in heat.items() if c >= TEMP_STOP}
        if hot:
            raise Refused("too hot to power on: " + ", ".join(f"the {NAMES[m]} at {c:.0f} °C" for m, c in hot.items())
                          + f" (the stop is {TEMP_STOP} °C). Let it cool first")
        warm = {m: c for m, c in heat.items() if c >= TEMP_WARN}
        if warm:
            self.say("Warm: " + ", ".join(f"the {NAMES[m]} at {c:.0f} °C" for m, c in warm.items())
                     + f". Power-on stops at {TEMP_STOP} °C.")

    def power_on(self, joints=JOINTS, samples: list | None = None) -> dict:
        """Turn torque on for `joints`, one at a time from the base out, without the arm moving, and return how far
        each shifted as it came on, in degrees.

        A servo drives to whatever its goal register holds the moment torque comes on, and after a power-up that's
        0, and writing its goal with the torque off turns the torque on (Oct 29). So this runs LeRobot's configure()
        with its final torque-on held back, sets the gains over the 16 it gives every joint and caps the torque, all
        before any torque; then, one joint at a time, sets its goal to where it is now from a fresh reading, which
        brings it on, and confirms its torque-on. The other joints' goals aren't written, so they stay off. Refused,
        with nothing written, after a cut until clear_stop(), if torque is already on anywhere (configure() turns every
        motor off first, which would drop a held arm), or if check() refuses. RuntimeError, with torque still off, if
        LeRobot's setup fails or a gain or a cap doesn't read back. Tripped, with every motor cut, if a joint's goal
        doesn't read back, it moves more than MAX_JUMP as it comes on, a cut arrives from elsewhere, or the bus fails.
        A reply that flags the input voltage is sent again (_flagged). `samples`, when given, gets every reading of
        the watch."""
        unknown = [m for m in joints if m not in JOINTS]
        if unknown or not joints:
            raise ValueError(f"power_on takes joints from {', '.join(JOINTS)}, not {list(joints)}")
        joints = [m for m in JOINTS if m in joints]
        samples = [] if samples is None else samples
        with self._lock:
            self._refuse_if_stopped()
            bus = self.robot.bus
            self._free()
            try:
                on = bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
            except ConnectionError as e:
                raise Refused(f"the servos didn't answer ({e}). Is their power on?") from e
            if any(on.values()):
                raise Refused(f"torque is already on for the {listed(m for m in JOINTS if on[m])}. LeRobot's setup "
                              "turns every motor off before it starts, so hold the arm, cut() the motors and "
                              "clear_stop() first")
            self.powered = set()
            self._jaw_goal, self._gripping = None, False
            self.check(joints)
            if self._limits is None:
                self._limits = dict(bus.sync_read("Torque_Limit", normalize=False, num_retry=2))
            enable = bus.enable_torque
            bus.enable_torque = lambda *args, **kwargs: None  # configure() ends by turning torque on: not yet
            try:
                self._flagged(self.robot.configure)  # settings only, so it can run again
            finally:
                del bus.enable_torque  # back to the bus's own method
            for m, p in self.gains.items():  # EEPROM, which configure() leaves unlocked
                self._flagged(bus.write, "P_Coefficient", m, p, normalize=False, num_retry=2)
                if self._flagged(bus.read, "P_Coefficient", m, normalize=False, num_retry=2) != p:
                    raise RuntimeError(f"the {NAMES[m]}'s gain didn't take, so torque stayed off")
            for m in JOINTS:
                self._flagged(bus.write, "Torque_Limit", m, self.caps[m], normalize=False, num_retry=2)
            caps = bus.sync_read("Torque_Limit", normalize=False, num_retry=2)
            wrong = [m for m in JOINTS if caps[m] != self.caps[m]]
            if wrong:
                raise RuntimeError(f"the torque cap didn't take on the {listed(wrong)}, so torque stayed off")
            shift, still = {m: 0.0 for m in joints}, {}
            for joint in joints:
                self._stopping(f"the {NAMES[joint]} was coming on")
                still[joint] = self._hold_here(joint)  # its goal write brings it on
                try:
                    self._flagged(enable, joint, num_retry=2)
                except KeyboardInterrupt:
                    raise  # Ctrl-C: the joints already on hold where they are
                except Exception as e:  # noqa: BLE001 - a servo flagging an error, or the bus
                    how = (f"flagged an error as its torque came on ({e})" if isinstance(e, RuntimeError) else
                           f"didn't confirm its torque-on ({e})" if isinstance(e, (ConnectionError, OSError)) else
                           f"couldn't be turned on ({type(e).__name__}: {e})")
                    raise self._trip(f"the {NAMES[joint]} {how}") from e
                self.powered.add(joint)
                if joint == "gripper":
                    self._jaw_goal = still[joint]
                self._watch(joint, still, shift, samples)
            return shift

    def move_to(self, q, seconds: float | None = None, samples: list | None = None, aim: bool = True) -> Reading:
        """Move the five arm joints to URDF angles q (radians, in ARM order) together along one quintic, taking at least
        `seconds` and as long as the speed caps need, then hold there SETTLE seconds. Then, unless `aim` is False,
        re-aim every joint that stopped more than AIM short of its target (see _reaim). Returns the last reading.

        Refused, with nothing written, after a cut until clear_stop(), and unless all five are powered with their
        torque still on at their caps, each starts inside where moves may take it, each target is RANGE_MARGIN inside
        its joint's range, and the path keeps the tool point TOOL_FLOOR and the elbow and wrist axes JOINT_FLOOR above
        the table. From a start under a floor, the path may not go lower than the start, and has to end over the floor
        unless it only turns the base. Tripped, with every motor cut, if a joint strays MAX_STRAY from where it should
        be, the gripper strays MAX_STRAY from its own goal, the arm goes SAG under a floor, a cut arrives from
        elsewhere, or the bus fails. Ctrl-C holds the arm where it is and re-raises. `samples`, when given, gets one
        entry a tick: t, goal and q in degrees, and every joint's load."""
        target = np.asarray(q, float)
        if target.shape != (len(ARM),) or not np.all(np.isfinite(target)):
            raise Refused(f"a move needs an angle for each of the {len(ARM)} arm joints, not {q!r}")
        if seconds is not None and not 0 <= seconds <= MAX_SECONDS:
            raise Refused(f"a move takes from 0 to {MAX_SECONDS:g} seconds, not {seconds!r}")
        with self._lock:
            self._refuse_if_stopped()
            missing = [m for m in ARM if m not in self.powered]
            if missing:
                raise Refused(f"the {listed(missing)} isn't powered: power the arm on first")
            self._free()
            start = self._ready("the move")
            where, turn = self._past(start.raw, ARM)
            if where:
                raise Refused(f"the arm starts past the end of its range: {where}. Hold it, cut() the motors, move it "
                              "back in by hand, and power it on again" + (f". {turn}" if turn else ""))
            for i, m in enumerate(ARM):
                lo, hi = (math.degrees(v) for v in self.joints.limits(m))
                lo, hi = lo + RANGE_MARGIN, hi - RANGE_MARGIN
                if not lo <= math.degrees(target[i]) <= hi:
                    raise Refused(f"the {NAMES[m]}'s target, {math.degrees(target[i]):+.1f}°, is outside where moves "
                                  f"may take it, {lo:+.1f}° to {hi:+.1f}°")
            fastest = min_seconds(start.q, target, math.radians(MAX_JOINT_SPEED), self.chain, MAX_TIP_SPEED)
            path = joint_path(start.q, target, max(seconds or 0.0, fastest, 1 / HZ), HZ)
            floors = self._floors(start, target, path)
            reading = self._run(path, start, floors, samples)
            return self._reaim(target, start, floors, reading, samples) if aim else reading

    def hold(self) -> None:
        """Every powered joint holds where it is now: its goal set to its present position, in one sync write. A
        gripper that's gripping keeps its goal at the closed end, so it doesn't let go of what it holds."""
        with self._lock:
            if not self.powered:
                return
            bus = self.robot.bus
            self._free()
            present = bus.sync_read("Present_Position", normalize=False, num_retry=2)
            here = {m: present[m] for m in JOINTS if m in self.powered and not (m == "gripper" and self._gripping)}
            bus.sync_write("Goal_Position", here, normalize=False)
            if "gripper" in here:
                self._jaw_goal = here["gripper"]

    def grip(self, samples: list | None = None) -> Grip:
        """Close the gripper on what's between its jaws, its goal ramped to the closed end at GRIP_SPEED, and keep it
        there: the jaws press with the gripper's P gain times how far short of closed they stopped, up to its cap.
        Returns where they stopped and whether that's on something (Grip.holding).

        Refused, with nothing written, after a cut until clear_stop(), unless the gripper is powered with every powered
        joint's torque still on at its cap, and when the jaws start within GRIP_CLEAR of closed: open them around the
        object first. Tripped, with every motor cut, if an arm joint moves MAX_STRAY from where it was or under a
        floor, the jaws open MAX_STRAY instead of closing, a cut arrives from elsewhere, or the bus fails. Ctrl-C holds
        the arm and stops the jaws where they are, squeezing nothing, in case it was a finger they were closing on.
        `samples` gets one entry a tick: t, the gripper's goal and position in raw ticks, and its load."""
        with self._lock:
            start = self._ready_gripper("the grip")
            closed, now = self._closed_end(), start.raw["gripper"]
            if now - closed < GRIP_CLEAR:
                raise Refused(f"the jaws are {now - closed} ticks from closed, under the {GRIP_CLEAR} a grip starts from: "
                              "open them around what they should hold first")
            self._jaw_goal, self._gripping = closed, True
            return self._grip(self._jaw(self._jaw_path(now, closed, GRIP_SPEED), start, samples, closing=True))

    def release(self, opening: float = 100.0, samples: list | None = None) -> Grip:
        """Open the gripper to `opening` percent of its range (0 closed, 100 fully open), its goal ramped there at
        GRIPPER_SPEED, letting go of whatever it held, and hold it there. A jaw that trails its goal by GRIP_LAG is
        stopped where it is, and says so: something's in the way. Either way it returns once the jaw has had SETTLE
        seconds to come to rest. Returns where it ended.

        Refused, with nothing written, for an opening outside 0-100, after a cut until clear_stop(), and unless the
        gripper is powered with every powered joint's torque still on at its cap. Tripped, with every motor cut, if an
        arm joint moves MAX_STRAY from where it was or under a floor, the jaws move MAX_STRAY the wrong way, a cut
        arrives from elsewhere, or the bus fails. Ctrl-C holds the arm and the jaws where they are."""
        if not isinstance(opening, (int, float)) or not 0 <= opening <= 100:
            raise Refused(f"the gripper opens from 0 to 100 percent, not {opening!r}")
        with self._lock:
            start = self._ready_gripper("the release")
            s = self.joints.servos["gripper"]
            goal = round(s.range_min + opening / 100 * (s.range_max - s.range_min))
            self._jaw_goal, self._gripping = goal, False
            path = self._jaw_path(start.raw["gripper"], goal, GRIPPER_SPEED)
            return self._grip(self._jaw(path, start, samples, closing=False))

    def jaw(self) -> Grip:
        """Where the jaws are, how hard the gripper pushes, and whether it's holding something, read only. A grip
        whose object slipped out reads not holding: the jaws closed on to their closed end."""
        with self._lock:
            self._free()
            return self._grip(self._read())

    def cut(self) -> bool:
        """Every motor off at once, and the arm drops. Safe from any thread and from a signal handler: a move or
        power-on running elsewhere stops at its next reading, cuts too, and raises Tripped. The stop latches, so
        power_on refuses until clear_stop(). True once every servo reads its torque back off; False, after saying so,
        if that couldn't be confirmed, and then only the strip is sure."""
        self._stopped.set()  # before the lock: a move holding it sees this at its next tick and lets go
        with self._lock:
            doubt = self._cut()
        if doubt:
            self.say(f"Every motor was told to cut, but {doubt}. Switch the servo power off at the strip.")
        return not doubt

    def clear_stop(self) -> None:
        """Let power_on() run again after a cut. Every cut, whether from cut(), another thread or a trip, stays latched
        until this, so call it only once someone has checked the arm."""
        with self._lock:  # not while a cut from another thread is still on its way to the bus
            self._stopped.clear()

    def close(self) -> None:
        """Cut every motor, stopping a move running on another thread at its next tick first. Then, once every servo
        reads its torque off, put each servo's torque cap back as it was before the first power-on, lock every servo's
        EEPROM again (LeRobot's setup leaves it unlocked), and close the port without LeRobot's own torque-off, which
        would unlock it. Best effort, and it says so when the cut couldn't be confirmed. The arm drops when the motors
        let go, so hold it first."""
        self._stopped.set()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            doubt = self._cut()
            bus = self.robot.bus
            if doubt:
                self.say(f"Every motor was told to cut, but {doubt}, so the torque caps stay as they are. Switch the "
                         "servo power off at the strip.")
            elif self._limits:
                for m, value in self._limits.items():
                    try:
                        self._flagged(bus.write, "Torque_Limit", m, value, normalize=False, num_retry=2)
                    except Exception:  # noqa: BLE001 - Torque_Limit is in RAM, so a power cycle restores it anyway
                        pass
            if self._limits is not None:  # this Arm ran LeRobot's setup, which unlocks every servo
                for m in JOINTS:
                    try:
                        self._flagged(bus.write, "Lock", m, 1, normalize=False, num_retry=2)
                    except Exception:  # noqa: BLE001 - a power cycle locks them again too (Oct 28)
                        pass
            try:
                bus.disconnect(disable_torque=False)
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> Arm:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        """Dropped without close(): cut every motor anyway, best effort. follower() turns LeRobot's own clean-up off,
        because it would also unlock every servo's EEPROM."""
        if not getattr(self, "_closed", True):
            try:
                cut_torque(self.robot)
            except Exception:  # noqa: BLE001
                pass

    def _free(self) -> None:
        """Before talking to the bus: clear what an interrupted request left behind, and drain its reply."""
        if free_port(self.robot.bus):
            drain(self.robot.bus, self.sleep)

    def _read(self) -> Reading:
        bus = self.robot.bus
        raw = bus.sync_read("Present_Position", normalize=False, num_retry=2)
        load = bus.sync_read("Present_Load", normalize=False, num_retry=2)
        return Reading(self.clock(), dict(raw), self.joints.q(raw), dict(load))

    def _cut(self) -> str | None:
        """Cut every motor, latch the stop, forget what's powered, and read every servo's torque back, sending the cut
        again while one still reads on. None once all six read off, otherwise what's in doubt. The bus lock is held."""
        self._stopped.set()
        self.powered = set()
        self._jaw_goal, self._gripping = None, False
        doubt = "nothing confirmed it"
        for _ in range(CUT_TRIES):
            if cut_torque(self.robot):
                drain(self.robot.bus, self.sleep)
            try:
                on = self.robot.bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
            except Exception as e:  # noqa: BLE001
                doubt = f"the servos didn't answer to confirm it ({e})"
                continue
            still = [m for m in JOINTS if on.get(m)]
            if not still:
                return None
            doubt = f"the {listed(still)} still reads its torque on"
        return doubt

    def _trip(self, what: str) -> Tripped:
        """Cut every motor, and the Tripped to raise for `what` went wrong, saying whether the cut was confirmed."""
        doubt = self._cut()
        if doubt:
            return Tripped(f"{what}. Every motor was told to cut, but {doubt}: switch the servo power off at the strip")
        return Tripped(f"{what}, so every motor was cut")

    def _refuse_if_stopped(self) -> None:
        if self._stopped.is_set():
            raise Refused("the arm is stopped: its motors were cut, and they stay off until clear_stop(), once "
                          "someone has checked the arm")

    def _stopping(self, during: str) -> None:
        """Tripped, after cutting every motor again, once a cut has latched (from another thread or a signal
        handler)."""
        if self._stopped.is_set():
            raise self._trip(f"stopped while {during}")

    def _ready(self, during: str) -> Reading:
        """Where the arm is, once every powered joint reads its torque still on at its cap. Refused, with nothing
        written, when one doesn't (a servo that reset on its own, or a power cut), and Tripped if the bus fails. The
        bus lock is held."""
        bus = self.robot.bus
        try:
            on = bus.sync_read("Torque_Enable", normalize=False, num_retry=2)
            caps = bus.sync_read("Torque_Limit", normalize=False, num_retry=2)
            start = self._read()
        except Exception as e:  # noqa: BLE001
            raise self._trip(f"{failure(e)} before {during} ({e})") from e
        lost = [f"the {NAMES[m]}'s torque is off" if not on[m] else
                f"the {NAMES[m]}'s cap reads {caps[m] / 10:g}%, not {self.caps[m] / 10:g}%"
                for m in JOINTS if m in self.powered and (not on[m] or caps[m] != self.caps[m])]
        if lost:
            self.powered = {m for m in self.powered if on[m]}
            if self.powered:  # one servo reset on its own: the rest still hold
                raise Refused(f"the arm was partly reset ({'; '.join(lost)}). Hold it, cut() the motors and "
                              "clear_stop(), then power it on again")
            raise Refused(f"the arm lost power or was reset ({'; '.join(lost)}): power it on again")
        return start

    def _ready_gripper(self, during: str) -> Reading:
        """_ready() for grip() and release(): refused first after a cut until clear_stop(), and unless the gripper is
        powered. The bus lock is held."""
        self._refuse_if_stopped()
        if "gripper" not in self.powered:
            raise Refused("the gripper isn't powered: power the arm on first")
        self._free()
        return self._ready(during)

    def _past(self, raw: dict, joints) -> tuple[str, str]:
        """For each of `joints` past where moves may take it, 'the wrist flex by 2.2°', joined, and which way to turn a
        wrist roll back by hand, or ('', '') when none is."""
        past = {m: v for m, v in self.joints.past_limits(raw).items() if m in joints}
        where = ", ".join(f"the {NAMES[m]} by {v:.1f}°" for m, v in past.items())
        if "wrist_roll" not in past:
            return where, ""
        way = "clockwise" if self.joints.angle("wrist_roll", raw["wrist_roll"]) > 0 else "counterclockwise"
        return where, (f"Turn the gripper {way}, looking from the wrist toward the jaw tips, unless it got there by "
                       f"turning more than half a turn {way}, which its reading can't show: the wrist roll's limit is "
                       "its cable")

    def _hold_here(self, joint: str) -> int:
        """Set `joint`'s goal to where it is now, which turns its torque on, and return that, in raw ticks. Tripped if
        the bus fails or the goal doesn't read back."""
        bus = self.robot.bus
        try:
            now = bus.sync_read("Present_Position", normalize=False, num_retry=2)[joint]
            self._flagged(bus.write, "Goal_Position", joint, now, normalize=False, num_retry=2)
            goal = bus.sync_read("Goal_Position", normalize=False, num_retry=2)[joint]
        except Exception as e:  # noqa: BLE001
            raise self._trip(f"{failure(e)} as the {NAMES[joint]} came on ({e})") from e
        if goal != now:
            raise self._trip(f"the {NAMES[joint]}'s hold-here goal didn't take as it came on")
        return now

    def _flagged(self, call, *args, **kwargs):
        """call(*args, **kwargs), a write or setting read that's safe to repeat, sent again while its reply flags the
        input voltage, up to VOLTAGE_TRIES times in all: on these servos that flag is a reading glitch (glitch()). The
        bus lock is held."""
        for n in range(1, VOLTAGE_TRIES + 1):
            try:
                return call(*args, **kwargs)
            except RuntimeError as e:
                if n == VOLTAGE_TRIES or not glitch(e):
                    raise

    def _floors(self, start: Reading, target: np.ndarray, path: np.ndarray) -> tuple[float, float]:
        """The heights the tool point and the floor axes may not go under during this move, in metres: their floors,
        or where they start when that's lower. Refused if the path, in the whole ticks its goals go out in, goes under
        one, or starts under a floor and ends there without only turning the base."""
        goals = np.array([self.joints.q(self.joints.raw(row)) for row in path])
        lows = lowest(self.chain, goals, FLOOR_JOINTS)
        ends = lowest(self.chain, goals[-1], FLOOR_JOINTS)
        starts = lowest(self.chain, start.q, FLOOR_JOINTS)
        aim = self.joints.raw(target)
        turn = all(aim[m] == start.raw[m] for m in ARM[1:])  # only the base turns: every height stays put
        floors = []
        for what, low, end, begin, floor, need in (
                ("tool point", lows[0], ends[0], starts[0], TOOL_FLOOR, "its {} mm floor"),
                ("elbow or wrist", lows[1], ends[1], starts[1], JOINT_FLOOR, "the {} mm their servos need")):
            need = need.format(round(1000 * floor))
            if begin >= floor:
                if low < floor:
                    raise Refused(f"the move would take the {what} down to {1000 * low:.0f} mm above the table, under "
                                  f"{need}")
            elif low < begin - LEVEL:
                raise Refused(f"the move would take the {what} down to {1000 * low:.0f} mm above the table, lower "
                              f"than the {1000 * begin:.0f} mm it starts at, which is already under {need}")
            elif end < floor and not turn:
                raise Refused(f"the move would leave the {what} {1000 * end:.0f} mm above the table, still under "
                              f"{need}: from under a floor, a move has to climb back over it or only turn the base")
            floors.append(min(floor, begin))
        return floors[0], floors[1]

    def _degrees(self, joint: str, raw: float, ref: float) -> float:
        """How far `raw` is from `ref`, as a change in the joint's URDF angle, in degrees."""
        return math.degrees(self.joints.angle(joint, raw) - self.joints.angle(joint, ref))

    def _watch(self, joint: str, still: dict, shift: dict, samples: list) -> None:
        """Read the arm as fast as the bus allows for WATCH seconds after `joint`'s torque came on. A powered joint more
        than MAX_JUMP from `still` (raw ticks), a cut from elsewhere, or a failed read cuts every motor and raises
        Tripped."""
        t0 = self.clock()
        while self.clock() - t0 < WATCH:
            self._stopping(f"the {NAMES[joint]} was coming on")
            try:
                now = self._read()
            except Exception as e:  # noqa: BLE001
                raise self._trip(f"{failure(e)} as the {NAMES[joint]} came on ({e})") from e
            moved = {m: self._degrees(m, now.raw[m], still[m]) for m in JOINTS if m in self.powered}
            samples.append({"t": round(now.t - t0, 4), "came_on": joint, **{m: round(v, 2) for m, v in moved.items()},
                            **{f"{m}_load": now.load[m] for m in moved}})
            for m, v in moved.items():
                shift[m] = max(shift[m], abs(v))
            worst = max(moved, key=lambda m: abs(moved[m]))
            if abs(moved[worst]) > MAX_JUMP:
                raise self._trip(f"the {NAMES[worst]} moved {moved[worst]:+.1f}° when it should have held still, "
                                 f"{1000 * (now.t - t0):.0f} ms after the {NAMES[joint]}'s torque came on (load "
                                 f"{now.load[worst]:+d})")

    def _run(self, path: np.ndarray, start: Reading, floors: tuple, samples: list | None,
             now: Reading | None = None) -> Reading:
        """Send the path's rows one tick apart, then its last row for SETTLE seconds, checking every joint and the
        floors after each write. `now`, for a re-aim, is where the joints are as its path starts; the gripper is
        checked against where it was at the move's start. The bus lock is held by move_to."""
        bus = self.robot.bus
        rows = list(path) + [path[-1]] * max(1, round(SETTLE * HZ))
        others = [m for m in JOINTS if m in self.powered and m not in ARM]  # the gripper: it should stay put
        present, reading = (now or start).raw, (now or start)
        t0 = self.clock()
        try:
            for k, row in enumerate(rows, 1):
                self._stopping("the arm was moving")
                want = self.joints.raw(row)
                bus.sync_write("Goal_Position", {m: min(present[m] + STEP_TICKS, max(present[m] - STEP_TICKS, want[m]))
                                                 for m in ARM}, normalize=False)
                reading = self._read()
                present = reading.raw
                if samples is not None:
                    samples.append({"t": round(reading.t - t0, 4), "goal": [round(math.degrees(v), 2) for v in row],
                                    "q": [round(math.degrees(v), 2) for v in reading.q], "load": dict(reading.load)})
                for m in ARM:
                    off = self._degrees(m, present[m], want[m])
                    if abs(off) > MAX_STRAY:
                        raise self._trip(f"the {NAMES[m]} was {off:+.1f}° from where it should be, partway through "
                                         "the move")
                for m in others:  # moving toward its own goal is following it: a grip whose object slipped out
                    goal = start.raw[m] if self._jaw_goal is None else self._jaw_goal
                    away = (abs(present[m] - goal) - abs(start.raw[m] - goal)) / TICKS_PER_DEGREE
                    if away > MAX_STRAY:
                        off = self._degrees(m, present[m], start.raw[m])
                        raise self._trip(f"the {NAMES[m]} moved {off:+.1f}° while the arm was moving")
                for what, height, floor in zip(("tool point", "elbow or wrist"),
                                               lowest(self.chain, reading.q, FLOOR_JOINTS), floors):
                    if height < floor - SAG:
                        raise self._trip(f"the {what} went down to {1000 * height:.1f} mm above the table, more than "
                                         f"{1000 * SAG:.0f} mm under the {1000 * floor:.0f} mm the move keeps it "
                                         "over, partway through the move")
                wait = t0 + k / HZ - self.clock()
                if wait > 0:
                    self.sleep(wait)
        except Tripped:
            raise
        except KeyboardInterrupt:
            try:
                self.hold()
            except Exception:  # noqa: BLE001 - it can't hold, so let go instead
                self.cut()
            raise
        except Exception as e:  # noqa: BLE001 - the bus, or a bug: whatever it is, cut the motors
            raise self._trip(f"{failure(e)} while the arm was moving ({e})") from e
        return reading

    def _reaim(self, target: np.ndarray, start: Reading, floors: tuple, reading: Reading,
               samples: list | None) -> Reading:
        """Each joint stops short of its goal by its load over its P gain, and holds there. So, while a joint is more
        than AIM from its target, for up to AIM_ROUNDS rounds: move its goal on by how far it stopped short, at most
        MAX_OFFSET past its target and never past its range; ramp every goal there from the last ones under the speed
        caps; and hold them SETTLE seconds with every check a move has. A round whose goals would go under the move's
        floors isn't run. Returns the last reading. The bus lock is held by move_to."""
        ends = np.array([self.joints.limits(m) for m in ARM])
        reach, near = math.radians(MAX_OFFSET), math.radians(AIM)
        goal = target
        for _ in range(AIM_ROUNDS):
            short = target - reading.q
            if np.all(np.abs(short) <= near):
                break
            aim = np.where(np.abs(short) > near, goal + short, goal)
            aim = np.clip(np.clip(aim, target - reach, target + reach), ends[:, 0], ends[:, 1])
            if np.array_equal(aim, goal):  # every joint still short is at the most a re-aim may ask of it
                break
            seconds = min_seconds(goal, aim, math.radians(MAX_JOINT_SPEED), self.chain, MAX_TIP_SPEED)
            path = joint_path(goal, aim, max(seconds, 1 / HZ), HZ)
            tool, axis = lowest(self.chain, path, FLOOR_JOINTS)
            if tool < floors[0] or axis < floors[1]:
                break
            reading = self._run(path, start, floors, samples, now=reading)
            goal = aim
        return reading

    def _closed_end(self) -> int:
        """The gripper's closed end in raw ticks: on this arm, the low end of its calibrated range."""
        return self.joints.servos["gripper"].range_min

    def _jaw_path(self, now: int, goal: int, speed: float) -> list:
        """The gripper's goals from `now` to `goal`, raw ticks, one a tick along a quintic peaking at `speed` degrees a
        second."""
        seconds = abs(goal - now) / TICKS_PER_DEGREE * PEAK / speed
        return [round(v) for v in joint_path([now], [goal], max(seconds, 1 / HZ), HZ)[:, 0]]

    def _jaw(self, path: list, start: Reading, samples: list | None, closing: bool) -> Reading:
        """Send the gripper's goals one tick apart, then the last for SETTLE seconds, checking after each write that
        every powered arm joint stays within MAX_STRAY of where it was and, with all five powered, over its floors.
        Closing on something, the jaws stop short of their goal, which is the grip, but they may not open MAX_STRAY.
        Otherwise a jaw trailing its goal by GRIP_LAG is stopped where it is, and says so, and that goal is held SETTLE
        seconds with the same checks while the jaw comes to rest: one stopped at 110°/s coasted on 6.6° (Oct 29), and a
        grip started then took that for the jaws opening. One going MAX_STRAY the wrong way trips. The bus lock is
        held."""
        bus = self.robot.bus
        floors = tuple(min(f, h) for f, h in zip((TOOL_FLOOR, JOINT_FLOOR), lowest(self.chain, start.q, FLOOR_JOINTS)))
        arm = [m for m in ARM if m in self.powered]
        was = start.raw["gripper"]
        way = 1 if path[-1] >= was else -1  # raw rises as the jaws open
        settle = max(1, round(SETTLE * HZ))
        rows, stopped = path + [path[-1]] * settle, False
        reading, t0, k = start, self.clock(), 0
        try:
            while k < len(rows):
                goal, k = rows[k], k + 1
                self._stopping("the gripper was moving")
                bus.sync_write("Goal_Position", {"gripper": goal}, normalize=False)
                reading = self._read()
                jaw = reading.raw["gripper"]
                if samples is not None:
                    samples.append({"t": round(reading.t - t0, 4), "goal": goal, "jaw": jaw,
                                    "load": reading.load["gripper"]})
                for m in arm:
                    off = self._degrees(m, reading.raw[m], start.raw[m])
                    if abs(off) > MAX_STRAY:
                        raise self._trip(f"the {NAMES[m]} moved {off:+.1f}° while the gripper was moving")
                wrong = (was - jaw) * way / TICKS_PER_DEGREE
                if wrong > MAX_STRAY:
                    raise self._trip(f"the gripper went {wrong:.1f}° the wrong way while it was "
                                     + ("closing" if way < 0 else "opening"))
                if not closing and not stopped and (goal - jaw) * way / TICKS_PER_DEGREE > GRIP_LAG:
                    bus.sync_write("Goal_Position", {"gripper": jaw}, normalize=False)
                    self._jaw_goal, rows, stopped = jaw, rows[:k] + [jaw] * settle, True
                    self.say(f"The gripper stopped at {self.joints.gripper_percent(jaw):.0f}% open, short of where it "
                             "was going: something's in the way.")
                if len(arm) == len(ARM):
                    for what, height, floor in zip(("tool point", "elbow or wrist"),
                                                   lowest(self.chain, reading.q, FLOOR_JOINTS), floors):
                        if height < floor - SAG:
                            raise self._trip(f"the {what} went down to {1000 * height:.1f} mm above the table, more "
                                             f"than {1000 * SAG:.0f} mm under the {1000 * floor:.0f} mm it was over, "
                                             "while the gripper was moving")
                wait = t0 + k / HZ - self.clock()
                if wait > 0:
                    self.sleep(wait)
        except Tripped:
            raise
        except KeyboardInterrupt:  # the jaws stop where they are, squeezing nothing: there may be a finger in them
            self._gripping = False
            try:
                self.hold()
            except Exception:  # noqa: BLE001 - it can't hold, so let go instead
                self.cut()
            raise
        except Exception as e:  # noqa: BLE001 - the bus, or a bug: whatever it is, cut the motors
            raise self._trip(f"{failure(e)} while the gripper was moving ({e})") from e
        return reading

    def _grip(self, reading: Reading) -> Grip:
        jaw = reading.raw["gripper"]
        gap = jaw - self._closed_end()
        return Grip(round(self.joints.gripper_percent(jaw), 1), gap, reading.load["gripper"],
                    self._gripping and gap >= EMPTY)
