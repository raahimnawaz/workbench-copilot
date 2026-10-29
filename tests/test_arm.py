"""The arm driver, against a fake LeRobot follower in raw ticks."""
import gc
import math
import re
import signal
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from test_joints import CALIBRATION
from wbc.config import ROOT
from wbc.motion import arm as driver
from wbc.motion.arm import Arm, Refused, Tripped
from wbc.motion.joints import ARM, JOINTS, JointMap
from wbc.motion.kinematics import load_chain
from wbc.motion.trajectory import joint_path, lowest, min_seconds

ROBOT_CFG = yaml.safe_load((ROOT / "config" / "robot.yaml").read_text())
CHAIN = load_chain(ROBOT_CFG, ROOT)
MAP = JointMap.from_config(ROBOT_CFG, CALIBRATION, CHAIN)
TPD = 4096 / 360

# Poses in URDF degrees, ARM order, with where forward kinematics puts them
START = [0, -20, 40, 30, 0]          # the tool point 64 mm above the table, every axis above 144 mm
TARGET = [15, -10, 40, 30, 20]       # 28 mm above it, and the path from START stays above 27 mm
UNDER = [0, -40, 78, 40, 0]          # the tool point 16 mm under the table
LOW = [0, -40, 70, 40, 0]            # the tool point 6 mm up, under its 10 mm floor
LOWER = [0, -40, 72, 40, 0]          # the tool point on the table
SIDEWAYS = [10, -40, 70, 40, 0]      # LOW with the base turned: the same height
NEAR = [0, -40, 68, 40, 0]           # the tool point 11 mm up, just over its floor, and the path from START too
WRIST_DOWN = [0, 110, -90, -95, 0]   # from the L pose the tool point stays above 150 mm, but a wrist axis goes to 10 mm


def pose(degrees, gripper=1500):
    """Raw ticks for the arm at URDF angles in degrees, with the gripper at `gripper`."""
    return {**MAP.raw(np.radians(degrees)), "gripper": gripper}


class FakeBus:
    """LeRobot's bus for the fake follower, in raw values only, as the driver uses it. One-servo writes are
    acknowledged and fail with the power off; a broadcast sync write just goes unheard. Like scservo_sdk's port, it
    refuses every request while it's still marked busy by one that was interrupted awaiting its reply, and that reply
    still arrives, 16 ms later: a read before the input is flushed after that takes it for its own."""

    def __init__(self, arm):
        self.arm = arm
        self.port_handler = SimpleNamespace(is_using=False, closePort=lambda: arm.events.append("port closed"),
                                            ser=SimpleNamespace(reset_input_buffer=self.flush))

    def request(self):
        if self.port_handler.is_using:
            raise ConnectionError("[TxRxResult] Port is in use!")

    def flush(self):
        arm = self.arm
        if arm.stale is not None and (arm.clock is None or arm.clock.t >= arm.stale_lands):
            arm.stale = None  # it had come in, and goes

    @property
    def is_calibrated(self):
        return self.arm.calibrated

    def connect(self, handshake=True):
        self.arm.events.append("bus connect")

    def disconnect(self, disable_torque=True):
        self.arm.events.append(f"disconnect, torque off: {disable_torque}")

    def sync_read(self, name, motors=None, *, normalize=True, num_retry=0):
        assert not normalize
        self.request()
        arm = self.arm
        arm.reads += 1
        if arm.stale is not None:  # the interrupted request's reply, arriving first
            values, arm.stale = arm.stale, None
            return values
        if not arm.power or (arm.fail_after is not None and arm.reads > arm.fail_after):
            raise ConnectionError("there is no status packet")
        values = self.values(name)
        if arm.reads in (arm.interrupt_at, arm.wedge_at):  # while the reply is on its way: the port stays marked busy
            self.port_handler.is_using = True
            arm.stale, arm.stale_lands = values, (arm.clock.t if arm.clock else 0.0) + 0.016
            raise (KeyboardInterrupt if arm.reads == arm.interrupt_at else
                   OSError("device reports readiness to read but returned no data"))
        return values

    def values(self, name):
        arm = self.arm
        if name == "Present_Position":
            arm.creep()
            raw = {m: round(arm.pos[m]) for m in JOINTS}
            if arm.coast:
                raw["gripper"] += arm.coast.pop(0)
            return raw
        if name == "Present_Temperature":
            return arm.temps.pop(0) if arm.temps else dict(arm.temperature)
        if name == "Torque_Enable":
            return {m: int(on) for m, on in arm.torque.items()}
        return dict({"Present_Load": arm.load, "Goal_Position": arm.goal, "Torque_Limit": arm.limit}[name])

    def sync_write(self, name, values, *, normalize=True, num_retry=0):
        assert not normalize
        self.request()
        arm = self.arm
        if name == "Torque_Enable":
            assert not any(values.values())
            if arm.lost_cuts:
                arm.lost_cuts -= 1
                arm.events.append("torque-off lost")
                return
            arm.events.append("all torque off")
            if arm.power:
                arm.torque.update({m: False for m in values})
            return
        assert name == "Goal_Position"
        arm.sent.append(({m: round(arm.pos[m]) for m in values}, dict(values)))  # where each joint was, and its goal
        if arm.power:
            arm.goal.update(values)
            arm.follow()

    def write(self, name, motor, value, *, normalize=True, num_retry=0):
        assert not normalize
        self.request()
        arm = self.arm
        if not arm.power:
            raise ConnectionError("there is no status packet")
        # one goal at a time goes in only to bring its joint on, or again after a flagged reply
        assert name != "Goal_Position" or not arm.torque[motor] or value == arm.goal[motor]
        arm.events.append(f"{name} {motor} {value}")
        if (name, motor) not in arm.deaf:
            registers = {"Goal_Position": arm.goal, "P_Coefficient": arm.gain, "Torque_Limit": arm.limit,
                         "Lock": arm.lock}
            registers[name][motor] = value
            if name == "Goal_Position":  # the servo turns its torque on for it (Oct 29)
                arm.come_on(motor)
        arm.reply(name, motor, value)

    def read(self, name, motor, *, normalize=True, num_retry=0):
        assert name == "P_Coefficient" and not normalize
        self.request()
        return self.arm.gain[motor]

    def enable_torque(self, motors=None, num_retry=0):  # LeRobot's writes Torque_Enable 1, then Lock 1
        self.request()
        arm = self.arm
        for m in [motors] if isinstance(motors, str) else list(motors or JOINTS):
            arm.lock[m] = 1
            arm.come_on(m)
            arm.reply("Torque_Enable", m, 1)

    def disable_torque(self, motors=None, num_retry=0):  # LeRobot's writes Torque_Enable 0, then Lock 0
        self.arm.torque = {m: False for m in JOINTS}
        self.arm.lock = {m: 0 for m in JOINTS}


class FakeFollower:
    """Stands in for LeRobot's SO101Follower, in raw ticks with the arm's real calibration. A joint with its torque on
    goes to its goal at each goal write, a sagging one to a set number of ticks off it, a stuck one not at all, and a
    runaway one keeps moving every time it's read. Something in a joint's way (`between`, its lowest and highest raw
    positions) stops it there, as an object between the jaws stops the gripper. Every goal register starts at 0, as
    after a power-up, and writing one goal with the torque off turns that joint's torque on, as the real servos did on
    Oct 29. A reply in `flagged` takes the write and then flags the input voltage, as the gripper's did at 15:32. With
    `jaw` set, the gripper follows its goal as the capped one did at 15:57, a share of the way each write up to a top
    speed, and `coast` offsets its next readings, as a jaw stopped at speed carried on and came back."""

    def __init__(self, start=None, calibration=True, calibrated=True, stuck=(), deaf=(), jolt=None, runaway=None,
                 sag=None):
        self.pos = {m: float(v) for m, v in (start or pose(START)).items()}
        self.goal = {m: 0 for m in JOINTS}
        self.torque = {m: False for m in JOINTS}
        self.limit = {m: 500 if m == "gripper" else 1000 for m in JOINTS}
        self.gain = {m: 32 if m == "elbow_flex" else 16 for m in JOINTS}  # in EEPROM: the elbow kept its 32
        self.lock = {m: 1 for m in JOINTS}
        self.load = {m: 0 for m in JOINTS}
        self.temperature, self.temps = {m: 27 for m in JOINTS}, []
        self.calibration = {m: SimpleNamespace(**c) for m, c in CALIBRATION.items()} if calibration else {}
        self.calibration_fpath = Path("/calibration/robots/so_follower/wbc_follower.json")
        self.calibrated, self.power = calibrated, True
        self.stuck, self.deaf, self.jolt, self.runaway = set(stuck), set(deaf), jolt or {}, runaway or {}
        self.sag, self.between = sag or {}, {}
        self.events, self.sent = [], []
        self.reads, self.fail_after, self.interrupt_at, self.wedge_at = 0, None, None, None
        self.lost_cuts, self.after_on = 0, None  # torque-off broadcasts to lose; called with each joint as it comes on
        self.flagged = {}  # replies still to flag the input voltage: {(register, joint) or "configure": how many}
        self.jaw, self.coast = None, []  # (share of the way, top ticks) each goal write; ticks the next readings carry
        self.stale, self.stale_lands, self.clock = None, 0.0, None  # an interrupted read's reply, when it lands, and
                                                                    # the driver's clock to tell
        self.bus = FakeBus(self)

    def configure(self):  # like LeRobot's: every motor off and unlocked, P 16 on all six, then torque on
        self.bus.disable_torque()
        if self.flagged.get("configure"):  # as at 15:32 on Oct 29, before anything else was set
            self.flagged["configure"] -= 1
            raise RuntimeError("Failed to write 'Torque_Enable' on id_=6 with '0' after 1 tries. [RxPacketError] Input "
                               "voltage error!")
        self.gain = {m: 16 for m in JOINTS}
        self.events.append("configured")
        self.bus.enable_torque()

    def come_on(self, m):
        """m's torque on, if it's off: it drives to its goal register, plus any jolt, unless it's stuck."""
        if self.torque[m]:
            return
        self.torque[m] = True
        self.events.append(f"torque on {m}")
        if m not in self.stuck:
            self.pos[m] = self.goal[m] + self.jolt.get(m, 0)
        if self.after_on:
            self.after_on(m)

    def reply(self, name, motor, value):
        """The status reply to a write the servo took: LeRobot raises if it flags the input voltage."""
        if self.flagged.get((name, motor)):
            self.flagged[(name, motor)] -= 1
            raise RuntimeError(f"Failed to write '{name}' on id_={JOINTS.index(motor) + 1} with '{value}' after 1 "
                               "tries. [RxPacketError] Input voltage error!")

    def connect(self, calibrate=True):
        raise AssertionError("robot.connect() can recalibrate the arm and turns torque on: the driver never calls it")

    def follow(self):
        for m in JOINTS:
            if self.torque[m] and m not in self.stuck:
                low, high = self.between.get(m, (-math.inf, math.inf))
                aim = self.goal[m] + self.sag.get(m, 0)
                if m == "gripper" and self.jaw:
                    share, top = self.jaw
                    aim = self.pos[m] + max(-top, min(top, share * (aim - self.pos[m])))
                self.pos[m] = min(high, max(low, aim))

    def creep(self):
        for m, rate in self.runaway.items():
            if self.torque[m]:
                self.pos[m] += rate

    def power_cut(self):
        """The strip switched off and on again: the servos come back limp, every goal 0, their caps gone, locked
        (Oct 28)."""
        self.torque = {m: False for m in JOINTS}
        self.goal = {m: 0 for m in JOINTS}
        self.limit = {m: 500 if m == "gripper" else 1000 for m in JOINTS}
        self.lock = {m: 1 for m in JOINTS}


class Clock:
    """Time that passes as the driver sleeps, and a millisecond each time it looks."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.001
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def make(robot=None, **kwargs):
    robot = robot or FakeFollower()
    clock, said = Clock(), []
    robot.clock = clock
    arm = Arm(robot, MAP, CHAIN, gains={"elbow_flex": 32}, say=said.append, clock=clock, sleep=clock.sleep, **kwargs)
    return arm, robot, said


def powered(start=START, **kwargs):
    """An arm powered on at URDF angles `start`, in degrees, with the fake's record of writes cleared."""
    arm, robot, said = make(FakeFollower(start=pose(start), **kwargs))
    arm.power_on()
    robot.events.clear()
    robot.sent.clear()
    return arm, robot, said


def wrote(robot):
    """Everything the driver sent the servos except the bus connect: writes, setup, torque, goals."""
    return [e for e in robot.events if e != "bus connect"] + robot.sent


def tool_height(arm):
    return lowest(CHAIN, arm.read().q, driver.FLOOR_JOINTS)[0]


def at_the_tenth_goal(robot, then):
    """Make the fake call then() right after the move's tenth goal write."""
    write = robot.bus.sync_write

    def sync_write(name, values, **kwargs):
        write(name, values, **kwargs)
        if name == "Goal_Position" and len(robot.sent) == 10:
            then()

    robot.bus.sync_write = sync_write


def start_and_wait_for_the_stop(arm, *threads):
    """Start threads that stop the arm, and wait until the stop is raised, so they're waiting on the bus lock."""
    for t in threads:
        t.start()
    deadline = time.monotonic() + 2
    while not arm._stopped.is_set() and time.monotonic() < deadline:
        time.sleep(0.001)


# ------------------------------------------------------------------ power-on

def test_power_on_sets_gains_and_caps_before_any_torque_then_brings_the_joints_on_one_at_a_time_from_the_base_out():
    arm, robot, said = make()
    shift = arm.power_on()
    events = robot.events
    first_on = events.index("torque on shoulder_pan")
    assert events.index("configured") < events.index("P_Coefficient elbow_flex 32") < first_on
    assert all(any(e.startswith(f"Torque_Limit {m} ") for e in events[:first_on]) for m in JOINTS)
    assert [e for e in events if e.startswith("torque on")] == [f"torque on {m}" for m in JOINTS]
    # one goal each, from a fresh reading as its joint's turn comes, since writing it turns the torque on
    assert [e for e in events if e.startswith("Goal_Position")] == [f"Goal_Position {m} {pose(START)[m]}"
                                                                    for m in JOINTS]
    for m in JOINTS:
        assert events[events.index(f"torque on {m}") - 1] == f"Goal_Position {m} {pose(START)[m]}"
    assert robot.gain == {**{m: 16 for m in JOINTS}, "elbow_flex": 32}  # configure() set 16, then the elbow's 32
    assert robot.limit == {**{m: 500 for m in ARM}, "gripper": 240}
    assert arm.powered == set(JOINTS) and max(shift.values()) < 0.1
    assert robot.pos == pytest.approx(pose(START))


def test_without_the_hold_here_goals_lerobots_setup_would_drive_every_joint_to_0():
    robot = FakeFollower()
    robot.configure()  # LeRobot's own configure(), with its torque-on left in
    assert robot.pos == {m: 0 for m in JOINTS}


def test_power_on_goes_base_out_whatever_order_the_joints_come_in():
    arm, robot, said = make()
    arm.power_on(["wrist_flex", "shoulder_pan", "elbow_flex"])
    assert [e for e in robot.events if e.startswith("torque on")] == [
        "torque on shoulder_pan", "torque on elbow_flex", "torque on wrist_flex"]


def test_powering_the_base_alone_leaves_the_other_five_off():
    arm, robot, said = make()  # on Oct 29 a hold-here goal written to all six first brought all six on
    arm.power_on(["shoulder_pan"])
    assert robot.torque == {m: m == "shoulder_pan" for m in JOINTS} and arm.powered == {"shoulder_pan"}


@pytest.mark.parametrize("register, joint, match", [
    ("P_Coefficient", "elbow_flex", "the elbow's gain didn't take"),
    ("Torque_Limit", "shoulder_lift", "the torque cap didn't take on the shoulder lift"),
])
def test_a_setting_that_doesnt_take_keeps_torque_off(register, joint, match):
    arm, robot, said = make(FakeFollower(deaf={(register, joint)}))
    with pytest.raises(RuntimeError, match=match):
        arm.power_on()
    assert not any(robot.torque.values()) and arm.powered == set()
    assert not any(e.startswith("torque on") for e in robot.events)


def test_a_hold_here_goal_that_doesnt_take_cuts_every_motor_before_its_joint_comes_on():
    arm, robot, said = make(FakeFollower(deaf={("Goal_Position", "wrist_flex")}))
    with pytest.raises(Tripped, match="the wrist flex's hold-here goal didn't take as it came on, so every motor was "
                                      "cut"):
        arm.power_on()
    assert "torque on elbow_flex" in robot.events and "torque on wrist_flex" not in robot.events
    assert robot.events[-1] == "all torque off" and not any(robot.torque.values()) and arm.powered == set()


@pytest.mark.parametrize("flagged", [
    {"configure": 2},  # LeRobot's setup, as at 15:32 on Oct 29
    {("P_Coefficient", "elbow_flex"): 1},
    {("Torque_Limit", "gripper"): 2},
    {("Goal_Position", "gripper"): 1},  # the write that brings it on
    {("Torque_Enable", "gripper"): 2},
], ids=["setup", "gain", "cap", "hold-here goal", "torque-on"])
def test_a_reply_that_flags_the_input_voltage_is_sent_again(flagged):
    robot = FakeFollower()
    robot.flagged = dict(flagged)
    arm, robot, said = make(robot)
    arm.power_on()
    assert arm.powered == set(JOINTS) and all(robot.torque.values()) and not any(robot.flagged.values())
    assert "all torque off" not in robot.events


@pytest.mark.parametrize("flagged, error, match", [
    ("configure", RuntimeError, "Input voltage error"),
    (("Torque_Enable", "gripper"), Tripped,
     r"the gripper flagged an error as its torque came on \(.*Input voltage error!\), so every motor was cut"),
], ids=["setup", "torque-on"])
def test_a_reply_that_flags_the_input_voltage_every_time_still_fails(flagged, error, match):
    robot = FakeFollower()
    robot.flagged = {flagged: driver.VOLTAGE_TRIES}
    arm, robot, said = make(robot)
    with pytest.raises(error, match=match):
        arm.power_on()
    assert not any(robot.torque.values()) and arm.powered == set()


def test_a_joint_that_runs_away_as_its_torque_comes_on_cuts_every_motor():
    arm, robot, said = make(FakeFollower(runaway={"elbow_flex": -8}))
    samples = []
    with pytest.raises(Tripped, match=r"the elbow moved -3\.\d° when it should have held still, \d+ ms after the "
                                      r"elbow's torque came on .*, so every motor was cut"):
        arm.power_on(samples=samples)
    assert [e for e in robot.events if e.startswith("torque on")] == [
        "torque on shoulder_pan", "torque on shoulder_lift", "torque on elbow_flex"]
    assert robot.events[-1] == "all torque off" and not any(robot.torque.values()) and arm.powered == set()
    assert samples[-1]["came_on"] == "elbow_flex" and samples[-1]["elbow_flex"] < -3


def test_a_jolt_as_a_joint_comes_on_stops_before_the_next_joint():
    arm, robot, said = make(FakeFollower(jolt={"wrist_flex": 68}))
    with pytest.raises(Tripped, match=r"the wrist flex moved \+6\.0° when it should have held still"):
        arm.power_on()
    assert "torque on wrist_roll" not in robot.events and robot.events[-1] == "all torque off"


def test_a_powered_joint_that_shifts_as_a_later_one_comes_on_cuts_every_motor():
    arm, robot, said = make()

    def sag(joint):
        if joint == "elbow_flex":
            robot.pos["shoulder_lift"] -= 50  # the lift gives 4.4° as the elbow takes the forearm

    robot.after_on = sag
    with pytest.raises(Tripped, match=r"the shoulder lift moved -4\.4° when it should have held still, \d+ ms after "
                                      "the elbow's torque came on"):
        arm.power_on()
    assert robot.events[-1] == "all torque off" and not any(robot.torque.values())


def test_a_joint_moved_while_the_ones_before_it_came_on_comes_on_where_it_is_now():
    arm, robot, said = make()
    moved = pose(START)["wrist_flex"] + 50  # 4.4°, by a hand steadying the arm while the shoulder lift came on

    def nudge(joint):
        if joint == "shoulder_lift":
            robot.pos["wrist_flex"] = moved

    robot.after_on = nudge
    arm.power_on()
    assert arm.powered == set(JOINTS) and robot.pos["wrist_flex"] == moved == robot.goal["wrist_flex"]  # no snap back


def test_a_torque_on_that_a_servo_answers_with_an_error_cuts_every_motor():
    arm, robot, said = make()
    enable, flags = robot.bus.enable_torque, ["overload"]  # once: a flag other than the voltage isn't sent again

    def enable_or_flag(motors=None, num_retry=0):
        enable(motors, num_retry=num_retry)
        if motors == "elbow_flex" and flags:
            raise RuntimeError(f"the elbow flags {flags.pop()}")  # LeRobot raises on a status bit after the servo took it

    robot.bus.enable_torque = enable_or_flag
    with pytest.raises(Tripped, match=r"the elbow flagged an error as its torque came on \(the elbow flags overload\), "
                                      "so every motor was cut"):
        arm.power_on()
    assert robot.events[-1] == "all torque off" and not any(robot.torque.values()) and arm.powered == set()


def test_a_bus_failure_during_the_power_on_watch_cuts_every_motor():
    arm, robot, said = make()

    def fail(joint):
        if joint == "shoulder_lift":
            robot.fail_after = robot.reads  # every read from here on fails

    robot.after_on = fail
    with pytest.raises(Tripped, match="the bus failed as the shoulder lift came on"):
        arm.power_on()
    assert "torque on elbow_flex" not in robot.events and arm.powered == set()
    assert robot.events[-1] == "all torque off" and not any(robot.torque.values())


def test_a_cut_from_another_thread_stops_a_power_on_before_the_next_joint():
    arm, robot, said = make()
    cutter = threading.Thread(target=arm.cut)

    def cut(joint):
        if joint == "shoulder_lift":
            start_and_wait_for_the_stop(arm, cutter)

    robot.after_on = cut
    with pytest.raises(Tripped, match="stopped while the shoulder lift was coming on, so every motor was cut"):
        arm.power_on()
    cutter.join(timeout=2)
    assert "torque on elbow_flex" not in robot.events and not any(robot.torque.values()) and arm.powered == set()


@pytest.mark.parametrize("setup, match", [
    (lambda robot: robot.torque.update(elbow_flex=True), r"torque is already on for the elbow.*cut\(\) the motors"),
    (lambda robot: robot.pos.update(wrist_flex=826),
     r"past the end of its range: the wrist flex by 2\.2°. Move it in a little, toward the middle"),
    (lambda robot: robot.pos.update(wrist_roll=4071),  # where the roll read in the L pose on Oct 28
     r"the wrist roll by 15\.1°.*Turn the gripper clockwise, looking from the wrist toward the jaw tips, unless it got "
     "there by turning more than half a turn clockwise"),
    (lambda robot: robot.pos.update(wrist_roll=150), r"the wrist roll by 9\.5°.*Turn the gripper counterclockwise"),
    (lambda robot: robot.temperature.update(shoulder_lift=66), "too hot to power on: the shoulder lift at 66 °C"),
    (lambda robot: robot.temperature.update(wrist_roll=65), "too hot to power on: the wrist roll at 65 °C"),
    (lambda robot: setattr(robot, "power", False), r"the servos didn't answer \(.*\)\. Is their power on\?"),
], ids=["torque already on", "past its range", "roll past its upper limit", "roll past its lower limit", "hot",
        "at the stop", "no power"])
def test_power_on_refuses_and_writes_nothing(setup, match):
    arm, robot, said = make()
    setup(robot)
    with pytest.raises(Refused, match=match):
        arm.power_on()
    assert wrote(robot) == [] and arm.powered == set()


@pytest.mark.parametrize("readings, said_then", [
    ((57, 57, 57, 57, 70), ["Warm: the elbow at 57 °C. Power-on stops at 65 °C."]),  # one glitched reading, Oct 28
    ((70, 27, 27, 27, 27), []),  # a glitch in the first reading
], ids=["warm, one glitch", "a glitch first"])
def test_power_on_goes_by_the_median_temperature_and_mentions_a_warm_servo(readings, said_then):
    robot = FakeFollower()
    robot.temps = [{**robot.temperature, "elbow_flex": c} for c in readings]
    arm, robot, said = make(robot)
    arm.power_on()
    assert arm.powered == set(JOINTS) and said == said_then


def test_a_servo_whose_median_temperature_is_hot_is_refused_despite_low_glitches():
    robot = FakeFollower()
    robot.temps = [{**robot.temperature, "elbow_flex": c} for c in (66, 27, 66, 27, 66)]
    arm, robot, said = make(robot)
    with pytest.raises(Refused, match="too hot to power on: the elbow at 66 °C"):
        arm.power_on()


# ------------------------------------------------------------------ moves

def test_a_move_reaches_its_target_with_every_goal_near_its_joint_and_no_faster_than_the_caps():
    arm, robot, said = powered()
    samples = []
    reading = arm.move_to(np.radians(TARGET), samples=samples)
    assert np.degrees(reading.q) == pytest.approx(TARGET, abs=0.1)
    assert all(abs(goal[m] - where[m]) <= driver.STEP_TICKS for where, goal in robot.sent for m in ARM)
    q0 = MAP.q(pose(START))
    fastest = min_seconds(q0, np.radians(TARGET), math.radians(driver.MAX_JOINT_SPEED), CHAIN, driver.MAX_TIP_SPEED)
    settle = round(driver.SETTLE * driver.HZ)
    assert len(robot.sent) == len(joint_path(q0, np.radians(TARGET), fastest, driver.HZ)) + settle
    assert samples[-1]["t"] >= fastest + driver.SETTLE - 1 / driver.HZ
    goals = np.radians([s["goal"] for s in samples])  # the path itself, tick by tick
    joint_speed = np.degrees(np.abs(np.diff(goals, axis=0))).max() * driver.HZ
    tips = np.array([CHAIN.fk(q)[:3, 3] for q in goals])
    tip_speed = np.linalg.norm(np.diff(tips, axis=0), axis=1).max() * driver.HZ
    assert joint_speed <= driver.MAX_JOINT_SPEED and tip_speed <= 1.01 * driver.MAX_TIP_SPEED
    assert len(samples) == len(robot.sent) and set(samples[0]) == {"t", "goal", "q", "load"}
    assert samples[-1]["q"] == pytest.approx(TARGET, abs=0.1) and set(samples[-1]["load"]) == set(JOINTS)


def test_a_move_takes_at_least_the_seconds_asked():
    arm, robot, said = powered()
    arm.move_to(np.radians(TARGET), seconds=4)
    assert len(robot.sent) == 4 * driver.HZ + round(driver.SETTLE * driver.HZ)


def test_no_goal_runs_more_than_max_step_ahead_of_a_joint_that_lags(monkeypatch):
    monkeypatch.setattr(driver, "MAX_STRAY", 90.0)  # let the lag grow past the stray cut, to see the clamp alone
    arm, robot, said = powered()
    robot.stuck.add("shoulder_pan")
    arm.move_to(np.radians(TARGET))
    assert max(abs(goal[m] - where[m]) for where, goal in robot.sent for m in ARM) == driver.STEP_TICKS


@pytest.mark.parametrize("start, target, match", [
    (START, [0, -20, 96, 30, 0], r"the elbow's target, \+96\.0°, is outside where moves may take it, -99\.8° to "
                                 r"\+90\.3°"),
    (START, [0, -20, 91.3, 30, 0], r"the elbow's target, \+91\.3°, is outside"),  # its end is +92.3°
    (START, [0, -20, 40, 30, 170], r"the wrist roll's target, \+170\.0°, is outside"),
    (START, UNDER, r"the tool point down to -1\d mm above the table, under its 10 mm floor"),
    ([0, 40, 85, 55, -55], [15, -25, 30, 45, -55], r"the tool point down to -3\d mm above the table"),  # both ends up
    ([0, 0, 0, 0, 0], WRIST_DOWN, r"the elbow or wrist down to 1\d mm above the table, under the 30 mm their servos"),
    ([-20, 75, 70, 45, 0], [0, 75, 70, 25, 0], r"the elbow or wrist down to 26 mm"),  # the wrist roll's axis alone
    (LOW, LOWER, r"the tool point down to -?0 mm above the table, lower than the 6 mm it starts at"),
    (LOW, [0, -40, 69, 40, 0], r"leave the tool point \d mm above the table, still under its 10 mm floor: from under "
                               "a floor, a move has to climb back over it or only turn the base"),
], ids=["past the elbow's range", "inside the margin", "past the roll's cable", "under the table", "dips between ends",
        "wrist under its floor", "roll axis under its floor", "lower than a low start", "staying low"])
def test_a_move_that_would_leave_the_envelope_is_refused_with_nothing_written(start, target, match):
    arm, robot, said = powered(start)
    with pytest.raises(Refused, match=match):
        arm.move_to(np.radians(target))
    assert wrote(robot) == [] and arm.powered == set(JOINTS)


def test_a_move_from_under_the_floor_may_turn_the_base_or_climb_back_over_it():
    arm, robot, said = powered(LOW)
    assert np.degrees(arm.move_to(np.radians(SIDEWAYS)).q) == pytest.approx(SIDEWAYS, abs=0.1)
    assert np.degrees(arm.move_to(np.radians(START)).q) == pytest.approx(START, abs=0.1)


@pytest.mark.parametrize("start, toward, floor", [
    (START, UNDER, driver.TOOL_FLOOR),
    ([0, 0, 0, 0, 0], WRIST_DOWN, driver.JOINT_FLOOR),
], ids=["tool point", "wrist axes"])
def test_small_moves_cant_walk_the_arm_under_a_floor(monkeypatch, start, toward, floor):
    """Each move used to be allowed a millimetre under its own start once near a floor, so a chain of small ones
    ratcheted the tool point from 64 mm to 15 mm under the table (review, Oct 29)."""
    monkeypatch.setattr(driver, "SETTLE", 0.02)  # fewer ticks per move, to run quickly
    arm, robot, said = powered(start)
    a, b = np.radians(start), np.radians(toward)
    reached = []
    with pytest.raises(Refused, match="under"):
        for k in range(1, 201):
            arm.move_to(a + (b - a) * k / 200)
            reached.append(lowest(CHAIN, arm.read().q, driver.FLOOR_JOINTS)[0 if floor == driver.TOOL_FLOOR else 1])
    assert floor <= min(reached) < floor + 0.003


def test_from_under_the_floor_a_little_lower_is_refused_too():
    arm, robot, said = powered(LOW)
    for _ in range(3):  # each one a quarter degree more fold: 0.8 mm lower, which used to pass
        with pytest.raises(Refused, match="lower than the 6 mm it starts at"):
            arm.move_to(np.radians([0, -40, 70.25, 40, 0]))
    assert tool_height(arm) == pytest.approx(lowest(CHAIN, np.radians(LOW), driver.FLOOR_JOINTS)[0], abs=0.0005)


def test_the_arm_going_under_a_floor_partway_through_a_move_cuts_every_motor():
    arm, robot, said = powered(sag={"elbow_flex": 45})  # the elbow settles 4° past its goals: inside the 5° stray
    with pytest.raises(Tripped, match=r"the tool point went down to \d\.\d mm above the table, more than 5 mm under the "
                                      "10 mm the move keeps it over, partway through the move, so every motor was cut"):
        arm.move_to(np.radians(NEAR))
    assert robot.events[-1] == "all torque off" and not any(robot.torque.values())


@pytest.mark.parametrize("q, seconds, match", [
    (np.zeros(6), None, "an angle for each of the 5 arm joints"),
    ([0, 0, float("nan"), 0, 0], None, "an angle for each of the 5 arm joints"),
    (np.radians(TARGET), -1, "a move takes from 0 to 60 seconds"),
    (np.radians(TARGET), 600, "a move takes from 0 to 60 seconds"),
])
def test_a_move_needs_five_angles_and_a_sensible_time(q, seconds, match):
    arm, robot, said = powered()
    with pytest.raises(Refused, match=match):
        arm.move_to(q, seconds=seconds)
    assert wrote(robot) == []


def test_a_move_is_refused_until_the_arm_is_powered():
    arm, robot, said = make()
    with pytest.raises(Refused, match="the base, shoulder lift, elbow, wrist flex, wrist roll isn't powered"):
        arm.move_to(np.radians(TARGET))
    assert wrote(robot) == []


def test_a_move_wont_start_from_a_wrist_roll_past_its_limits():
    arm, robot, said = powered()
    robot.pos["wrist_roll"] = 4071  # +177.9°, turned past its limit while the arm held
    with pytest.raises(Refused, match=r"the arm starts past the end of its range: the wrist roll by 15\.1°.*Turn the "
                                      "gripper clockwise"):
        arm.move_to(np.radians([*START[:4], -150]))  # it would have turned the roll -327.9°, winding its cable
    assert wrote(robot) == []


def test_after_a_power_cut_a_move_is_refused_until_the_arm_is_powered_on_again():
    arm, robot, said = powered()
    robot.power_cut()
    with pytest.raises(Refused, match=r"the arm lost power or was reset \(the base's torque is off; .*\): power it on "
                                      "again"):
        arm.move_to(np.radians(TARGET))
    assert robot.sent == [] and arm.powered == set()
    arm.power_on()  # the caps and the hold-here goals go back on before any torque
    assert np.degrees(arm.move_to(np.radians(TARGET)).q) == pytest.approx(TARGET, abs=0.1)


def test_a_servo_that_resets_on_its_own_leaves_the_others_held():
    arm, robot, said = powered()
    robot.torque["wrist_flex"], robot.goal["wrist_flex"], robot.limit["wrist_flex"] = False, 0, 1000  # a brown-out
    with pytest.raises(Refused, match=r"the arm was partly reset \(the wrist flex's torque is off\)\. Hold it, cut\(\) "
                                      "the motors"):
        arm.move_to(np.radians(TARGET))
    assert robot.sent == [] and arm.powered == set(JOINTS) - {"wrist_flex"}
    arm.hold()  # the five still on keep being held
    assert set(robot.sent[-1][1]) == set(JOINTS) - {"wrist_flex"}
    with pytest.raises(Refused, match="torque is already on"):
        arm.power_on()


def test_a_cap_that_was_reset_refuses_the_move():
    arm, robot, said = powered()
    robot.limit["elbow_flex"] = 1000
    with pytest.raises(Refused, match=r"the elbow's cap reads 100%, not 50%"):
        arm.move_to(np.radians(TARGET))
    assert robot.sent == [] and arm.powered == set(JOINTS)


def test_a_stuck_joint_trips_the_move_and_cuts_every_motor():
    arm, robot, said = powered()
    robot.stuck.add("shoulder_pan")
    with pytest.raises(Tripped, match=r"the base was -5\.\d° from where it should be, partway through the move, so "
                                      "every motor was cut"):
        arm.move_to(np.radians(TARGET))
    assert robot.events[-1] == "all torque off" and not any(robot.torque.values()) and arm.powered == set()


@pytest.mark.parametrize("when", ["before", "while"])
def test_a_bus_error_cuts_every_motor(when):
    arm, robot, said = powered()
    robot.fail_after = robot.reads + (0 if when == "before" else 20)
    with pytest.raises(Tripped, match=rf"the bus failed {when} .*\(there is no status packet\)"):
        arm.move_to(np.radians(TARGET))
    assert robot.events[-1] == "all torque off" and not any(robot.torque.values()) and arm.powered == set()


def test_a_serial_error_that_leaves_the_port_busy_still_cuts_every_motor():
    arm, robot, said = powered()
    robot.wedge_at = robot.reads + 20
    with pytest.raises(Tripped, match="the bus failed while the arm was moving .*, so every motor was cut"):
        arm.move_to(np.radians(TARGET))
    assert robot.events[-1] == "all torque off" and not any(robot.torque.values())


def test_anything_else_going_wrong_mid_move_is_named_and_cuts_every_motor():
    arm, robot, said = powered()
    at_the_tenth_goal(robot, lambda: robot.pos.update(elbow_flex=None))  # a reading nothing can use
    with pytest.raises(Tripped, match=r"the driver hit a TypeError while the arm was moving \(.*\), so every motor was "
                                      "cut"):
        arm.move_to(np.radians(TARGET))
    assert not any(robot.torque.values())


def test_a_gripper_drifting_during_a_move_cuts_every_motor():
    arm, robot, said = powered()
    robot.stuck.add("gripper")  # it can't hold its goal, and creeps open
    robot.runaway = {"gripper": 4}
    with pytest.raises(Tripped, match=r"the gripper moved \+5\.\d° while the arm was moving"):
        arm.move_to(np.radians(TARGET))
    assert not any(robot.torque.values())


def test_ctrl_c_mid_move_holds_the_arm_where_it_is():
    arm, robot, said = powered()
    robot.interrupt_at = robot.reads + 31  # partway through the move, while a reply is awaited
    with pytest.raises(KeyboardInterrupt):
        arm.move_to(np.radians(TARGET))
    where, goal = robot.sent[-1]
    assert goal == where and set(goal) == set(JOINTS)  # every powered joint told to stay where it is
    assert 0 < robot.pos["shoulder_pan"] < pose(TARGET)["shoulder_pan"]
    assert all(robot.torque.values()) and arm.powered == set(JOINTS)


def test_after_ctrl_c_a_late_reply_isnt_taken_for_where_the_joints_are():
    """The reply to the read that Ctrl-C interrupted can still be on its way. Taken for hold()'s reading of where the
    joints are, a Present_Load reply would send every joint to its load in ticks."""
    arm, robot, said = powered()
    robot.interrupt_at = robot.reads + 32  # a Present_Load read, partway through the move
    with pytest.raises(KeyboardInterrupt):
        arm.move_to(np.radians(TARGET))
    where, goal = robot.sent[-1]
    assert goal == where and set(goal) == set(JOINTS) and all(robot.torque.values())


def test_hold_sets_every_powered_joint_to_stay_where_it_is():
    arm, robot, said = powered()
    robot.pos["elbow_flex"] += 10  # pushed a little by hand
    arm.hold()
    where, goal = robot.sent[-1]
    assert goal == where and goal["elbow_flex"] == pose(START)["elbow_flex"] + 10


# ------------------------------------------------------------------ re-aim

def test_joints_that_stop_short_are_re_aimed_onto_their_target():
    """On Oct 29 every joint held 1-4° short of its goal, by its load over its P gain, and stayed there."""
    arm, robot, said = powered(sag={"shoulder_lift": -20, "elbow_flex": 25})  # 1.8° one way, 2.2° the other
    reading = arm.move_to(np.radians(TARGET))
    assert np.degrees(reading.q) == pytest.approx(TARGET, abs=driver.AIM)
    last = robot.sent[-1][1]  # each goal now past its target by how far its joint stops short
    assert abs(last["shoulder_lift"] - (pose(TARGET)["shoulder_lift"] + 20)) <= 1
    assert abs(last["elbow_flex"] - (pose(TARGET)["elbow_flex"] - 25)) <= 1


def test_without_aim_a_move_leaves_the_joints_where_they_stopped():
    arm, robot, said = powered(sag={"shoulder_lift": -20})
    reading = arm.move_to(np.radians(TARGET), aim=False)
    assert math.degrees(reading.q[1]) == pytest.approx(TARGET[1] - 20 / TPD, abs=0.1)
    assert robot.sent[-1][1]["shoulder_lift"] == pose(TARGET)["shoulder_lift"]


def test_a_re_aim_sets_no_goal_more_than_max_offset_past_its_target():
    arm, robot, said = powered(sag={"wrist_flex": -50})  # 4.4° short: inside the 5° stray, past the 4° a re-aim may add
    reading = arm.move_to(np.radians(TARGET))
    assert max(goal["wrist_flex"] for _, goal in robot.sent) - pose(TARGET)["wrist_flex"] <= driver.MAX_OFFSET * TPD + 1
    assert math.degrees(reading.q[3]) == pytest.approx(TARGET[3] - (50 / TPD - driver.MAX_OFFSET), abs=0.1)


def test_a_re_aim_never_sets_a_goal_past_its_joints_range():
    arm, robot, said = powered(sag={"wrist_roll": -35})  # 3.1° short of +160°, which is 2.8° inside the cable's limit
    reading = arm.move_to(np.radians([0, -20, 40, 30, 160]))
    top = math.degrees(MAP.limits("wrist_roll")[1])
    assert max(goal["wrist_roll"] for _, goal in robot.sent) <= MAP.raw(np.radians([0, 0, 0, 0, top]))["wrist_roll"]
    assert math.degrees(reading.q[4]) == pytest.approx(top - 35 / TPD, abs=0.1)


def test_a_re_aim_keeps_to_the_speed_caps():
    arm, robot, said = powered(sag={"shoulder_lift": -40, "elbow_flex": 40})  # 3.5° short each
    samples = []
    arm.move_to(np.radians(TARGET), samples=samples)
    q0 = MAP.q(pose(START))
    fastest = min_seconds(q0, np.radians(TARGET), math.radians(driver.MAX_JOINT_SPEED), CHAIN, driver.MAX_TIP_SPEED)
    assert len(samples) > len(joint_path(q0, np.radians(TARGET), fastest, driver.HZ)) + driver.SETTLE * driver.HZ
    goals = np.radians([s["goal"] for s in samples])  # the move, then the re-aim, tick by tick
    joint_speed = np.degrees(np.abs(np.diff(goals, axis=0))).max() * driver.HZ
    tips = np.array([CHAIN.fk(q)[:3, 3] for q in goals])
    tip_speed = np.linalg.norm(np.diff(tips, axis=0), axis=1).max() * driver.HZ
    assert joint_speed <= driver.MAX_JOINT_SPEED and tip_speed <= 1.01 * driver.MAX_TIP_SPEED


def test_a_re_aim_that_would_take_the_arm_under_a_floor_isnt_run():
    arm, robot, said = powered(sag={"elbow_flex": -23})  # held 2° open, so the tool point is higher than planned
    reading = arm.move_to(np.radians(NEAR))  # 11 mm up: folding the elbow 2° more aims the tool point under 10 mm
    assert max(goal["elbow_flex"] for _, goal in robot.sent) == pose(NEAR)["elbow_flex"]
    assert math.degrees(reading.q[2]) == pytest.approx(NEAR[2] - 23 / TPD, abs=0.1)


def test_a_joint_running_away_during_a_re_aim_cuts_every_motor():
    arm, robot, said = powered(sag={"shoulder_lift": -20})
    move = len(joint_path(MAP.q(pose(START)), np.radians(TARGET),
                          min_seconds(MAP.q(pose(START)), np.radians(TARGET), math.radians(driver.MAX_JOINT_SPEED),
                                      CHAIN, driver.MAX_TIP_SPEED), driver.HZ)) + round(driver.SETTLE * driver.HZ)
    write = robot.bus.sync_write

    def sync_write(name, values, **kwargs):
        write(name, values, **kwargs)
        if name == "Goal_Position" and len(robot.sent) == move + 1:  # the re-aim's first goal
            robot.stuck.add("elbow_flex")
            robot.runaway = {"elbow_flex": 8}
    robot.bus.sync_write = sync_write
    with pytest.raises(Tripped, match=r"the elbow was [+-]\d\.\d° from where it should be, partway through the move"):
        arm.move_to(np.radians(TARGET))
    assert not any(robot.torque.values())


# ------------------------------------------------------------------ gripper

CLOSED, OPEN = CALIBRATION["gripper"]["range_min"], CALIBRATION["gripper"]["range_max"]


def test_grip_closes_on_what_is_between_the_jaws_and_keeps_squeezing():
    arm, robot, said = powered()
    robot.between["gripper"] = (1400, math.inf)  # something about 10 mm across
    grip = arm.grip()
    assert grip.holding and grip.gap == 1400 - CLOSED and robot.pos["gripper"] == 1400
    assert robot.goal["gripper"] == CLOSED  # its goal stays at the closed end: it keeps pressing
    assert all(set(goal) == {"gripper"} for _, goal in robot.sent)  # the arm's goals left alone
    assert arm.jaw() == grip


def test_grip_on_nothing_closes_the_jaws_and_says_so():
    arm, robot, said = powered()
    grip = arm.grip()
    assert not grip.holding and grip.gap == 0 and grip.opening == 0 and robot.pos["gripper"] == CLOSED


def test_grip_closes_no_faster_than_its_speed():
    arm, robot, said = powered()
    samples = []
    arm.grip(samples=samples)
    step = max(abs(b["goal"] - a["goal"]) for a, b in zip(samples, samples[1:]))
    assert step <= driver.GRIP_SPEED * TPD / driver.HZ + 1  # goals go out in whole ticks: a step may round up by one
    assert set(samples[0]) == {"t", "goal", "jaw", "load"}


def test_grip_wont_start_with_the_jaws_nearly_shut():
    arm, robot, said = make(FakeFollower(start=pose(START, gripper=CLOSED + 10)))
    arm.power_on()
    robot.sent.clear()
    with pytest.raises(Refused, match="the jaws are 10 ticks from closed, under the 30 a grip starts from"):
        arm.grip()
    assert robot.sent == []


def test_release_opens_to_the_share_asked_and_lets_go():
    arm, robot, said = powered()
    robot.between["gripper"] = (1400, math.inf)
    arm.grip()
    grip = arm.release(60)
    assert robot.goal["gripper"] == round(CLOSED + 0.6 * (OPEN - CLOSED)) and robot.pos["gripper"] == robot.goal["gripper"]
    assert not grip.holding and grip.opening == pytest.approx(60, abs=0.1)
    assert arm.jaw() == grip


def test_release_opens_no_faster_than_its_speed():
    arm, robot, said = powered()
    samples = []
    arm.release(samples=samples)
    step = max(abs(b["goal"] - a["goal"]) for a, b in zip(samples, samples[1:]))
    assert step <= driver.GRIPPER_SPEED * TPD / driver.HZ + 1  # whole ticks: a step may round up by one
    assert robot.pos["gripper"] == OPEN


def test_release_stops_a_jaw_that_meets_something_and_says_so():
    arm, robot, said = powered()
    robot.between["gripper"] = (-math.inf, 1600)  # a hand against the outside of the jaw
    grip = arm.release()
    assert robot.goal["gripper"] == 1600 and grip.opening == pytest.approx(100 * (1600 - CLOSED) / (OPEN - CLOSED), abs=0.1)
    assert said[-1] == "The gripper stopped at 21% open, short of where it was going: something's in the way."
    assert all(robot.torque.values())  # nothing was cut


def test_release_keeps_to_a_speed_the_capped_jaw_can_follow():
    arm, robot, said = make(FakeFollower(start=pose(START, gripper=CLOSED + 12)))
    arm.power_on()
    robot.jaw = (0.15, 25)  # as at 15:57 on Oct 29, opening: about 6° behind at 60°/s, and 110°/s at most
    grip = arm.release(60)
    assert said == [] and grip.opening == pytest.approx(60, abs=0.5)  # at 120°/s it fell 10° behind and stopped


def test_a_stopped_release_holds_the_jaw_while_it_comes_to_rest_so_a_grip_right_after_doesnt_trip():
    arm, robot, said = make(FakeFollower(start=pose(START, gripper=CLOSED + 12)))
    arm.power_on()
    robot.between["gripper"] = (-math.inf, 1700)
    write, sent = robot.bus.sync_write, []

    def sync_write(name, values, **kwargs):
        write(name, values, **kwargs)
        if "gripper" in values:
            if sent and values["gripper"] < sent[-1] and robot.between:  # the stop: as at 15:57, the jaw carries on
                robot.between.clear()                                    # 6.6° and comes back
                robot.coast = [8, 34, 54, 75, 75, 40, 10]
            sent.append(values["gripper"])

    robot.bus.sync_write = sync_write
    grip = arm.release(80)
    assert said == ["The gripper stopped at 27% open, short of where it was going: something's in the way."]
    assert robot.coast == [] and grip.gap == 1700 - CLOSED  # at rest where it stopped before release() returned
    assert not arm.grip().holding  # straight after, as the bench script's step 9 did, without tripping


@pytest.mark.parametrize("opening", [-5, 101, float("nan"), "50"])
def test_release_takes_an_opening_from_0_to_100(opening):
    arm, robot, said = powered()
    with pytest.raises(Refused, match="the gripper opens from 0 to 100 percent"):
        arm.release(opening)
    assert robot.sent == []


def test_grip_and_release_need_the_gripper_powered():
    arm, robot, said = make(FakeFollower(start=pose(START)))
    arm.power_on(ARM)
    for act in (arm.grip, arm.release):
        with pytest.raises(Refused, match="the gripper isn't powered: power the arm on first"):
            act()


def test_grip_and_release_are_refused_after_a_cut():
    arm, robot, said = powered()
    arm.cut()
    robot.sent.clear()
    for act in (arm.grip, arm.release):
        with pytest.raises(Refused, match="the arm is stopped"):
            act()
    assert robot.sent == []


def test_grip_is_refused_when_a_joint_lost_its_torque():
    arm, robot, said = powered()
    robot.torque["wrist_flex"], robot.limit["wrist_flex"] = False, 1000  # a brown-out
    with pytest.raises(Refused, match=r"the arm was partly reset \(the wrist flex's torque is off\)"):
        arm.grip()
    assert robot.sent == []


def test_an_arm_joint_moving_while_the_gripper_closes_cuts_every_motor():
    arm, robot, said = powered()
    robot.stuck.add("elbow_flex")
    robot.runaway = {"elbow_flex": 5}
    with pytest.raises(Tripped, match=r"the elbow moved \+\d\.\d° while the gripper was moving, so every motor was cut"):
        arm.grip()
    assert not any(robot.torque.values())


def test_jaws_forced_open_while_closing_cut_every_motor():
    arm, robot, said = powered()
    robot.stuck.add("gripper")
    robot.runaway = {"gripper": 6}
    with pytest.raises(Tripped, match=r"the gripper went \d\.\d° the wrong way while it was closing"):
        arm.grip()
    assert not any(robot.torque.values())


def test_a_held_object_slipping_out_during_a_move_doesnt_cut_the_arm():
    arm, robot, said = powered()
    robot.between["gripper"] = (1400, math.inf)
    arm.grip()
    robot.sent.clear()
    at_the_tenth_goal(robot, robot.between.clear)  # it slips out, and the jaws close on to their goal
    arm.move_to(np.radians(TARGET))
    assert all(robot.torque.values()) and robot.pos["gripper"] == CLOSED and not arm.jaw().holding


def test_a_grip_pried_open_during_a_move_cuts_every_motor():
    arm, robot, said = powered()
    robot.between["gripper"] = (1400, math.inf)
    arm.grip()
    robot.stuck.add("gripper")
    robot.runaway = {"gripper": 4}  # pulled open, a little more at every reading
    with pytest.raises(Tripped, match=r"the gripper moved \+5\.\d° while the arm was moving"):
        arm.move_to(np.radians(TARGET))
    assert not any(robot.torque.values())


def test_hold_keeps_a_grip_squeezing():
    arm, robot, said = powered()
    robot.between["gripper"] = (1400, math.inf)
    arm.grip()
    arm.hold()
    where, goal = robot.sent[-1]
    assert "gripper" not in goal and set(goal) == set(ARM) and robot.goal["gripper"] == CLOSED and arm.jaw().holding


def test_ctrl_c_during_a_grip_stops_the_jaws_where_they_are_squeezing_nothing():
    arm, robot, said = powered()
    robot.interrupt_at = robot.reads + 20  # partway through closing
    with pytest.raises(KeyboardInterrupt):
        arm.grip()
    where, goal = robot.sent[-1]
    assert goal == where and CLOSED < goal["gripper"] < 1500 and all(robot.torque.values())
    assert not arm.jaw().holding


def test_a_cut_ends_a_grip():
    arm, robot, said = powered()
    robot.between["gripper"] = (1400, math.inf)
    arm.grip()
    arm.cut()
    assert not arm.jaw().holding and not any(robot.torque.values())


# ------------------------------------------------------------------ stopping

def test_a_cut_from_another_thread_stops_a_move_at_its_next_tick():
    arm, robot, said = powered()
    cutter = threading.Thread(target=arm.cut)
    at_the_tenth_goal(robot, lambda: start_and_wait_for_the_stop(arm, cutter))
    with pytest.raises(Tripped, match="stopped while the arm was moving, so every motor was cut"):
        arm.move_to(np.radians(TARGET))
    cutter.join(timeout=2)
    assert not cutter.is_alive() and len(robot.sent) == 10
    assert not any(robot.torque.values()) and arm.powered == set()


def test_a_power_on_queued_behind_a_stop_is_refused():
    """Two parallel tool calls: a power-on waiting on the bus lock behind a move used to re-power the arm right after
    a stop, in 20 of 20 runs (review, Oct 29)."""
    arm, robot, said = powered()
    out = {}

    def power_on():
        try:
            arm.power_on()
            out["power_on"] = "torque on"
        except Refused as e:
            out["power_on"] = str(e)

    queued, stopper = threading.Thread(target=power_on), threading.Thread(target=arm.cut)
    at_the_tenth_goal(robot, lambda: start_and_wait_for_the_stop(arm, queued, stopper))
    with pytest.raises(Tripped, match="stopped while the arm was moving"):
        arm.move_to(np.radians(TARGET))
    for t in (queued, stopper):
        t.join(timeout=2)
    assert out["power_on"].startswith("the arm is stopped") and not any(robot.torque.values())
    assert not any(e.startswith("torque on") for e in robot.events)


def test_a_cut_from_a_signal_handler_stops_the_move_on_its_own_thread():
    """The usual wiring, Ctrl-C calling cut(), runs the handler on the moving thread itself: its stop used to clear
    before the move could see it, and the move went on sending goals (review, Oct 29)."""
    arm, robot, said = powered()
    at_the_tenth_goal(robot, lambda: signal.raise_signal(signal.SIGINT))
    handler = signal.signal(signal.SIGINT, lambda *args: arm.cut())
    try:
        with pytest.raises(Tripped, match="stopped while the arm was moving"):
            arm.move_to(np.radians(TARGET))
    finally:
        signal.signal(signal.SIGINT, handler)
    assert len(robot.sent) == 10 and not any(robot.torque.values())


@pytest.mark.parametrize("how", ["cut", "trip"])
def test_after_a_cut_the_arm_stays_stopped_until_clear_stop(how):
    arm, robot, said = powered()
    if how == "cut":
        assert arm.cut() is True
    else:
        robot.stuck.add("shoulder_pan")
        with pytest.raises(Tripped):
            arm.move_to(np.radians(TARGET))
        robot.stuck.clear()
    robot.events.clear()
    robot.sent.clear()
    for call in (arm.power_on, lambda: arm.move_to(np.radians(TARGET))):
        with pytest.raises(Refused, match=r"the arm is stopped: its motors were cut, and they stay off until "
                                          r"clear_stop\(\)"):
            call()
    assert wrote(robot) == []
    arm.clear_stop()
    arm.power_on()
    assert arm.powered == set(JOINTS)


def test_ctrl_c_in_power_on_leaves_a_busy_port_that_a_cut_still_gets_through():
    """scservo_sdk marks the port busy for each request and clears it only when the reply comes, so Ctrl-C while one
    was awaited used to make every later request fail, and cut() sent nothing (review, Oct 29)."""
    arm, robot, said = make()

    def interrupt(joint):
        if joint == "elbow_flex":
            robot.interrupt_at = robot.reads + 2  # partway through the elbow's watch

    robot.after_on = interrupt
    with pytest.raises(KeyboardInterrupt):
        arm.power_on()
    assert robot.bus.port_handler.is_using and robot.torque["elbow_flex"]
    assert arm.cut() is True and not any(robot.torque.values()) and said == []


def test_a_cut_is_read_back_and_sent_again_until_every_servo_is_off():
    arm, robot, said = powered()
    robot.lost_cuts = 2  # both copies of the first broadcast
    assert arm.cut() is True and not any(robot.torque.values())
    assert robot.events == ["torque-off lost", "torque-off lost", "all torque off", "all torque off"] and said == []


@pytest.mark.parametrize("lost, power, doubt", [
    (2 * driver.CUT_TRIES, True, "the base, shoulder lift, elbow, wrist flex, wrist roll, gripper still reads its "
                                 "torque on"),
    (0, False, r"the servos didn't answer to confirm it \(there is no status packet\)"),
], ids=["every broadcast lost", "no answer"])
def test_a_cut_that_cant_be_confirmed_says_so(lost, power, doubt):
    arm, robot, said = powered()
    robot.lost_cuts, robot.power = lost, power
    assert arm.cut() is False
    assert len(said) == 1 and said[0].startswith("Every motor was told to cut, but ")
    assert re.search(doubt, said[0]) and said[0].endswith("Switch the servo power off at the strip.")


def test_a_trip_whose_cut_isnt_confirmed_says_to_switch_off_the_strip():
    arm, robot, said = powered()
    robot.stuck.add("shoulder_pan")
    robot.lost_cuts = 99
    with pytest.raises(Tripped, match=r"partway through the move\. Every motor was told to cut, but the .* still reads "
                                      "its torque on: switch the servo power off at the strip"):
        arm.move_to(np.radians(TARGET))


# ------------------------------------------------------------------ close

def test_close_cuts_puts_the_caps_back_and_disconnects_without_lerobots_torque_off():
    arm, robot, said = powered()
    with arm:
        pass
    assert robot.events[0] == "all torque off" and robot.events[-1] == "disconnect, torque off: False"
    assert robot.limit == {**{m: 1000 for m in ARM}, "gripper": 500} and not any(robot.torque.values())
    assert arm.powered == set()
    arm.close()  # again: nothing more
    assert robot.events[-1] == "disconnect, torque off: False" and said == []


def test_close_puts_back_the_caps_from_before_the_first_power_on():
    arm, robot, said = powered()
    arm.cut()
    arm.clear_stop()
    arm.power_on()  # the caps it finds now are its own
    arm.close()
    assert robot.limit == {**{m: 1000 for m in ARM}, "gripper": 500}


def test_close_locks_every_servo_after_a_power_on_that_tripped():
    arm, robot, said = make(FakeFollower(jolt={"wrist_flex": 68}))
    with pytest.raises(Tripped):
        arm.power_on()
    assert robot.lock == {**{m: 1 for m in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]},
                          "wrist_roll": 0, "gripper": 0}  # LeRobot's setup unlocked them; only torque-on locks
    arm.close()
    assert robot.lock == {m: 1 for m in JOINTS}


def test_close_sends_a_lock_again_when_its_reply_flags_the_input_voltage():
    arm, robot, said = powered()
    robot.flagged = {("Lock", "gripper"): 1}
    arm.close()
    assert robot.events.count("Lock gripper 1") == 2 and robot.lock == {m: 1 for m in JOINTS}


def test_close_leaves_the_caps_when_the_cut_isnt_confirmed():
    arm, robot, said = powered()
    robot.lost_cuts = 99
    arm.close()
    assert robot.limit == {**{m: 500 for m in ARM}, "gripper": 240} and robot.events[-1].startswith("disconnect")
    assert len(said) == 1 and "so the torque caps stay as they are" in said[0]


def test_close_from_another_thread_stops_a_move_at_its_next_tick():
    """close() used to wait for a running move to finish, sending it 139 more goals, and then drop the arm."""
    arm, robot, said = powered()
    closer = threading.Thread(target=arm.close)
    at_the_tenth_goal(robot, lambda: start_and_wait_for_the_stop(arm, closer))
    with pytest.raises(Tripped, match="stopped while the arm was moving"):
        arm.move_to(np.radians(TARGET))
    closer.join(timeout=2)
    assert len(robot.sent) == 10 and not any(robot.torque.values())
    assert robot.events[-1] == "disconnect, torque off: False"


def test_an_arm_dropped_without_close_cuts_every_motor():
    arm, robot, said = powered()
    del arm
    gc.collect()
    assert robot.events[-1] == "all torque off" and not any(robot.torque.values())


# ------------------------------------------------------------------ open

def test_open_connects_only_the_bus_and_takes_the_map_and_gains_from_the_config():
    made = []

    def make_robot(port, robot_id, calibration_dir):
        made.append((port, robot_id, calibration_dir))
        return FakeFollower()

    arm = Arm.open("/dev/cu.fake", ROBOT_CFG, ROOT, calibration_dir="/elsewhere", make_robot=make_robot)
    assert made == [("/dev/cu.fake", "wbc_follower", "/elsewhere")]
    assert arm.robot.events == ["bus connect"]  # nothing written, and robot.connect() never ran: the fake raises
    assert arm.joints.servos == MAP.servos and arm.gains == {"elbow_flex": 32}


@pytest.mark.parametrize("setup, match", [
    (dict(calibration=False), "there's no calibration file at /calibration/robots/so_follower/wbc_follower.json"),
    (dict(calibrated=False), "the servos' calibration doesn't match the file"),
], ids=["no file", "mismatch"])
def test_open_refuses_a_missing_or_mismatched_calibration_and_closes_the_port(setup, match):
    robot = FakeFollower(**setup)
    with pytest.raises(Refused, match=match):
        Arm.open("/dev/cu.fake", ROBOT_CFG, ROOT, make_robot=lambda *args: robot)
    assert robot.events == ["bus connect", "disconnect, torque off: False"]


def test_a_failed_connect_closes_the_port_it_left_open():
    robot = FakeFollower()

    def connect(handshake=True):
        raise RuntimeError("motor check failed")  # LeRobot leaves the port open after this

    robot.bus.connect = connect
    with pytest.raises(RuntimeError, match="motor check failed"):
        Arm.open("/dev/cu.fake", ROBOT_CFG, ROOT, make_robot=lambda *args: robot)
    assert robot.events == ["port closed"]


def test_the_follower_needs_lerobot(monkeypatch):
    monkeypatch.setitem(sys.modules, "lerobot", None)  # as if it weren't installed
    with pytest.raises(RuntimeError, match=r"LeRobot is missing: uv pip install -e '\.\[lerobot\]'"):
        driver.follower("/dev/cu.fake")


def test_the_follower_leaves_out_lerobots_own_torque_off(monkeypatch):
    """LeRobot's clean-up on disconnect, which also runs when it's garbage collected, would unlock every servo's
    EEPROM, so the Arm cuts the motors itself."""
    made = {}
    so_follower = SimpleNamespace(SO101FollowerConfig=lambda **kwargs: made.update(kwargs) or kwargs,
                                  SO101Follower=lambda config: config)
    for name, module in [("lerobot", SimpleNamespace()), ("lerobot.robots", SimpleNamespace()),
                         ("lerobot.robots.so_follower", so_follower)]:
        monkeypatch.setitem(sys.modules, name, module)
    driver.follower("/dev/cu.fake", calibration_dir="~/elsewhere")
    assert made == {"port": "/dev/cu.fake", "id": "wbc_follower", "calibration_dir": Path("~/elsewhere").expanduser(),
                    "disable_torque_on_disconnect": False}


@pytest.mark.parametrize("kwargs, match", [
    (dict(cap=5), "the torque cap has to be from 10 to 100 percent, not 5"),
    (dict(gains={"elbow": 32}), "a P gain needs a joint from"),
    (dict(gains={"elbow_flex": 100}), "a P gain needs a joint from"),
])
def test_the_arm_takes_only_a_sensible_cap_and_gains(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Arm(FakeFollower(), MAP, CHAIN, **kwargs)
