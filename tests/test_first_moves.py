"""The first powered moves, against a fake LeRobot follower."""
import importlib.util
import math
import re
from types import SimpleNamespace

import pytest

from wbc.config import ROOT

_spec = importlib.util.spec_from_file_location("first_moves", ROOT / "scripts" / "first_moves.py")
first_moves = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(first_moves)

JOINTS = first_moves.JOINTS
TPD = first_moves.TICKS_PER_DEGREE


class FakeBus:
    def __init__(self, arm):
        self.arm = arm

    @property
    def is_calibrated(self):
        return self.arm.calibrated

    def sync_read(self, name, motors=None, *, normalize=True, num_retry=0):
        arm = self.arm
        if name == "Present_Position":
            arm.creep()
            return dict(arm.pos) if normalize else {m: arm.raw(m) for m in JOINTS}
        assert not normalize
        if name == "Present_Load":
            return {m: 0 for m in JOINTS}
        if name == "Goal_Position":
            return dict(arm.goal_raw)
        if name == "Torque_Limit":
            return dict(arm.torque_limit)
        assert name == "Torque_Enable"
        return {m: int(arm.torque[m]) for m in JOINTS}

    def write(self, name, motor, value, *, normalize=True, num_retry=0):
        assert not normalize
        assert name != "Goal_Position" or not self.arm.torque[motor]  # goals are set before torque comes on
        if name == "Goal_Position":
            if motor not in self.arm.deaf:
                self.arm.goal_raw[motor] = value
            self.arm.events.append(f"goal {motor}")
        elif name == "P_Coefficient":
            if motor not in self.arm.deaf:
                self.arm.gain[motor] = value
            self.arm.events.append(f"gain {motor} {value}")
        else:
            assert name == "Torque_Limit"
            self.arm.torque_limit[motor] = value
            if value != 1000:
                self.arm.events.append(f"cap {motor} {value}")

    def read(self, name, motor, *, normalize=True, num_retry=0):
        assert name == "P_Coefficient" and not normalize
        return self.arm.gain[motor]

    def sync_write(self, name, values, *, normalize=True, num_retry=0):
        assert name == "Torque_Enable" and not normalize and not any(values.values())
        for m in values:
            self.arm.torque[m] = False
        self.arm.events.append("all torque off")

    def enable_torque(self, motors=None, num_retry=0):
        for m in [motors] if isinstance(motors, str) else list(motors or JOINTS):
            self.arm.torque[m] = True
            self.arm.events.append(f"torque on {m}")
            if m not in self.arm.stuck:  # a servo drives to its goal register the moment torque comes on
                self.arm.pos[m] = (self.arm.goal_raw[m] - 2047) / TPD + self.arm.jolt.get(m, 0.0)

    def disable_torque(self, motors=None, num_retry=0):
        for m in JOINTS:
            self.arm.torque[m] = False


class FakeArm:
    """Stands in for LeRobot's SO101Follower. Joints are in LeRobot units, with 0 at the calibration pose and
    a calibrated range of ±105°. Each joint follows its goal while its torque is on, unless it's stuck, and a
    runaway joint keeps moving by its rate every time it's read. Every goal register starts stale, as it may
    be on the real arm after calibration and moving it by hand."""

    def __init__(self, start=None, stuck=(), deaf=(), jolt=None, runaway=None, calibrated=True):
        self.pos = {m: 0.0 for m in JOINTS}
        self.pos.update(start or {})
        self.goal_raw = {m: 2047 for m in JOINTS}
        self.torque = {m: False for m in JOINTS}
        self.torque_limit = {m: 1000 for m in JOINTS}
        self.gain = {m: 32 for m in JOINTS}  # the servos' factory P gain
        self.stuck, self.deaf, self.jolt, self.runaway = set(stuck), set(deaf), jolt or {}, runaway or {}
        self.calibrated = calibrated
        self.calibration = {m: SimpleNamespace(range_min=2047 - 1200, range_max=2047 + 1200) for m in JOINTS}
        self.events = []
        self.bus = FakeBus(self)

    def raw(self, m):
        return round(2047 + self.pos[m] * TPD)

    def creep(self):
        for m, rate in self.runaway.items():
            if self.torque[m]:
                self.pos[m] += rate

    def configure(self):  # like LeRobot's: set the motors up with torque off, then turn it on
        self.bus.disable_torque()
        self.gain = {m: 16 for m in JOINTS}
        self.events.append("configured")
        self.bus.enable_torque()

    def send_action(self, action):
        for key, value in action.items():
            m = key.removesuffix(".pos")
            assert self.torque[m]
            if m not in self.stuck:
                self.pos[m] = value
        return action

    def get_observation(self):
        return {f"{m}.pos": v for m, v in self.pos.items()}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.001
        return self.t


def session(arm, replies, **kw):
    """Run the whole session with scripted replies. Each prompt goes into arm.events, so tests can check
    what was asked before what was done. A reply can be a function, which runs as you'd act at that prompt."""
    replies = list(replies)

    def ask(prompt):
        arm.events.append(f"ask: {prompt.strip()}")
        if not replies:
            raise EOFError
        reply = replies.pop(0)
        return reply() if callable(reply) else reply

    said, record = [], {}
    ok = first_moves.run(arm, record, ask=ask, say=said.append, sleep=lambda s: None, clock=Clock(),
                         hz=10, seconds=1, **kw)
    return ok, record, said


def released_last(arm):
    """The session ended with the hold-the-arm prompt, then every motor off."""
    hold = max(i for i, e in enumerate(arm.events) if e.startswith("ask:"))
    return (arm.events[hold].startswith("ask: Hold the arm") and set(arm.events[hold + 1:]) == {"all torque off"}
            and not any(arm.torque.values()))


def cut_without_asking(arm):
    """Every motor went off at once, with no hold-the-arm prompt first."""
    return ("all torque off" in arm.events and not any(arm.torque.values())
            and not any(e.startswith("ask: Hold the arm") for e in arm.events))


def test_ramp_eases_in_and_out_and_ends_on_the_target():
    goals = first_moves.ramp(0.0, 10.0, seconds=2, hz=50)
    steps = [b - a for a, b in zip([0.0] + goals, goals)]
    assert len(goals) == 100 and goals[-1] == pytest.approx(10.0)
    assert all(s > 0 for s in steps) and steps[0] < 0.01 and steps[-1] < 0.01
    assert max(steps) == pytest.approx(10 * math.pi / 200, rel=0.01)


def test_each_joint_steps_toward_the_middle_of_its_range():
    assert first_moves.toward_middle("shoulder_pan", 108.4, 10) == -10
    assert first_moves.toward_middle("elbow_flex", -55.3, 10) == 10
    assert first_moves.toward_middle("gripper", 31.9, 10) == 10
    assert first_moves.toward_middle("gripper", 80.0, 10) == -10


def test_joints_come_on_one_at_a_time_after_every_goal_is_set_and_torque_is_capped():
    arm = FakeArm(start={"shoulder_lift": -99.6, "elbow_flex": -55.3, "wrist_flex": -73.3})
    shift = first_moves.power_on(arm, JOINTS, 50, [], clock=Clock())
    first_on = arm.events.index("torque on shoulder_pan")
    assert arm.events.index("configured") < first_on
    assert all(f"goal {m}" in arm.events[:first_on] and f"cap {m} 500" in arm.events[:first_on] for m in JOINTS)
    assert [e for e in arm.events if e.startswith("torque on")] == [f"torque on {m}" for m in JOINTS]
    assert max(shift.values()) < 0.1

    lazy = FakeArm(start={"elbow_flex": -55.3})  # LeRobot's configure() alone, with the stale goals
    lazy.configure()
    assert lazy.pos["elbow_flex"] == pytest.approx(0.0)


def test_a_gain_is_set_after_lerobots_setup_and_before_any_torque():
    arm = FakeArm()
    first_moves.power_on(arm, JOINTS, 50, [], gains={"elbow_flex": 32}, clock=Clock())
    events = arm.events
    assert events.index("configured") < events.index("gain elbow_flex 32") < events.index("torque on shoulder_pan")
    assert arm.gain == {**{m: 16 for m in JOINTS}, "elbow_flex": 32}


def test_a_gain_that_doesnt_take_keeps_torque_off():
    arm = FakeArm(deaf={"elbow_flex"})
    ok, record, said = session(arm, [""], gains={"elbow_flex": 32})
    assert not ok and not any(e.startswith("torque on") for e in arm.events)
    assert "elbow's gain didn't take" in record["stopped"]


def test_gains_parse_as_joint_equals_value():
    assert first_moves.parse_gains(["elbow_flex=32", "wrist_flex=24"]) == {"elbow_flex": 32, "wrist_flex": 24}
    assert first_moves.parse_gains([]) == {}
    for bad in ["elbow=32", "elbow_flex=200", "elbow_flex=", "elbow_flex", "elbow_flex=-8"]:
        with pytest.raises(ValueError):
            first_moves.parse_gains([bad])


def test_a_servo_that_runs_away_as_its_torque_comes_on_is_cut_before_the_next_joint():
    arm = FakeArm(runaway={"elbow_flex": -2.0})
    ok, record, said = session(arm, [""])
    assert not ok and cut_without_asking(arm)
    assert [e for e in arm.events if e.startswith("torque on")] == [
        "torque on shoulder_pan", "torque on shoulder_lift", "torque on elbow_flex"]
    assert "elbow moved -4.0° on its own" in record["stopped"] and record["results"] == []
    assert record["power_on"][-1]["came_on"] == "elbow_flex"
    assert arm.torque_limit == {m: 1000 for m in JOINTS}  # the cap is lifted again afterwards


def test_a_jump_as_a_joint_comes_on_stops_before_any_move():
    arm = FakeArm(jolt={"wrist_flex": 6.0})
    ok, record, said = session(arm, [""])
    assert not ok and record["samples"] == [] and cut_without_asking(arm)
    assert any("wrist flex moved +6.0° on its own" in s for s in said)


def test_a_goal_that_doesnt_take_keeps_torque_off():
    arm = FakeArm(start={"elbow_flex": -55.3}, deaf={"elbow_flex"})
    ok, record, said = session(arm, [""])
    assert not ok and not any(e.startswith("torque on") for e in arm.events)
    assert "didn't take on elbow_flex" in record["stopped"]


def test_a_full_run_records_which_way_plus_goes_for_every_joint():
    start = {"shoulder_pan": -11.9, "shoulder_lift": -99.6, "elbow_flex": -55.3, "wrist_flex": -73.3,
             "wrist_roll": 2.9, "gripper": 31.9}
    arm = FakeArm(start=start)
    # motors on, then Enter and an answer for each joint, then release
    ok, record, said = session(arm, ["", "", "2", "", "1", "", "2", "", "2", "", "1", "", "1", ""])
    results = record["results"]
    assert ok and released_last(arm)
    assert [r["asked"] for r in results] == [10, 10, 10, 10, -10, 10]
    assert {r["joint"]: r["plus_means"] for r in results} == {
        "shoulder_pan": "counterclockwise", "shoulder_lift": "forward", "elbow_flex": "open",
        "wrist_flex": "out", "wrist_roll": "counterclockwise", "gripper": "open"}
    assert all(r["moved"] == pytest.approx(r["asked"], abs=0.05) and r["back_off"] < 0.05 for r in results)
    assert arm.pos == pytest.approx(start, abs=0.05)
    assert "plus is" in said[-1] and "counterclockwise" in said[-1]
    assert arm.torque_limit == {m: 1000 for m in JOINTS}


def test_show_me_again_repeats_the_move_before_asking_again():
    arm = FakeArm(start={"elbow_flex": -55.3})
    ok, record, said = session(arm, ["", "", "r", "2", ""], joints=["elbow_flex"])
    assert ok and record["results"][0]["plus_means"] == "open"
    target = record["results"][0]["start"] + 10
    at_target = [abs(s["goal"] - target) < 0.01 for s in record["samples"]]
    assert sum(1 for was, now in zip(at_target, at_target[1:]) if now and not was) == 2  # out, back, out


def test_only_the_joints_asked_for_get_torque():
    arm = FakeArm(start={"elbow_flex": -55.3})
    ok, record, said = session(arm, ["", "", "2", ""], joints=["elbow_flex"])
    assert ok and [e for e in arm.events if e.startswith("torque on")] == ["torque on elbow_flex"]
    assert any(e.startswith("ask: Only the elbow will get torque") for e in arm.events)


def test_a_joint_that_doesnt_follow_its_goal_cuts_every_motor_mid_move():
    arm = FakeArm(stuck={"elbow_flex"})
    ok, record, said = session(arm, ["", "", "1", "", "1", ""])
    assert not ok and cut_without_asking(arm)
    assert [r["joint"] for r in record["results"]] == ["shoulder_pan", "shoulder_lift"]
    stray = re.search(r"the elbow was (-\d+\.\d)° from where it should be, partway through its move",
                      record["stopped"])
    assert stray and float(stray.group(1)) < -first_moves.MAX_STRAY


def test_ctrl_c_mid_move_leaves_the_arm_holding_until_its_released():
    arm = FakeArm()
    follow, calls = arm.send_action, []

    def interrupted(action):
        calls.append(action)
        if len(calls) == 5:
            raise KeyboardInterrupt
        return follow(action)

    arm.send_action = interrupted
    ok, record, said = session(arm, ["", "", ""])
    assert not ok and record["results"] == [] and released_last(arm)
    assert any("holding where it is" in s for s in said)


def test_joints_past_their_calibrated_range_are_moved_in_before_torque():
    arm = FakeArm(start={"wrist_flex": -108.0, "gripper": 107.0})

    def move_them_in():
        arm.pos.update(wrist_flex=-100.0, gripper=100.0)
        return ""

    ok, record, said = session(arm, [move_them_in, "q"])
    prompts = [e for e in arm.events if e.startswith("ask:")]
    assert "the wrist flex by 2.5°, the gripper by 1.5°" in prompts[0] and "motors will come on" in prompts[1]
    assert not ok and not any(e.startswith("torque on") for e in arm.events)


def test_it_asks_for_the_base_to_face_forward_before_the_motors_come_on():
    arm = FakeArm(start={"shoulder_pan": 100.0})

    def turn_base():
        arm.pos["shoulder_pan"] = -5.0
        return ""

    ok, record, said = session(arm, [turn_base, "q"])
    prompts = [e for e in arm.events if e.startswith("ask:")]
    assert "turned 100° from forward" in prompts[0] and "motors will come on" in prompts[1]
    assert not ok and not any(e.startswith("torque on") for e in arm.events)


def test_nothing_powers_on_when_the_servos_dont_match_the_calibration():
    arm = FakeArm(calibrated=False)
    ok, record, said = session(arm, [])
    assert not ok and arm.events == []
    assert "lerobot-calibrate" in said[-1]
