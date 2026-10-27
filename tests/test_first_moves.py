"""The first powered moves, against a fake LeRobot follower."""
import importlib.util
import math

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
        if name == "Present_Position":
            return dict(self.arm.pos) if normalize else {m: self.arm.raw(m) for m in JOINTS}
        assert name == "Goal_Position" and not normalize
        return dict(self.arm.goal_raw)

    def write(self, name, motor, value, *, normalize=True, num_retry=0):
        assert name == "Goal_Position" and not normalize
        if motor not in self.arm.deaf:
            self.arm.goal_raw[motor] = value
        self.arm.events.append(f"goal {motor}")

    def enable_torque(self, motors=None, num_retry=0):
        self.arm.torque = True
        self.arm.events.append("torque on")
        for m in JOINTS:  # a servo drives to its goal register the moment torque comes on
            if m not in self.arm.stuck:
                self.arm.pos[m] = (self.arm.goal_raw[m] - 2047) / TPD + self.arm.jolt.get(m, 0.0)

    def disable_torque(self, motors=None, num_retry=0):
        self.arm.torque = False
        self.arm.events.append("torque off")


class FakeArm:
    """Stands in for LeRobot's SO101Follower. Joints are in LeRobot units, with 0 at the calibration pose,
    and each one follows its goal while torque is on unless it's stuck. Every goal register starts stale,
    as it may be on the real arm after calibration and moving it by hand."""

    def __init__(self, start=None, stuck=(), deaf=(), jolt=None, calibrated=True):
        self.pos = {m: 0.0 for m in JOINTS}
        self.pos.update(start or {})
        self.goal_raw = {m: 2047 for m in JOINTS}
        self.stuck, self.deaf, self.jolt = set(stuck), set(deaf), jolt or {}
        self.calibrated = calibrated
        self.calibration = {m: "calibrated" for m in JOINTS}
        self.torque = False
        self.events = []
        self.bus = FakeBus(self)

    def raw(self, m):
        return round(2047 + self.pos[m] * TPD)

    def configure(self):  # like LeRobot's: set the motors up with torque off, then turn it on
        self.bus.disable_torque()
        self.events.append("configured")
        self.bus.enable_torque()

    def send_action(self, action):
        assert self.torque
        for key, value in action.items():
            m = key.removesuffix(".pos")
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

    said, results, log = [], [], []
    ok = first_moves.run(arm, results, log, ask=ask, say=said.append, sleep=lambda s: None, clock=Clock(),
                         hz=10, seconds=1, **kw)
    return ok, results, log, said


def released_last(arm):
    return arm.events[-2].startswith("ask: Hold the arm") and arm.events[-1] == "torque off"


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


def test_torque_comes_on_only_after_every_goal_is_set_to_where_the_arm_is():
    arm = FakeArm(start={"shoulder_lift": -99.6, "elbow_flex": -55.3, "wrist_flex": -73.3})
    shift = first_moves.power_on(arm, sleep=lambda s: None)
    on = arm.events.index("torque on")
    assert arm.events.index("configured") < on
    assert all(f"goal {m}" in arm.events[:on] for m in JOINTS)
    assert max(shift.values()) < 0.1

    lazy = FakeArm(start={"elbow_flex": -55.3})  # LeRobot's configure() alone, with the stale goals
    lazy.configure()
    assert lazy.pos["elbow_flex"] == pytest.approx(0.0)


def test_a_goal_that_doesnt_take_keeps_torque_off():
    arm = FakeArm(start={"elbow_flex": -55.3}, deaf={"elbow_flex"})
    ok, results, log, said = session(arm, ["", ""])
    assert not ok and "torque on" not in arm.events and log == []
    assert any("didn't take on elbow_flex" in s for s in said)


def test_a_full_run_records_which_way_plus_goes_for_every_joint():
    start = {"shoulder_pan": -11.9, "shoulder_lift": -99.6, "elbow_flex": -55.3, "wrist_flex": -73.3,
             "wrist_roll": 2.9, "gripper": 31.9}
    arm = FakeArm(start=start)
    # motors on, then Enter and an answer for each joint, then release
    ok, results, log, said = session(arm, ["", "", "2", "", "1", "", "2", "", "2", "", "1", "", "1", ""])
    assert ok and released_last(arm)
    assert [r["asked"] for r in results] == [10, 10, 10, 10, -10, 10]
    assert {r["joint"]: r["plus_means"] for r in results} == {
        "shoulder_pan": "counterclockwise", "shoulder_lift": "forward", "elbow_flex": "open",
        "wrist_flex": "out", "wrist_roll": "counterclockwise", "gripper": "open"}
    assert all(r["moved"] == pytest.approx(r["asked"], abs=0.05) and r["back_off"] < 0.05 for r in results)
    assert arm.pos == pytest.approx(start, abs=0.05)
    assert "plus is" in said[-1] and "counterclockwise" in said[-1]


def test_show_me_again_repeats_the_move_before_asking_again():
    arm = FakeArm(start={"elbow_flex": -55.3})
    ok, results, log, said = session(arm, ["", "", "r", "2", ""], joints=["elbow_flex"])
    assert ok and results[0]["plus_means"] == "open"
    target = results[0]["start"] + 10
    at_target = [abs(s["goal"] - target) < 0.01 for s in log]
    assert sum(1 for was, now in zip(at_target, at_target[1:]) if now and not was) == 2  # out, back, out


def test_a_joint_that_doesnt_follow_stops_the_run_and_the_arm_is_still_released():
    arm = FakeArm(stuck={"elbow_flex"})
    ok, results, log, said = session(arm, ["", "", "1", "", "1", "", ""])
    assert not ok and released_last(arm)
    assert [(r["joint"], r["ok"]) for r in results] == [
        ("shoulder_pan", True), ("shoulder_lift", True), ("elbow_flex", False)]
    assert any("elbow didn't follow" in s for s in said)


def test_a_jump_as_the_motors_come_on_stops_before_any_move():
    arm = FakeArm(jolt={"wrist_flex": 6.0})
    ok, results, log, said = session(arm, ["", ""])
    assert not ok and log == [] and released_last(arm)
    assert any("wrist flex shifted 6.0°" in s for s in said)


def test_ctrl_c_mid_move_leaves_the_arm_holding_until_its_released():
    arm = FakeArm()
    follow, calls = arm.send_action, []

    def interrupted(action):
        calls.append(action)
        if len(calls) == 5:
            raise KeyboardInterrupt
        return follow(action)

    arm.send_action = interrupted
    ok, results, log, said = session(arm, ["", "", ""])
    assert not ok and results == [] and released_last(arm)
    assert any("holding where it is" in s for s in said)


def test_it_asks_for_the_base_to_face_forward_before_the_motors_come_on():
    arm = FakeArm(start={"shoulder_pan": 120.0})

    def turn_base():
        arm.pos["shoulder_pan"] = -5.0
        return ""

    ok, results, log, said = session(arm, [turn_base, "q"])
    prompts = [e for e in arm.events if e.startswith("ask:")]
    assert "turned 120° from forward" in prompts[0] and "motors will come on" in prompts[1]
    assert not ok and "torque on" not in arm.events


def test_nothing_powers_on_when_the_servos_dont_match_the_calibration():
    arm = FakeArm(calibrated=False)
    ok, results, log, said = session(arm, [])
    assert not ok and arm.events == []
    assert "lerobot-calibrate" in said[-1]
