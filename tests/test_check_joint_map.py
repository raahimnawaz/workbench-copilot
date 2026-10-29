"""The read-only joint map check, against fake servos sharing one bus line."""
import copy
import importlib.util
import json

import numpy as np
import pytest
import yaml

from test_joints import CALIBRATION, HAND_TEST, L_POSE, REST, SEEN
from test_sts import FakeServo
from wbc.config import ROOT
from wbc.motion import sts
from wbc.motion.joints import ARM, JOINTS, JointMap
from wbc.motion.kinematics import load_chain

_spec = importlib.util.spec_from_file_location("check_joint_map", ROOT / "scripts" / "check_joint_map.py")
check_joint_map = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_joint_map)

ROBOT_CFG = yaml.safe_load((ROOT / "config" / "robot.yaml").read_text())
CHAIN = load_chain(ROBOT_CFG, ROOT)
MAP = JointMap.from_config(ROBOT_CFG, CALIBRATION, CHAIN)


class FakeLine:
    """The arm's six servos on one bus line, each answering only its own id, with the limits lerobot-calibrate wrote.
    Every request is kept, so a test can check nothing was written."""

    def __init__(self, raw=None, torque=()):
        self.servos = {}
        for i, m in enumerate(JOINTS, 1):
            servo = FakeServo(servo_id=i)
            servo.set_u16(sts.MIN_POSITION_LIMIT, CALIBRATION[m]["range_min"])
            servo.set_u16(sts.MAX_POSITION_LIMIT, CALIBRATION[m]["range_max"])
            servo.regs[sts.TORQUE_ENABLE] = int(m in torque)
            self.servos[m] = servo
        self.requests, self.out = [], bytearray()
        self.pose(raw or REST)

    def pose(self, raw):
        """The arm moved by hand to raw positions {joint: ticks}."""
        for m, value in raw.items():
            self.servos[m].set_u16(sts.FEEDBACK_START, value)

    def reset_input_buffer(self):
        self.out.clear()

    def write(self, data):
        self.requests.append(bytes(data))
        for servo in self.servos.values():
            servo.write(data)
            self.out += servo.out
            servo.out.clear()

    def read(self, n):
        chunk = bytes(self.out[:n])
        del self.out[:n]
        return chunk

    def close(self):
        pass

    def written(self):
        """Every request that was neither a ping nor a read."""
        return [r for r in self.requests if r[4] not in (sts.PING, sts.READ)]


def held(line, raw):
    """A reply that first moves the arm by hand to `raw`, then presses Enter."""
    def act():
        line.pose(raw)
        return ""
    return act


def hand_test(line, backwards=None):
    """Replies for a check that repeats the Oct 28 moves, and holds the L pose. The joint named `backwards` reads as
    if its servo were mounted the other way round."""
    replies = []
    for joint, *_ in check_joint_map.MOVES:
        before, raw = HAND_TEST[joint]
        after = {**before, joint: 2 * before[joint] - raw if joint == backwards else raw}
        replies += [held(line, before), held(line, after)]
    return replies + [held(line, L_POSE)]


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def session(line, replies=(), mode="check", calibration=None, robot_cfg=None, sleep=None):
    """Run the script's session on the fake line with scripted replies to its prompts."""
    replies, said, record, clock = list(replies), [], {}, Clock()

    def ask(prompt):
        said.append(prompt)
        if not replies:
            raise EOFError
        reply = replies.pop(0)
        return reply() if callable(reply) else reply

    bus = sts.Bus("fake", serial_factory=lambda port, **kwargs: line)
    ok = check_joint_map.run(bus, mode, robot_cfg or ROBOT_CFG, record, calibration=calibration, ask=ask,
                             say=lambda text="", **kwargs: said.append(text), sleep=sleep or clock.sleep, clock=clock,
                             seconds=1.0)
    return ok, record, said


def summary(said):
    """The summary table's rows, {name: result}."""
    rows = "\n".join(said).split("result\n", 1)[1].split("\n\n")[0].splitlines()
    return {row[:14].strip(): row[37:].strip() for row in rows}


@pytest.mark.parametrize("joint, setup, action, way", check_joint_map.MOVES, ids=[m for m, *_ in check_joint_map.MOVES])
def test_each_move_asked_for_turns_the_urdf_angle_the_way_the_script_expects(joint, setup, action, way):
    """In the pose the hand test started from, a small URDF turn the script's way is the move it asks for, as forward
    kinematics shows it: counterclockwise from above, forward, fold, gripper down, clockwise from the wrist."""
    q0 = MAP.q(HAND_TEST[joint][0])
    q1 = q0 + way * np.radians(10) * np.eye(len(ARM))[ARM.index(joint)]
    assert SEEN[joint](CHAIN, q0, q1) > 0


def test_a_check_repeating_the_oct_28_moves_passes_every_move_and_the_l_pose():
    line = FakeLine()
    ok, record, said = session(line, hand_test(line))
    assert ok and line.written() == []
    turned = {r["joint"]: r["turned"] for r in record["moves"]}
    assert turned == pytest.approx({"shoulder_pan": -21.6, "shoulder_lift": 15.4, "elbow_flex": 21.0,
                                    "wrist_flex": 41.0, "wrist_roll": -57.7}, abs=0.05)
    assert all(r["ok"] for r in record["moves"]) and record["l_pose"]["ok"]
    assert said[-1].startswith("\nThe joint map matches the arm")
    assert any("The shoulder lift reads +1.7°, the elbow +1.1° and the wrist flex -0.4°: within 8°" in s
               for s in said)


def test_a_joint_whose_raw_runs_backwards_fails_its_move_and_the_check():
    line = FakeLine()
    ok, record, said = session(line, hand_test(line, backwards="elbow_flex"))
    assert not ok and line.written() == []
    assert [r["joint"] for r in record["moves"] if not r["ok"]] == ["elbow_flex"]
    assert any("The elbow's URDF angle went -21.0°" in s and "should go plus, so its sign in config/robot.yaml looks "
               "wrong" in s for s in said)
    assert "the elbow turned the wrong way" in said[-1]
    assert summary(said) == {"base": "right", "shoulder lift": "right", "elbow": "WRONG", "wrist flex": "right",
                             "wrist roll": "right", "L pose": "right"}


def test_a_wrong_sign_in_robot_yaml_fails_that_joints_move():
    robot_cfg = copy.deepcopy(ROBOT_CFG)
    robot_cfg["servos"]["wrist_flex"]["sign"] = -1
    line = FakeLine()
    ok, record, said = session(line, hand_test(line), robot_cfg=robot_cfg)
    assert not ok and [r["joint"] for r in record["moves"] if not r["ok"]] == ["wrist_flex"]


def test_a_roll_across_the_encoder_seam_reads_the_short_way_round():
    line = FakeLine()
    replies = hand_test(line)
    before = {**HAND_TEST["wrist_roll"][0], "wrist_roll": 100}
    replies[8:10] = [held(line, before), held(line, {**before, "wrist_roll": 3684})]  # 45° clockwise, across raw 0
    ok, record, said = session(line, replies)
    assert ok and record["moves"][-1]["turned"] == pytest.approx(-45.0, abs=0.1)


class GlitchyLine(FakeLine):
    """The wrist flex's first position reading after each move by hand comes back 0, once."""

    def pose(self, raw):
        super().pose(raw)
        self.glitch = True

    def write(self, data):
        servo = self.servos["wrist_flex"]
        if getattr(self, "glitch", False) and data[2] == 4 and data[4] == sts.READ and data[5] == sts.FEEDBACK_START:
            self.glitch = False
            kept = bytes(servo.regs[sts.FEEDBACK_START:sts.FEEDBACK_START + 2])
            servo.set_u16(sts.FEEDBACK_START, 0)
            super().write(data)
            servo.regs[sts.FEEDBACK_START:sts.FEEDBACK_START + 2] = kept
            return
        super().write(data)


def test_one_glitched_position_reading_doesnt_change_a_verdict():
    line = GlitchyLine()
    ok, record, said = session(line, hand_test(line))
    assert ok and all(r["ok"] for r in record["moves"]) and record["l_pose"]["ok"]


def test_a_move_under_5_degrees_is_asked_for_again():
    line = FakeLine()
    before, raw = HAND_TEST["shoulder_pan"]
    nudged = {**before, "shoulder_pan": before["shoulder_pan"] - 30}  # 2.6°
    ok, record, said = session(line, [held(line, before), held(line, nudged)] + hand_test(line))
    assert ok and any("turned only -2.6°, under the 5° this needs" in s for s in said)
    assert sum(s.startswith("\nBASE.") for s in said) == 2 and len(record["moves"]) == 5


def test_a_check_that_isnt_finished_doesnt_pass():
    line = FakeLine()
    before, raw = HAND_TEST["shoulder_pan"]
    ok, record, said = session(line, [held(line, before), held(line, {**before, "shoulder_pan": raw}), "s", "q"])
    assert not ok and [r["joint"] for r in record["moves"]] == ["shoulder_pan", "shoulder_lift"]
    assert summary(said) == {"base": "right", "shoulder lift": "skipped", "elbow": "not run", "wrist flex": "not run",
                             "wrist roll": "not run", "L pose": "not run"}
    assert said[-1] == ("\nEvery move tried turned the URDF's way, but this isn't a full check, so it isn't a pass. Not "
                        "checked: shoulder lift, elbow, wrist flex, wrist roll, L pose.")


@pytest.mark.parametrize("replies", [["q"], ["s"] * 6], ids=["quit at once", "all skipped"])
def test_a_check_that_checks_nothing_doesnt_pass(replies):
    ok, record, said = session(FakeLine(), replies)
    assert not ok and said[-1] == "\nNothing was checked, so this isn't a pass."


def test_ctrl_c_mid_check_still_gives_the_summary():
    line = FakeLine()
    before, raw = HAND_TEST["shoulder_pan"]

    def ctrl_c():
        raise KeyboardInterrupt

    ok, record, said = session(line, [held(line, before), held(line, {**before, "shoulder_pan": raw}), ctrl_c])
    assert not ok and "\nStopped." in said and summary(said)["base"] == "right"


def test_a_servo_going_silent_mid_check_is_named_and_the_summary_given():
    line = FakeLine()
    before = HAND_TEST["shoulder_pan"][0]

    def unplug():
        line.servos["wrist_roll"].silent = True
        return ""

    ok, record, said = session(line, [held(line, before), unplug])
    assert not ok and "\nStopped: the wrist roll (servo 5): no reply: check the port, servo power and servo id." in said
    assert summary(said)["base"] == "not run"


@pytest.mark.parametrize("joint", ["shoulder_lift", "elbow_flex", "wrist_flex"])
def test_an_l_pose_off_by_more_than_8_degrees_fails(joint):
    line = FakeLine()
    replies = ["s"] * 5 + [held(line, {**L_POSE, joint: L_POSE[joint] + 120})]  # 10.5° off
    ok, record, said = session(line, replies)
    assert not ok and not record["l_pose"]["ok"]
    assert said[-1] == "\nNo move was checked, and the L pose doesn't read near the URDF's zero."


def rows(said):
    """once's table as {joint: [raw, LeRobot, URDF, range...]}: the name takes the first 14 characters."""
    return {s[2:16].strip(): s[16:].split() for s in said if s.startswith("  ") and not s.startswith("  The")}


def test_once_prints_each_joint_and_where_the_tool_point_is():
    line = FakeLine()  # the rest pose of Oct 28
    ok, record, said = session(line, mode="once")
    assert ok and line.written() == []
    table = rows(said)
    assert table["base"] == ["2100", "-7.3°", "+4.7°", "inside"]
    assert table["shoulder lift"] == ["783", "-109.5°", "-111.1°", "inside"]  # LeRobot's 0 is mid-range
    assert table["gripper"] == ["1343", "4.2%", "inside"]
    # the tool point at rest, folded with the gripper pointing up: 309 mm above the table, as measured on Oct 28
    assert said[-1].startswith("  The tool point is at x 60, y -10, z 309 mm,") and said[-1].endswith("83° up.")


def test_once_says_which_joints_are_past_where_moves_may_take_them():
    line = FakeLine({**REST, "wrist_flex": 826, "wrist_roll": 4071})  # lowered to rest by hand; the L pose's roll
    ok, record, said = session(line, mode="once")
    table = rows(said)
    assert table["wrist flex"][0] == "826" and table["wrist flex"][3:] == ["past", "its", "end", "by", "2.2°"]
    assert table["wrist roll"][3:] == ["past", "its", "end", "by", "15.1°"]


def test_watch_reads_until_its_time_is_up_and_writes_nothing():
    line = FakeLine({**REST, "wrist_roll": 4071})  # where the roll read in the L pose: past its cable's limit
    ok, record, said = session(line, mode="watch")
    lines = [s for s in said if s.startswith("\r")]
    assert ok and len(lines) == 5 and line.written() == []
    assert "base   +4.7°" in lines[0] and "PAST ITS END: wrist roll" in lines[0]


def test_watch_ends_cleanly_at_ctrl_c():
    def sleep(seconds):
        raise KeyboardInterrupt

    ok, record, said = session(FakeLine(), mode="watch", sleep=sleep)
    assert ok and said[-1] == ""


def test_it_wont_run_with_a_motor_on():
    line = FakeLine(torque={"elbow_flex"})
    ok, record, said = session(line, mode="check")
    assert not ok and line.written() == []
    assert said == ["Torque is on for the elbow. This check needs the motors off, so don't force the arm. Switch the "
                    "servo power off and on again (they come back with torque off), then run this again."]


def test_a_servo_that_doesnt_answer_is_named():
    line = FakeLine()
    line.servos["wrist_roll"].silent = True
    ok, record, said = session(line, mode="once")
    assert not ok and said[0].startswith("No answer from the wrist roll.")


def test_a_joint_with_a_stop_but_the_factory_limits_is_refused():
    line = FakeLine()
    line.servos["elbow_flex"].set_u16(sts.MIN_POSITION_LIMIT, 0)  # a new servo, not calibrated yet
    line.servos["elbow_flex"].set_u16(sts.MAX_POSITION_LIMIT, 4095)
    ok, record, said = session(line, mode="once")
    assert not ok and said == ["The position limits on the elbow are 0–4095, the whole turn, as a servo comes from the "
                               "factory. Only the wrist roll, which has no stop, should be calibrated that way, so run "
                               "lerobot-calibrate (a new servo?), then this again."]


def test_limits_that_dont_make_a_range_say_to_calibrate_again():
    line = FakeLine()
    line.servos["gripper"].set_u16(sts.MAX_POSITION_LIMIT, 0)
    line.servos["gripper"].set_u16(sts.MIN_POSITION_LIMIT, 0)
    ok, record, said = session(line, mode="once")
    assert not ok and said == ["The joint map can't be built: the gripper's calibrated range runs from 0 to 0, and its "
                               "minimum has to be below its maximum. Run lerobot-calibrate again."]


def test_a_calibration_file_that_differs_from_the_servos_is_reported():
    old = {**CALIBRATION, "gripper": {**CALIBRATION["gripper"], "range_min": 1331}}  # the file before Oct 28's change
    ok, record, said = session(FakeLine(), mode="once", calibration=old)
    assert said[0] == "The gripper's range is 1331–2827 in the calibration file but 1278–2827 in the servo."
    assert "fails until the file matches them" in said[1]
    ok, record, said = session(FakeLine(), mode="once", calibration=CALIBRATION)
    assert not any("calibration file" in s for s in said)


def test_the_check_needs_a_port():
    with pytest.raises(SystemExit):
        check_joint_map.main(["check"])


def test_main_sends_nothing_but_ping_and_read(monkeypatch):
    line = FakeLine()
    real = sts.Bus
    monkeypatch.setattr(sts, "Bus", lambda port, **kwargs: real(port, serial_factory=lambda p, **k: line))
    assert check_joint_map.main(["once", "--port", "/dev/cu.fake"]) == 0
    assert line.written() == []


def test_a_calibration_file_of_the_wrong_shape_is_refused(tmp_path, capsys):
    path = tmp_path / "wbc_follower.json"
    path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(SystemExit):
        check_joint_map.main(["once", "--port", "/dev/cu.fake", "--calibration", str(path)])
    assert "isn't a LeRobot calibration file" in capsys.readouterr().err
