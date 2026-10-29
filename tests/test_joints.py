"""The joint map, against the arm's real calibration and the signs measured by hand on Oct 28."""
import copy
import json
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from wbc.config import ROOT
from wbc.motion import joints
from wbc.motion.joints import ARM, JOINTS, JointMap, Servo
from wbc.motion.kinematics import load_chain, pose_to_task

# The arm's calibration as lerobot-calibrate wrote it (Oct 27; the gripper's closed end moved to its stop on Oct 28)
CALIBRATION = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "homing_offset": -155, "range_min": 820, "range_max": 3545},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "homing_offset": 1531, "range_min": 702, "range_max": 3355},
    "elbow_flex": {"id": 3, "drive_mode": 0, "homing_offset": -1831, "range_min": 889, "range_max": 3097},
    "wrist_flex": {"id": 4, "drive_mode": 0, "homing_offset": 1712, "range_min": 851, "range_max": 3163},
    "wrist_roll": {"id": 5, "drive_mode": 0, "homing_offset": 286, "range_min": 0, "range_max": 4095},
    "gripper": {"id": 6, "drive_mode": 0, "homing_offset": 31, "range_min": 1278, "range_max": 2827},
}
# Read off the arm on Oct 28 with the motors off, as each joint was turned by hand in a pose where the move reads
# clearly: the raw positions before, and where the turned joint ended up.
HAND_TEST = {
    # at rest, the base turned about 30° counterclockwise seen from above
    "shoulder_pan": ({"shoulder_pan": 2100, "shoulder_lift": 783, "elbow_flex": 3095, "wrist_flex": 1160,
                      "wrist_roll": 3060, "gripper": 1343}, 1854),
    # at rest, the upper arm tipped about 20° forward, toward where the arm reaches
    "shoulder_lift": ({"shoulder_pan": 2105, "shoulder_lift": 783, "elbow_flex": 3095, "wrist_flex": 1153,
                       "wrist_roll": 3059, "gripper": 1298}, 958),
    # upper arm about upright, forearm about level: the elbow folded about 20°
    "elbow_flex": ({"shoulder_pan": 2106, "shoulder_lift": 2076, "elbow_flex": 2062, "wrist_flex": 1112,
                    "wrist_roll": 3058, "gripper": 1298}, 2301),
    # forearm level, wrist straight: the gripper tipped about 30° down
    "wrist_flex": ({"shoulder_pan": 2105, "shoulder_lift": 2032, "elbow_flex": 2096, "wrist_flex": 2046,
                    "wrist_roll": 4067, "gripper": 1298}, 2513),
    # the same pose, rolled about 45° clockwise looking along the gripper from the wrist toward the jaw tips
    "wrist_roll": ({"shoulder_pan": 2107, "shoulder_lift": 2110, "elbow_flex": 2028, "wrist_flex": 2085,
                    "wrist_roll": 4065, "gripper": 1298}, 3408),
}
REST = HAND_TEST["shoulder_pan"][0]
L_POSE = {"shoulder_pan": 2105, "shoulder_lift": 2066, "elbow_flex": 2060, "wrist_flex": 2042, "wrist_roll": 4071,
          "gripper": 1298}  # lerobot-calibrate's Part 1 pose, held by hand on Oct 28


@pytest.fixture(scope="module")
def robot_cfg():
    return yaml.safe_load((ROOT / "config" / "robot.yaml").read_text())


@pytest.fixture(scope="module")
def chain(robot_cfg):
    return load_chain(robot_cfg, ROOT)


@pytest.fixture(scope="module")
def joint_map(robot_cfg, chain):
    return JointMap.from_config(robot_cfg, CALIBRATION, chain)


def turned(joint):
    """The hand test's pose before `joint` was turned, and the same pose with only that joint moved."""
    before, raw = HAND_TEST[joint]
    return before, {**before, joint: raw}


# What each hand-test move did to the arm in the URDF, seen through forward kinematics. Each is positive for the
# way the hand turned the joint.

def turn_from_above(chain, q0, q1) -> float:
    """Degrees the tool point turned about the pan axis, counterclockwise seen from above."""
    pan = chain.frames(q0)[1][0][0]
    (x0, y0), (x1, y1) = chain.fk(q0)[:2, 3] - pan[:2], chain.fk(q1)[:2, 3] - pan[:2]
    return math.degrees(math.atan2(x0 * y1 - y0 * x1, x0 * x1 + y0 * y1))


def tip_forward(chain, q0, q1) -> float:
    """Degrees the upper arm, from the shoulder axis to the elbow axis, tipped from straight up toward +x, the way
    the arm reaches. At rest it leans back a little past level, so tipping it forward mostly lifts the elbow: the
    elbow's x hardly changes, so that can't show the direction."""
    def lean(q):
        _, axes = chain.frames(q)
        x, _, z = axes[2][0] - axes[1][0]
        return math.degrees(math.atan2(x, z))

    return lean(q1) - lean(q0)


def elbow_closing(chain, q0, q1) -> float:
    """Degrees the angle between the upper arm and the forearm, at the elbow axis, closed."""
    def opening(q):
        _, axes = chain.frames(q)
        upper, fore = axes[1][0] - axes[2][0], axes[3][0] - axes[2][0]
        return math.degrees(math.acos(upper @ fore / np.linalg.norm(upper) / np.linalg.norm(fore)))

    return opening(q0) - opening(q1)


def tool_drop(chain, q0, q1) -> float:
    """Millimetres the tool point went down."""
    return (chain.fk(q0)[2, 3] - chain.fk(q1)[2, 3]) * 1000


def roll_clockwise_from_the_wrist(chain, q0, q1) -> float:
    """Degrees the jaw turned clockwise for someone looking along the gripper from the wrist toward the jaw tips."""
    T0, T1 = chain.fk(q0), chain.fk(q1)
    toward_tips = T0[:3, 0]  # the tool's approach axis
    across0, across1 = T0[:3, 2], T1[:3, 2]  # a direction fixed across the jaw
    # positive about the way you look is clockwise to you
    return math.degrees(math.atan2(np.cross(across0, across1) @ toward_tips, across0 @ across1))


SEEN = {"shoulder_pan": turn_from_above, "shoulder_lift": tip_forward, "elbow_flex": elbow_closing,
        "wrist_flex": tool_drop, "wrist_roll": roll_clockwise_from_the_wrist}


def test_robot_yaml_gives_ids_1_to_6_a_plus_sign_on_every_arm_joint_and_the_elbow_p_32(robot_cfg, joint_map):
    assert [joint_map.servos[m].id for m in JOINTS] == [1, 2, 3, 4, 5, 6]
    assert [joint_map.servos[m].sign for m in ARM] == [1] * 5
    assert robot_cfg["gains"] == {"elbow_flex": 32}


def test_raw_2047_is_the_urdfs_zero_on_every_joint(joint_map):
    assert [joint_map.angle(m, 2047) for m in JOINTS] == [0.0] * 6
    assert np.array_equal(joint_map.q({m: 2047 for m in JOINTS}), np.zeros(5))
    assert joint_map.raw(np.zeros(5)) == {m: 2047 for m in ARM}


def test_raw_and_urdf_angles_round_trip_for_every_tick_in_range(joint_map):
    for m in ("shoulder_pan", "elbow_flex", "wrist_roll", "gripper"):
        s = joint_map.servos[m]
        assert all(joint_map.ticks(m, joint_map.angle(m, r)) == r for r in range(s.range_min, s.range_max + 1))
    assert joint_map.raw(joint_map.q(REST)) == {m: REST[m] for m in ARM}
    assert all(isinstance(v, int) for v in joint_map.raw(joint_map.q(REST)).values())


def test_an_angle_between_ticks_goes_to_the_nearest_tick(joint_map):
    tick = 2 * math.pi / 4096
    assert joint_map.ticks("wrist_flex", 100.4 * tick) == 2147
    assert joint_map.ticks("wrist_flex", 100.6 * tick) == 2148
    with pytest.raises(ValueError, match="one angle for each of the 5 arm joints"):
        joint_map.raw(np.zeros(6))


def test_a_servo_mounted_the_other_way_round_reads_backwards_and_still_round_trips(joint_map):
    mirrored = JointMap({**joint_map.servos, "elbow_flex": replace(joint_map.servos["elbow_flex"], sign=-1)})
    assert mirrored.angle("elbow_flex", 2301) == -joint_map.angle("elbow_flex", 2301)
    assert all(mirrored.ticks("elbow_flex", mirrored.angle("elbow_flex", r)) == r for r in range(889, 3098))
    assert [math.degrees(v) for v in mirrored.limits("elbow_flex")] == pytest.approx([-92.3, 101.8], abs=0.05)


def test_the_base_turned_counterclockwise_by_hand_turns_counterclockwise_seen_from_above(joint_map, chain):
    before, after = turned("shoulder_pan")
    assert turn_from_above(chain, joint_map.q(before), joint_map.q(after)) == pytest.approx(21.6, abs=0.1)


def test_the_upper_arm_tipped_forward_by_hand_tips_forward(joint_map, chain):
    before, after = turned("shoulder_lift")
    assert tip_forward(chain, joint_map.q(before), joint_map.q(after)) == pytest.approx(15.4, abs=0.5)


def test_the_elbow_folded_by_hand_closes_the_elbow_angle(joint_map, chain):
    before, after = turned("elbow_flex")
    assert elbow_closing(chain, joint_map.q(before), joint_map.q(after)) == pytest.approx(21.0, abs=0.5)


def test_the_gripper_tipped_down_by_hand_at_the_level_pose_lowers_the_tool_point(joint_map, chain):
    before, after = turned("wrist_flex")
    q0, q1 = joint_map.q(before), joint_map.q(after)
    assert tool_drop(chain, q0, q1) == pytest.approx(105, abs=1)  # from 228 mm above the table to 123
    pitch = math.degrees(pose_to_task(chain.fk(q1))[3] - pose_to_task(chain.fk(q0))[3])
    assert pitch == pytest.approx(41.0, abs=0.5)  # positive pitch points the tool down


def test_the_wrist_rolled_clockwise_by_hand_turns_the_jaw_clockwise_seen_from_the_wrist(joint_map, chain):
    before, after = turned("wrist_roll")
    assert roll_clockwise_from_the_wrist(chain, joint_map.q(before), joint_map.q(after)) == \
        pytest.approx(57.7, abs=0.5)


@pytest.mark.parametrize("joint", ARM)
def test_with_its_sign_flipped_each_hand_test_move_reads_the_wrong_way(joint_map, chain, joint):
    flipped = JointMap({**joint_map.servos, joint: replace(joint_map.servos[joint], sign=-1)})
    before, after = turned(joint)
    assert SEEN[joint](chain, flipped.q(before), flipped.q(after)) < 0


def test_the_l_pose_held_by_hand_reads_within_2_degrees_of_the_urdfs_zero(joint_map):
    lift, elbow, wrist_flex = np.degrees(joint_map.q(L_POSE))[1:4]
    assert max(abs(lift), abs(elbow), abs(wrist_flex)) < 2  # the base and the roll can't be judged by eye


def test_the_rest_pose_reads_folded_with_the_gripper_pointing_up(joint_map, chain):
    T, axes = chain.frames(joint_map.q(REST))
    assert T[2, 0] > 0.99  # the approach axis points up, as it does on the arm
    assert [round(axes[i][0][2] * 1000) for i in (2, 3, 4)] == [102, 151, 212]  # mm: elbow, wrist flex, wrist roll


def test_limits_are_the_calibrated_ranges_in_urdf_degrees_and_the_wrist_roll_gets_the_urdfs(joint_map, chain):
    expected = {"shoulder_pan": [-107.8, 131.7], "shoulder_lift": [-118.2, 115.0], "elbow_flex": [-101.8, 92.3],
                "wrist_flex": [-105.1, 98.1], "wrist_roll": [-157.2, 162.8]}
    for m, ends in expected.items():
        assert [math.degrees(v) for v in joint_map.limits(m)] == pytest.approx(ends, abs=0.05)
    assert joint_map.limits("wrist_roll") == (chain.lower[4], chain.upper[4])
    assert JointMap(joint_map.servos).limits("wrist_roll") == (-math.pi, math.pi)  # without the URDF's, a full turn


def test_only_the_wrist_roll_has_no_stop(joint_map):
    assert [m for m in JOINTS if joint_map.servos[m].full_turn] == ["wrist_roll"]


def test_lerobot_degrees_match_lerobots_formula(joint_map):
    at_2047 = {"shoulder_pan": -11.912, "shoulder_lift": 1.626, "elbow_flex": 4.747, "wrist_flex": 3.516,
               "wrist_roll": -0.044}  # what LeRobot 0.6.1 reads at raw 2047
    assert {m: joint_map.lerobot_degrees(m, 2047) for m in at_2047} == pytest.approx(at_2047, abs=0.0005)
    # 0 at the middle of the calibrated range, 4095 ticks to 360°, and not clamped past an end
    assert joint_map.lerobot_degrees("wrist_flex", (851 + 3163) / 2) == 0
    assert joint_map.lerobot_degrees("wrist_flex", 826) == pytest.approx((826 - 2007) * 360 / 4095)


def test_gripper_percent_is_lerobots_0_to_100_clamped_to_the_range(joint_map):
    assert joint_map.gripper_percent(1278) == 0 and joint_map.gripper_percent(2827) == 100
    assert joint_map.gripper_percent(1270) == 0 and joint_map.gripper_percent(2900) == 100
    assert joint_map.gripper_percent(1348) == pytest.approx(4.52, abs=0.005)  # closed by hand on the board, Oct 28


def test_outside_says_how_far_past_its_range_each_joint_is(joint_map):
    lowered = {**REST, "wrist_flex": 826, "gripper": 1270}  # as the arm can sit after being lowered to rest by hand
    assert joint_map.outside(lowered) == pytest.approx({"wrist_flex": 2.197, "gripper": 0.703}, abs=0.001)
    assert joint_map.outside(REST) == {}
    assert joint_map.outside({"wrist_flex": 851, "elbow_flex": 3097}) == {}  # on an end is inside
    assert joint_map.outside({"elbow_flex": 3100}) == pytest.approx({"elbow_flex": 0.264}, abs=0.001)


@pytest.mark.parametrize("spoil, match", [
    (lambda cfg, cal: cfg["servos"].pop("elbow_flex"), "the elbow is missing from the servos in config/robot.yaml"),
    (lambda cfg, cal: cal.pop("gripper"), "the gripper is missing from the calibration"),
    (lambda cfg, cal: cfg["servos"]["shoulder_pan"].pop("sign"), "the base needs a sign of 1 or -1"),
    (lambda cfg, cal: cfg["servos"]["wrist_flex"].update(sign=2),
     "the wrist flex needs a sign of 1 or -1 in config/robot.yaml, not 2"),
    (lambda cfg, cal: cfg["servos"]["gripper"].pop("id"), "the gripper needs a servo id"),
    (lambda cfg, cal: cfg["servos"]["wrist_roll"].update(id=4),
     "the wrist roll is servo 4 in config/robot.yaml but servo 5 in the calibration"),
    (lambda cfg, cal: cal["shoulder_lift"].update(drive_mode=1), "the shoulder lift's calibration has drive_mode 1"),
    (lambda cfg, cal: cal["elbow_flex"].update(range_min=3097), "the elbow's calibrated range runs from 3097 to 3097"),
    (lambda cfg, cal: cal["gripper"].pop("range_max"), "the calibration has no range for the gripper"),
], ids=["joint not in the config", "joint not calibrated", "no sign", "sign not 1 or -1", "no id",
        "ids disagree", "flipped drive mode", "empty range", "no range"])
def test_from_config_refuses_a_map_that_doesnt_add_up(robot_cfg, spoil, match):
    cfg, cal = copy.deepcopy(robot_cfg), copy.deepcopy(CALIBRATION)
    spoil(cfg, cal)
    with pytest.raises(ValueError, match=match):
        JointMap.from_config(cfg, cal)


def test_from_config_refuses_a_chain_that_isnt_the_arms(robot_cfg, joint_map):
    toy = SimpleNamespace(names=["pan", "lift"], lower=np.array([-2.0, -1.5]), upper=np.array([2.0, 1.5]))
    with pytest.raises(ValueError, match="the chain's joints are pan, lift"):
        JointMap.from_config(robot_cfg, CALIBRATION, toy)
    with pytest.raises(ValueError, match="none for the gripper"):
        JointMap({m: s for m, s in joint_map.servos.items() if m != "gripper"})


def test_from_config_takes_lerobots_records_or_just_the_ranges_read_off_the_servos(robot_cfg, joint_map):
    records = {m: SimpleNamespace(**c) for m, c in CALIBRATION.items()}  # like LeRobot's MotorCalibration
    assert JointMap.from_config(robot_cfg, records).servos == joint_map.servos
    ranges = {m: {"range_min": c["range_min"], "range_max": c["range_max"]} for m, c in CALIBRATION.items()}
    assert JointMap.from_config(robot_cfg, ranges).servos == joint_map.servos  # no ids or drive modes to check


def test_load_calibration_reads_the_file_lerobot_calibrate_writes(tmp_path, robot_cfg):
    path = tmp_path / f"{joints.CALIBRATION_ID}.json"
    path.write_text(json.dumps(CALIBRATION, indent=4))
    calibration = joints.load_calibration(path)
    assert calibration == CALIBRATION
    assert JointMap.from_config(robot_cfg, calibration).servos["gripper"] == Servo("gripper", 6, 1, 1278, 2827)


def test_calibration_dir_follows_lerobots_environment_chain(monkeypatch, tmp_path):
    for name in ("HF_LEROBOT_CALIBRATION", "HF_LEROBOT_HOME", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    follower = ("robots", "so_follower")
    assert joints.calibration_dir() == tmp_path.joinpath(".cache", "huggingface", "lerobot", "calibration", *follower)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert joints.calibration_dir() == tmp_path.joinpath("xdg", "huggingface", "lerobot", "calibration", *follower)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert joints.calibration_dir() == tmp_path.joinpath("hf", "lerobot", "calibration", *follower)
    monkeypatch.setenv("HF_LEROBOT_HOME", "~/lerobot")  # LeRobot expands ~, as here
    assert joints.calibration_dir() == tmp_path.joinpath("lerobot", "calibration", *follower)
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path / "calibration"))
    assert joints.calibration_dir() == tmp_path.joinpath("calibration", *follower)
