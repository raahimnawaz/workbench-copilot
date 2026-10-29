"""Registering the arm to the carrier, and the pose that points the gripper at a hole, on the real URDF."""
import math

import numpy as np
import pytest
import yaml

from test_joints import CALIBRATION, REST
from wbc.config import ROOT
from wbc.motion import carrier
from wbc.motion.arm import FLOOR_JOINTS, JOINT_FLOOR, TOOL_FLOOR, Refused
from wbc.motion.carrier import FLIP, MISS, Registration, aim, fit, on_line, point_at, point_mm, trouble
from wbc.motion.joints import JointMap
from wbc.motion.kinematics import load_chain, rot_z
from wbc.motion.trajectory import lowest

ROBOT_CFG = yaml.safe_load((ROOT / "config" / "robot.yaml").read_text())
BENCH = yaml.safe_load((ROOT / "config" / "bench.yaml").read_text())
PARTS = yaml.safe_load((ROOT / "config" / "parts.yaml").read_text())
CHAIN = load_chain(ROBOT_CFG, ROOT)
MAP = JointMap.from_config(ROBOT_CFG, CALIBRATION, CHAIN)
CORNERS = ["tag10", "tag11", "tag12", "tag13"]


def placed(yaw_deg=-86.0, middle=(0.23, 0.0, 0.0)) -> Registration:
    """The sheet in front of the arm, its middle `middle` in the arm's frame, its x axis turned yaw_deg from the arm's
    straight ahead: about how it lies on Raahim's bench, with the arm at the sheet's top edge."""
    yaw = math.radians(yaw_deg)
    return Registration(yaw, np.asarray(middle) - rot_z(yaw) @ FLIP @ np.array([139.7, 107.95, 0.0]) / 1000)


TRUE = placed()


def touches(names, reg=TRUE, off=None):
    """Carrier mm and tool points for touches exactly on each named point, or `off` (metres, arm frame) away."""
    mm = np.array([point_mm(BENCH, PARTS, n) for n in names])
    tool = np.array([reg.to_arm(m) for m in mm])
    if off:
        for name, d in off.items():
            tool[names.index(name)] += d
    return mm, tool


def test_a_fit_finds_the_turn_and_shift_it_was_made_with():
    names = CORNERS + ["a1", "j63"]
    reg = fit(names, *touches(names))
    assert math.degrees(reg.yaw) == pytest.approx(-86.0, abs=1e-9)
    assert reg.shift == pytest.approx(TRUE.shift, abs=1e-12)
    assert reg.rms_mm == pytest.approx(0, abs=1e-9) and reg.scale == pytest.approx(1)


def test_carrier_y_runs_down_the_page_so_the_frame_stays_right_handed():
    """Unturned, a point right on the sheet is +x in the arm's frame, and one down the page is -y: the sheet seen from
    above, as the camera sees it."""
    reg = Registration(0.0, np.zeros(3))
    assert reg.to_arm([10, 20, 5]) == pytest.approx([0.010, -0.020, 0.005])
    assert reg.to_carrier(reg.to_arm([12, 34, 5])) == pytest.approx([12, 34, 5])


def test_a_point_touched_off_its_mark_shows_up_as_the_worst():
    names = CORNERS + ["a1"]
    reg = fit(names, *touches(names, off={"tag12": [0.004, 0.0, 0.0]}))
    assert reg.worst["name"] == "tag12" and reg.worst["error_mm"] > 2
    assert math.degrees(reg.yaw) == pytest.approx(-86.0, abs=1.0)


def test_a_sheet_printed_too_large_shows_in_the_scale():
    names = CORNERS
    mm, tool = touches(names)
    middle = tool.mean(axis=0)
    reg = fit(names, mm, middle + 1.03 * (tool - middle))
    assert reg.scale == pytest.approx(1.03, abs=1e-3)


@pytest.mark.parametrize("names, message", [
    (["tag10", "tag11"], "at least 3"),
    (["tag10", "tag11", "tag10", "tag12"], "touched twice: tag10"),
    (["a1", "a20", "a40", "a63"], "one line"),
], ids=["too-few", "twice", "in-a-line"])
def test_a_fit_refuses_points_that_cant_fix_the_carrier(names, message):
    with pytest.raises(ValueError, match=message):
        fit(names, *touches(names))


def test_a_registration_round_trips_through_its_file(tmp_path):
    reg = fit(CORNERS, *touches(CORNERS))
    reg.save(tmp_path / "r.json")
    back = Registration.load(tmp_path / "r.json")
    assert back.yaw == reg.yaw and back.shift == pytest.approx(reg.shift) and back.points == reg.points


def test_a_missing_registration_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="register_arm.py touch"):
        Registration.load(tmp_path / "none.json")


def test_tags_sit_on_the_sheet_and_holes_on_the_boards_top():
    assert point_mm(BENCH, PARTS, "tag12") == pytest.approx([239.4, 153.9, 0])
    a1 = BENCH["boards"]["main"]["a1_mm"]
    assert point_mm(BENCH, PARTS, "a1") == pytest.approx([*a1, BENCH["carrier"]["board_top_mm"]])
    # rows grow to the right on the sheet and the letters run up it, so b2 is a pitch right and a pitch up from a1
    assert point_mm(BENCH, PARTS, "b2")[:2] == pytest.approx([a1[0] + 2.54, a1[1] - 2.54])


@pytest.mark.parametrize("name", ["tag9", "tagx", "k5", "a0", "a64", "a+", "5a", ""])
def test_a_name_thats_not_a_tag_or_a_hole_is_refused(name):
    with pytest.raises(ValueError):
        point_mm(BENCH, PARTS, name)


@pytest.mark.parametrize("name", CORNERS + ["a1", "j1", "a63", "j63", "a5", "a26"])
def test_aim_points_the_gripper_through_the_spot_from_hover_back(name):
    spot = TRUE.to_arm(point_mm(BENCH, PARTS, name))
    a = aim(CHAIN, MAP, spot, MAP.q(REST))
    for q, back in ((a.q, carrier.HOVER), (a.q_above, carrier.HOVER + carrier.ABOVE)):
        along, miss = on_line(CHAIN, q, spot)
        assert along == pytest.approx(back, abs=MISS) and miss <= MISS
        assert trouble(CHAIN, MAP, q) == ""
        tool, axis = lowest(CHAIN, q, FLOOR_JOINTS)
        assert tool >= TOOL_FLOOR and axis >= JOINT_FLOOR
    assert a.pitch >= 70  # near straight down, everywhere on the board


def test_aim_tries_the_pitches_steepest_first(monkeypatch):
    spot = TRUE.to_arm(point_mm(BENCH, PARTS, "tag13"))
    assert aim(CHAIN, MAP, spot, MAP.q(REST)).pitch == 90  # straight down works there, so it's the one taken
    monkeypatch.setattr(carrier, "PITCHES", (60, 50))
    a = aim(CHAIN, MAP, spot, MAP.q(REST))
    assert a.pitch == 60 and on_line(CHAIN, a.q, spot)[1] <= MISS


@pytest.mark.parametrize("name", ["a5", "j63", "tag10"])
def test_aim_keeps_the_wrist_roll_near_where_it_is(name):
    """Pointing doesn't care how the jaws are turned about its line, and the roll's cable does. From the rest pose on
    Oct 29 (roll -149.9°), a pose at roll 0 would first turn the gripper about 150°."""
    rest = np.radians([6.33, -110.57, 89.3, -92.55, -149.85])
    spot = TRUE.to_arm(point_mm(BENCH, PARTS, name))
    a = aim(CHAIN, MAP, spot, rest)
    assert abs(math.degrees(a.q[4]) + 149.85) < 20 and abs(math.degrees(a.q_above[4]) + 149.85) < 20
    assert on_line(CHAIN, a.q, spot)[1] <= MISS and trouble(CHAIN, MAP, a.q) == ""


@pytest.mark.parametrize("spot, message", [
    ([0.6, 0.0, 0.0], "out of reach"),
    ([0.03, 0.0, 0.01], "can't point at it"),
], ids=["far", "at-the-base"])
def test_aim_says_why_it_cant_point(spot, message):
    with pytest.raises(ValueError, match=message):
        aim(CHAIN, MAP, spot, MAP.q(REST))


def test_trouble_names_a_joint_outside_its_range_and_a_tool_point_under_the_floor():
    q = np.zeros(5)
    q[1] = math.radians(150)
    assert "shoulder lift" in trouble(CHAIN, MAP, q)
    low = carrier._toward(CHAIN, TRUE.to_arm([150, 100, 4]), 0.0, math.radians(85), MAP.q(REST))[0]  # 4 mm up
    assert "tool point would be 4 mm above the table" in trouble(CHAIN, MAP, low)


class FakeArm:
    """What point_at uses of the driver: the chain, joint map, a reading, and move_to, which it records."""

    def __init__(self, refuse_at=None):
        self.chain, self.joints, self.moves, self.refuse_at = CHAIN, MAP, [], refuse_at

    def read(self):
        return type("Reading", (), {"q": MAP.q(REST)})()

    def move_to(self, q):
        if len(self.moves) == self.refuse_at:
            raise Refused("the wrist roll isn't powered: power the arm on first")
        self.moves.append(np.asarray(q))


def test_point_at_goes_to_the_far_pose_on_the_line_then_in():
    arm = FakeArm()
    a = point_at(arm, TRUE, point_mm(BENCH, PARTS, "a5"))
    assert len(arm.moves) == 2
    assert arm.moves[0] == pytest.approx(a.q_above) and arm.moves[1] == pytest.approx(a.q)


def test_a_refusal_from_the_driver_comes_straight_through_and_stops_there():
    arm = FakeArm(refuse_at=0)
    with pytest.raises(Refused, match="power the arm on"):
        point_at(arm, TRUE, point_mm(BENCH, PARTS, "a5"))
    assert arm.moves == []
