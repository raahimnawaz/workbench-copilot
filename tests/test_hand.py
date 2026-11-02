"""Moves in mm, picking things up and putting them down (wbc/motion/hand.py), and the agent's arm tools, against the
arm tests' fake follower."""
import dataclasses
import io
import math
import queue
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml
from fakes import SilentHud
from test_arm import CALIBRATION, CHAIN, CLOSED, ROBOT_CFG, Clock, FakeFollower, make, pose

from wbc.agent import __main__ as agent
from wbc.agent import tools
from wbc.agent.approve import Keyboard
from wbc.agent.budget import BudgetExceeded
from wbc.config import ROOT
from wbc.motion import arm as driver
from wbc.motion.arm import Arm, Refused, Tripped
from wbc.motion.hand import (CLOSED_GAP_MM, FLOOR_MARGIN_MM, HOVER_MM, JAW_CLEAR_MM, JUST_OVER_MM, THIN, Hand, Thing,
                             line)
from wbc.motion.joints import ARM, JointMap
from wbc.motion.kinematics import load_chain, pose_to_task
from wbc.motion.table import Surface
from wbc.motion.trajectory import lowest

REST = [0, -20, 40, 30, 0]  # the arm tests' START: the tool point 64 mm above the table
TABLE = -0.017              # config/robot.yaml's table_mm
SCREWDRIVER = Thing("screwdriver", 205, -60, -30, heading_deg=15, length_mm=176, width_mm=27)
TRAY = Thing("tray", 175, 125, -10, heading_deg=5, length_mm=210, width_mm=120, height_mm=20)
HAND = {"hand": {"across_roll_deg": 0, "moving_jaw": "left"}}


def aside(thing) -> float:
    """How far a pick puts the tool point from the thing's middle, toward the fixed jaw."""
    return thing.width_mm / 2 + JAW_CLEAR_MM - CLOSED_GAP_MM / 2


def beside(hand, step, thing):
    """Where a step's tool point is from the thing's middle: (distance, the way to the moving jaw from the tool
    point dotted with the way to the thing's middle, the offset's line against the jaws' line)."""
    dx, dy = thing.x - step.xyz[0], thing.y - step.xyz[1]
    u = hand.toward_moving_jaw(step.q)
    return math.hypot(dx, dy), (u[0] * dx + u[1] * dy) / max(math.hypot(dx, dy), 1e-9), \
        line(math.degrees(math.atan2(dy, dx)) - step.jaws_deg)


def hand_on(table=TABLE, start=REST, **kwargs):
    """A Hand on a fake arm, powered on at `start` (URDF degrees)."""
    arm, robot, said = make(FakeFollower(start=pose(start)), table=table, **kwargs)
    hand = Hand(arm, HAND)
    hand.power_on()
    robot.events.clear()
    robot.sent.clear()
    return hand, robot


def tool_at(q) -> tuple:
    x, y, z, pitch, _ = pose_to_task(CHAIN.fk(q))
    return 1000 * np.array([x, y, z]), math.degrees(pitch)


# The bench as config/robot.yaml has it: the tool point at the jaw tips, the desk's table, the 5 mm tool floor, and the
# shared calibration (the wrist flex's range_min 820), with the arm folded where it rests (Oct 29)
BENCH_CFG = yaml.safe_load((ROOT / "config" / "robot.yaml").read_text())
BENCH_CHAIN = load_chain(BENCH_CFG, ROOT)
BENCH_MAP = JointMap.from_config(BENCH_CFG, {**CALIBRATION, "wrist_flex": {**CALIBRATION["wrist_flex"],
                                                                           "range_min": 820}}, BENCH_CHAIN)
FOLDED = [4.66, -111.09, 92.3, -105.7, -0.4]
HANDLE = dict(length_mm=133, width_mm=16)  # the bench's screwdriver


def on_the_bench(side="left", start=FOLDED, sag=None):
    """A Hand on the fake arm as the bench has it, powered on at `start` (URDF degrees)."""
    robot = FakeFollower(start={**BENCH_MAP.raw(np.radians(start)), "gripper": 1500}, sag=sag)
    clock = Clock()
    robot.clock = clock
    arm = Arm(robot, BENCH_MAP, BENCH_CHAIN, gains={"elbow_flex": 32}, say=lambda s: None, clock=clock,
              sleep=clock.sleep, table=Surface.from_config(BENCH_CFG), tool_floor=BENCH_CFG["tool_floor_mm"] / 1000)
    hand = Hand(arm, {"hand": {"across_roll_deg": 0, "moving_jaw": side}})
    hand.power_on()
    robot.events.clear()
    robot.sent.clear()
    return hand, robot


def out(r, bearing=0.0):
    """(x, y) mm, `r` mm from the base's pan axis at `bearing` degrees."""
    axis = BENCH_CFG["table"]["axis_mm"]
    return axis[0] + r * math.cos(math.radians(bearing)), axis[1] + r * math.sin(math.radians(bearing))


def test_a_line_reads_the_same_either_way_round():
    assert line(30) == line(210) == line(-150) == 30
    assert line(90) == line(-90) == line(270) == 90


def test_pointing_straight_down_at_the_rest_roll_the_jaws_close_along_the_wrist_flexs_axis():
    hand, _ = hand_on()
    q = hand.solve((200, 0, 20), 90, None)
    q[4] = 0.0  # the wrist roll at across_roll_deg
    flex = CHAIN.frames(q)[1][3][1]
    assert line(hand.jaws_heading(q) - math.degrees(math.atan2(flex[1], flex[0]))) == pytest.approx(0, abs=1)
    assert line(hand.jaws_heading(q) - 90) == pytest.approx(0, abs=3)  # straight out from the base: across the arm


@pytest.mark.parametrize("xyz, jaws", [((200, -50, 0), 30), ((200, -50, 0), -60), ((150, 100, -4), 0),
                                       ((280, 0, -4), 90), ((230, 60, 40), -20)])
def test_solve_puts_the_tool_point_there_pointing_straight_down_with_the_jaws_along_the_line_asked(xyz, jaws):
    hand, _ = hand_on()
    q = hand.solve(xyz, 90, jaws)
    at, pitch = tool_at(q)
    assert at == pytest.approx(np.array(xyz, float), abs=0.5)
    assert pitch == pytest.approx(90, abs=1)
    assert line(hand.jaws_heading(q) - jaws) == pytest.approx(0, abs=1)


def test_solve_turns_the_wrist_roll_the_short_way_from_where_it_is():
    hand, _ = hand_on()
    for start in (-120, 0, 120):
        q = hand.solve((150, 100, -4), 90, 0, q0=np.radians([0, -20, 40, 30, start]))
        other = hand.solve((150, 100, -4), 90, 0, q0=np.radians([0, -20, 40, 30, -start or 150]))
        assert abs(math.degrees(q[4]) - start) <= abs(math.degrees(other[4]) - start) + 0.5


def test_solve_refuses_somewhere_out_of_reach():
    hand, _ = hand_on()
    with pytest.raises(Refused, match="can't put the tool point at x 600"):
        hand.solve((600, 0, 0), 90, 0)


def test_a_thing_is_gripped_across_its_long_axis_and_a_round_one_any_way():
    assert SCREWDRIVER.jaws_deg == pytest.approx(line(15 + 90))
    assert Thing("cup", 240, 60, 65, length_mm=82, width_mm=80, height_mm=95).jaws_deg is None


def test_a_pick_grips_something_lying_flat_halfway_up_taking_it_as_thick_as_it_is_wide():
    hand, _ = hand_on()
    over, just, down, up = hand.plan_pick(SCREWDRIVER)  # 27 mm wide, on a level table at -17 mm
    assert down.xyz[2] == pytest.approx(-17 + 27 / 2)
    assert just.xyz == (down.xyz[0], down.xyz[1], pytest.approx(-17 + 27 + JUST_OVER_MM))
    assert over.xyz == up.xyz == (down.xyz[0], down.xyz[1], pytest.approx(-17 + 27 + HOVER_MM[0]))
    assert [line(s.jaws_deg - SCREWDRIVER.jaws_deg) for s in (over, just, down, up)] == pytest.approx([0] * 4, abs=1)
    assert np.array_equal(up.q, over.q)  # back up the way it came down


def test_a_pick_puts_the_fixed_jaw_beside_the_thing_and_the_moving_jaw_beyond_it():
    hand, _ = hand_on()
    for thing in (SCREWDRIVER, Thing("marker", 180, 40, -30, heading_deg=-40, length_mm=140, width_mm=17)):
        down = hand.plan_pick(thing)[2]
        far, toward, along = beside(hand, down, thing)
        assert far == pytest.approx(aside(thing), abs=0.5)  # the fixed jaw JAW_CLEAR_MM off its side
        assert toward == pytest.approx(1, abs=0.02)  # the thing lies between the tool point and the moving jaw
        assert along == pytest.approx(0, abs=1)  # across its long axis


def test_the_moving_jaws_side_decides_which_side_of_the_thing_the_fixed_jaw_comes_down():
    left, _ = hand_on()
    arm, _, _ = make(FakeFollower(start=pose(REST)), table=TABLE)
    right = Hand(arm, {"hand": {"across_roll_deg": 0, "moving_jaw": "right"}})
    right.power_on()
    a, b = left.plan_pick(SCREWDRIVER)[2], right.plan_pick(SCREWDRIVER)[2]
    assert math.hypot(a.xyz[0] - b.xyz[0], a.xyz[1] - b.xyz[1]) == pytest.approx(2 * aside(SCREWDRIVER), abs=1)


def test_a_pick_is_refused_until_the_moving_jaws_side_is_set():
    arm, robot, _ = make(FakeFollower(start=pose(REST)), table=TABLE)
    hand = Hand(arm, {"hand": {"across_roll_deg": 0}})
    hand.power_on()
    robot.sent.clear()
    with pytest.raises(Refused, match="set hand.moving_jaw in config/robot.yaml to left or right"):
        hand.pick_up(SCREWDRIVER)
    assert robot.sent == []
    with pytest.raises(ValueError, match="hand.moving_jaw is left or right, not 'up'"):
        Hand(arm, {"hand": {"moving_jaw": "up"}})


def test_jaws_that_cant_open_fully_stop_the_pick_over_the_thing():
    hand, robot = hand_on()
    robot.between["gripper"] = (-math.inf, 1600)  # something against the outside of the moving jaw
    result = hand.pick_up(SCREWDRIVER)
    assert not result["ok"] and "something's in their way, so they didn't go down" in result["error"]
    assert tool_at(hand.arm.read().q)[0][2] == pytest.approx(result["steps"][0]["z_mm"], abs=2)  # still over it


def test_a_pick_never_aims_under_the_drivers_floor():
    hand, _ = hand_on()
    stick = Thing("stick", 205, -60, -30, heading_deg=15, length_mm=140, width_mm=20)  # its middle 10 mm up
    floor = 1000 * (TABLE + driver.TOOL_FLOOR) + FLOOR_MARGIN_MM  # 11 mm up
    assert hand.plan_pick(stick)[2].xyz[2] == pytest.approx(floor)


def test_something_standing_up_is_gripped_under_its_top_however_tall():
    hand, _ = hand_on()
    cup = Thing("cup", 205, -60, 20, length_mm=82, width_mm=80, height_mm=40)
    assert hand.plan_pick(cup)[2].xyz[2] == pytest.approx(-17 + 40 - 15)  # not refused as too thin for the floor


def test_something_too_thin_for_the_floor_to_let_the_tips_near_its_middle_is_refused_before_anything_moves():
    hand, robot = hand_on()
    pen = Thing("pen", 205, -60, -30, heading_deg=15, length_mm=140, width_mm=8)  # halfway up is 4 mm over the table
    with pytest.raises(Refused, match="the pen is about 8 mm thick, and the jaw tips go no lower than 11 mm over the "
                                      "desk there .* so they'd close near its top and slip off"):
        hand.pick_up(pen)
    assert robot.sent == []


def test_on_the_desk_robot_yaml_measured_the_jaws_reach_halfway_up_a_16_mm_handle_16_cm_out():
    surface = Surface.from_config(ROBOT_CFG)
    hand, _ = hand_on(table=surface, tool_floor=ROBOT_CFG["tool_floor_mm"] / 1000)
    handle = Thing("screwdriver", 130, -100, -30, heading_deg=-90, length_mm=133, width_mm=16)
    down = hand.plan_pick(handle)[2]
    desk = 1000 * surface.desk(0.130, -0.100)
    assert desk + 6 < down.xyz[2] < desk + 10  # about its middle, 8 mm up
    floor = 1000 * (surface.z(down.xyz[0] / 1000, down.xyz[1] / 1000) + 0.005) + FLOOR_MARGIN_MM
    assert down.xyz[2] == pytest.approx(floor)


def test_pick_up_goes_over_the_thing_opens_goes_down_grips_and_lifts_it():
    hand, robot = hand_on()
    robot.between["gripper"] = (1400, math.inf)  # the screwdriver's handle between the jaws
    result = hand.pick_up(SCREWDRIVER)
    assert result["ok"] and result["holding"] == "screwdriver" and hand.held == "screwdriver"
    assert [s["step"] for s in result["steps"]] == ["over it", "just over it", "down to it", "back up"]
    at, pitch = tool_at(hand.arm.read().q)
    assert at == pytest.approx(np.array([result["steps"][3][k] for k in ("x_mm", "y_mm", "z_mm")]), abs=2)
    assert robot.goal["gripper"] == CLOSED  # still squeezing
    assert hand.arm.jaw().holding


def test_pick_up_on_nothing_is_a_miss_and_opens_the_jaws_again():
    hand, robot = hand_on()
    result = hand.pick_up(SCREWDRIVER)
    assert not result["ok"] and result["missed"] and hand.held is None
    assert hand.arm.jaw().opening > 90
    assert tool_at(hand.arm.read().q)[0][2] == pytest.approx(result["steps"][3]["z_mm"], abs=2)  # back up


@pytest.mark.parametrize("answer, why", [(False, "as you asked"), (None, "since the session is ending")])
def test_saying_no_over_the_thing_with_the_jaws_open_stops_there(answer, why):
    hand, robot = hand_on()
    asked = []

    def confirm(question):  # the jaws are open by the time it asks, so the fixed jaw's side shows
        asked.append((question, hand.arm.jaw().opening))
        return answer
    result = hand.pick_up(SCREWDRIVER, confirm=confirm)
    assert not result["ok"] and result["stopped"] == f"over the screwdriver with the jaws open, {why}"
    (question, opening), = asked
    assert opening > 90 and question.startswith("Over the screwdriver with the jaws open: the one that didn't move "
                                                 "should be beside it, not over it. Go down and grip it?")
    assert tool_at(hand.arm.read().q)[0] == pytest.approx(np.array([result["steps"][0][k] for k in ("x_mm", "y_mm",
                                                                                                    "z_mm")]), abs=2)


def test_a_hover_out_of_reach_straight_down_drops_to_the_next_height(monkeypatch):
    hand, _ = hand_on()
    solve = hand.solve

    def lower_than_60(xyz, *args, **kwargs):  # as if the arm couldn't point straight down any higher there
        if xyz[2] > 60:
            raise Refused("out of reach")
        return solve(xyz, *args, **kwargs)
    monkeypatch.setattr(hand, "solve", lower_than_60)
    over, down, up = hand.plan_put(TRAY)  # its rim at -17 + 20: a 60 mm hover is 63
    assert over.xyz[2] == up.xyz[2] == pytest.approx(-17 + 20 + HOVER_MM[1]) and down.xyz[2] == pytest.approx(33)


def test_put_down_lowers_it_over_the_place_lets_go_and_goes_back_up():
    hand, robot = hand_on()
    robot.between["gripper"] = (1400, math.inf)
    hand.pick_up(SCREWDRIVER)
    robot.between.pop("gripper")
    result = hand.put_down(TRAY)
    assert result["ok"] and result["put"] == "screwdriver" and result["on"] == "tray" and hand.held is None
    assert result["steps"][1]["z_mm"] == pytest.approx(-17 + TRAY.height_mm + 30)  # its rim, plus 30
    assert hand.arm.jaw().opening > 90


def test_put_down_with_nothing_held_is_refused_before_anything_moves():
    hand, robot = hand_on()
    with pytest.raises(Refused, match="hold nothing"):
        hand.put_down(TRAY)
    assert robot.sent == []


def test_a_pick_out_of_reach_is_refused_with_the_arm_where_it_was():
    hand, robot = hand_on()
    with pytest.raises(Refused):
        hand.pick_up(Thing("box", 600, 0, -30, length_mm=100, width_mm=40))
    assert robot.sent == []


def test_home_goes_back_to_where_the_motors_came_on():
    hand, robot = hand_on()
    robot.between["gripper"] = (1400, math.inf)
    hand.pick_up(SCREWDRIVER)
    hand.home()
    assert np.degrees(hand.arm.read().q) == pytest.approx(REST, abs=0.5)


@pytest.mark.parametrize("side, start, x, y, heading", [
    ("left", FOLDED, 288.8, 0, 45),         # review 3: the lift used to turn the wrist roll half a turn here
    ("right", FOLDED, *out(250, 60), -60),  # the same, at a bearing nothing was touched at
    ("left", FOLDED, 180.7, 29.1, 42),      # the moving jaw used to change sides between the plan's rounds
    ("left", FOLDED[:4] + [0], 216, -31, 80),
    ("left", REST, *out(180), 90), ("left", REST, *out(190), 90), ("left", REST, *out(250), 90),
    ("left", REST, *out(250), 75), ("right", FOLDED, 150, 0, 91.3), ("left", FOLDED, 130, 0, 91.2),
])
def test_a_pick_keeps_the_wrist_roll_one_way_round_from_over_the_thing_to_back_up(side, start, x, y, heading):
    hand, _ = on_the_bench(side, start)
    hand.arm.table = dataclasses.replace(hand.arm.table, bearing=(-math.pi, math.pi))  # as if touched all round
    thing = Thing("screwdriver", x, y, -30, heading_deg=heading, **HANDLE)
    steps = hand.plan_pick(thing)
    rolls = [math.degrees(s.q[4]) for s in steps]
    assert max(rolls) - min(rolls) < 2
    far, toward, along = beside(hand, steps[2], thing)
    assert far == pytest.approx(aside(thing), abs=0.5) and toward > 0.99 and along == pytest.approx(0, abs=1)


def test_picks_all_over_the_touched_desk_plan_with_the_fixed_jaw_beside_the_thing_and_one_roll():
    rng = np.random.default_rng(3)
    hands = {side: on_the_bench(side)[0] for side in ("left", "right")}
    for i in range(24):
        hand = hands[("left", "right")[i % 2]]
        thing = Thing("screwdriver", *out(rng.uniform(130, 260), rng.uniform(-90, 25)), -30,
                      heading_deg=rng.uniform(-90, 90), **HANDLE)
        steps = hand.plan_pick(thing)
        rolls = [math.degrees(s.q[4]) for s in steps]
        assert max(rolls) - min(rolls) < 2, (thing, rolls)
        far, toward, _ = beside(hand, steps[2], thing)
        assert far == pytest.approx(aside(thing), abs=0.5) and toward > 0.99, thing


def test_the_way_down_ends_with_a_short_move_from_just_over_the_things_top():
    hand, robot = on_the_bench()
    thing = Thing("screwdriver", *out(160), -30, heading_deg=90, **HANDLE)
    over, just, down, up = hand.plan_pick(thing)
    top = 1000 * hand.arm.table.desk(thing.x / 1000, thing.y / 1000) + 16
    assert just.xyz[2] == pytest.approx(top + JUST_OVER_MM) and down.xyz[2] < top < just.xyz[2] < over.xyz[2]
    robot.between["gripper"] = (1400, math.inf)
    assert hand.pick_up(thing)["ok"]


@pytest.mark.parametrize("ticks, goes_down", [(6, True), (19, False), (23, False)])
def test_a_pick_reads_how_far_the_arm_sags_just_over_the_thing_and_stops_there_before_it_would_trip(ticks, goes_down):
    thing = Thing("screwdriver", *out(220), -30, heading_deg=90, **HANDLE)
    hand, robot = on_the_bench(sag={"shoulder_lift": ticks})  # it settles `ticks` past every goal, tipping forward
    robot.between["gripper"] = (1400, math.inf)
    result = hand.pick_up(thing)  # never Tripped: 23 ticks used to trip the sag cut going down
    assert result["ok"] is goes_down
    if not goes_down:
        assert "the arm sagged enough that down at it, it would stop about" in result["error"]
        assert [s["step"] for s in result["steps"]] == ["over it", "just over it"]
        assert tool_at(hand.arm.read().q)[0][2] > result["steps"][1]["z_mm"] - 10  # still just over it
    assert hand.arm.stopped_short is not None


def test_the_sag_check_counts_only_the_joints_whose_shortfall_lowers_the_tips():
    # going down, the re-aim takes up at once a joint that stopped short holding the tips up, while one that sagged
    # can stay put in its friction, so the two don't cancel (review, Oct 30)
    hand, _ = on_the_bench()
    thing = Thing("screwdriver", *out(220), -30, heading_deg=90, **HANDLE)
    down = hand.plan_pick(thing)[2]

    def tips(q):
        return 1000 * lowest(BENCH_CHAIN, q, driver.FLOOR_JOINTS, hand.arm.table)[0]

    def per_radian(i):
        q = np.array(down.q, float)
        q[i] += 1e-4
        return (tips(q) - tips(down.q)) / 1e-4
    lift, elbow = ARM.index("shoulder_lift"), ARM.index("elbow_flex")
    short = np.zeros(len(ARM))
    short[lift] = 5.5 / per_radian(lift)     # sagged: the tips 5.5 mm lower, 0.5 mm over the cut
    short[elbow] = -2.0 / per_radian(elbow)  # and held them 2 mm up
    assert tips(down.q - short) > 2  # netted, they'd say go down
    hand.arm.stopped_short = short
    assert "the arm sagged enough that down at it, it would stop about" in hand._sagging(down, thing)


def test_a_grip_that_ends_a_tick_under_its_floor_still_goes_back_up_holding_the_thing():
    # each joint settles 2 ticks past its goals, inside AIM, so the grip ends a hair under the tool point's floor, and
    # the climb's first whole-tick goals dip 14-26 µm before they rise (review, Oct 30)
    hand, robot = on_the_bench(sag={"shoulder_lift": 2, "elbow_flex": 2})
    robot.between["gripper"] = (1400, math.inf)
    result = hand.pick_up(Thing("screwdriver", *out(220, 10), -30, heading_deg=0, **HANDLE))
    assert result["ok"] and hand.held == "screwdriver"


def test_a_grip_that_cant_go_back_up_says_the_jaws_hold_the_thing_so_the_next_pick_doesnt_drop_it(monkeypatch):
    monkeypatch.setattr(driver, "ROUNDING", 1e-6)  # so the climb above is refused
    monkeypatch.setattr(driver, "MAX_JOINT_SPEED", 30.0)  # the first caps: whether the climb's first tick dips
    monkeypatch.setattr(driver, "MAX_TIP_SPEED", 0.08)    # under that 1e-6 depends on the move's speed
    hand, robot = on_the_bench(sag={"shoulder_lift": 2, "elbow_flex": 2})
    robot.between["gripper"] = (1400, math.inf)
    result = hand.pick_up(Thing("screwdriver", *out(220, 10), -30, heading_deg=0, **HANDLE))
    assert not result["ok"] and result["holding"] == hand.held == "screwdriver" and len(result["steps"]) == 3
    assert result["error"].startswith("the jaws hold the screwdriver down at the desk: the arm couldn't go back up")
    with pytest.raises(Refused, match="already hold the screwdriver"):
        hand.pick_up(Thing("screwdriver", *out(200), -30, heading_deg=0, **HANDLE))


def test_a_miss_that_cant_go_back_up_says_so_not_that_nothing_moved(monkeypatch):
    monkeypatch.setattr(driver, "ROUNDING", 1e-6)  # so the climb from the grip is refused, as above
    monkeypatch.setattr(driver, "MAX_JOINT_SPEED", 30.0)  # and at the first caps, as above
    monkeypatch.setattr(driver, "MAX_TIP_SPEED", 0.08)
    hand, robot = on_the_bench(sag={"shoulder_lift": 2, "elbow_flex": 2})  # nothing between the jaws
    result = hand.pick_up(Thing("screwdriver", *out(220, 10), -30, heading_deg=0, **HANDLE))
    assert not result["ok"] and result["missed"] and hand.held is None and len(result["steps"]) == 3
    assert result["error"].startswith("the jaws closed on nothing where the screwdriver was, and the arm couldn't go "
                                      "back up")
    assert hand.arm.jaw().opening > 90


@pytest.mark.parametrize("side, r, bearing, width", [("right", 250, 28.4, 16), ("right", 200, 28.8, 16),
                                                     ("right", 250, 26.9, 12.5), ("left", 260, -93, 16),
                                                     ("left", 260, -92, 12.5)])
def test_a_pick_whose_tips_would_come_down_where_the_table_is_held_higher_beside_it_is_refused(side, r, bearing, width):
    # lying out from the base at a touched bearing's edge: the fixed jaw's side is past it, where the floors hold the
    # table higher, and the other way round is out of reach (review, Oct 30)
    hand, robot = on_the_bench(side)
    thing = Thing("screwdriver", *out(r, bearing), -30, heading_deg=bearing, length_mm=133, width_mm=width)
    with pytest.raises(Refused, match="the jaw tips go no lower than \\d+ mm over the desk there .*, held higher "
                                      "beside it, where no touch measured the desk\\), so they'd close near its top"):
        hand.pick_up(thing)
    assert robot.sent == []


def test_near_a_touched_bearings_edge_a_pick_takes_the_way_round_that_grips_low_enough():
    hand, _ = on_the_bench("right")
    thing = Thing("screwdriver", *out(250, 28.4), -30, heading_deg=58.4, **HANDLE)
    down = hand.plan_pick(thing)[2]  # the nearer roll would grip it 82% up
    table = hand.arm.table
    under = 1000 * table.line(math.hypot(thing.x / 1000 - table.axis[0], thing.y / 1000 - table.axis[1]))
    assert down.xyz[2] - under < THIN * 16 and math.degrees(down.q[4]) == pytest.approx(-149, abs=2)


@pytest.mark.parametrize("side, r, bearing, heading", [("left", 270, -90, -60), ("left", 280, 0, 30),
                                                       ("left", 270, 20, 45), ("left", 285, 0, 45),
                                                       ("left", 275, -60, 135), ("left", 275, -90, 90)])
def test_a_pick_far_out_is_solved_where_ik_from_where_the_arm_is_stalls(side, r, bearing, heading):
    # From the folded rest, IK stalls here with the wrist flex on its URDF limit, so each plans only with the retry.
    # The last three also need a try that never quite converged but passes the checks every answer has to
    hand, _ = on_the_bench(side)
    steps = hand.plan_pick(Thing("screwdriver", *out(r, bearing), -30, heading_deg=heading, **HANDLE))
    rolls = [math.degrees(s.q[4]) for s in steps]
    assert max(rolls) - min(rolls) < 2


def test_a_hover_ik_only_got_near_is_taken_only_when_no_hover_has_one_it_converged_on():
    hand, _ = on_the_bench()
    over = hand.plan_pick(Thing("screwdriver", *out(255, -65), -30, heading_deg=-45, **HANDLE))[0]
    x, y, z, pitch, _ = pose_to_task(BENCH_CHAIN.fk(over.q))  # not the 60 mm hover 1.8 mm off, but 45 mm, exactly
    assert np.linalg.norm(1000 * np.array([x, y, z]) - np.array(over.xyz)) < 0.2 and math.degrees(pitch) > 89.5


@pytest.mark.parametrize("side, bearing", [("left", 0), ("left", -15), ("right", 0)])
def test_a_put_after_a_pick_turns_the_wrist_roll_the_short_way(side, bearing):
    hand, robot = on_the_bench(side)
    robot.between["gripper"] = (1400, math.inf)
    assert hand.pick_up(Thing("screwdriver", *out(250), -30, heading_deg=45, **HANDLE))["ok"]
    now = math.degrees(hand.arm.read().q[4])
    bowl = Thing("bowl", *out(250, bearing), -30, length_mm=100, width_mm=100, height_mm=40)
    over, down, up = hand.plan_put(bowl)  # half a turn round, the jaws reach it from a higher hover
    assert abs(math.degrees(over.q[4]) - now) < 30


def test_the_driver_keeps_how_far_each_joint_stopped_short_before_it_re_aimed():
    hand, robot = on_the_bench(sag={"elbow_flex": 30})
    hand.arm.move_to(hand.arm.read().q + np.radians([0, 2, -6, 0, 0]))
    assert np.degrees(hand.arm.stopped_short) == pytest.approx([0, 0, -30 / (4096 / 360), 0, 0], abs=0.1)


def test_a_pick_where_no_touch_measured_the_desk_is_refused_before_anything_moves():
    hand, robot = on_the_bench()
    with pytest.raises(Refused, match="no touch measured the desk near the screwdriver, so the floors take it up to "
                                      "\\d+ mm higher than it may be there"):
        hand.pick_up(Thing("screwdriver", *out(200, 45), -30, heading_deg=0, **HANDLE))
    assert robot.sent == []


def test_the_drivers_table_lets_a_move_go_lower_than_the_bases_underside_but_keeps_its_floor():
    low = np.radians([0, -40, 70, 40, 0])  # the arm tests' LOW: the tool point 6 mm over the base's underside
    for table, allowed in ((0.0, False), (TABLE, True)):
        arm, robot, _ = make(FakeFollower(start=pose(REST)), table=table)
        arm.power_on()
        if allowed:
            arm.move_to(low)
            assert lowest(CHAIN, arm.read().q, driver.FLOOR_JOINTS, table)[0] > driver.TOOL_FLOOR
        else:
            with pytest.raises(Refused, match="under its 10 mm floor"):
                arm.move_to(low)


def test_a_table_far_from_the_desks_readings_is_taken_for_a_mistake_in_units():
    with pytest.raises(ValueError, match="its height at the base, -17000 mm, is outside -60 to 10 mm"):
        make(table=-17)


def test_a_tool_floor_outside_5_to_30_mm_is_refused():
    with pytest.raises(ValueError, match="the tool point's floor has to be from 5 to 30 mm, not 4"):
        make(tool_floor=0.004)  # under SAG, the sag cut would come only under the table


def test_robot_yaml_sets_the_table_the_tool_floor_and_the_jaws_line():
    assert Surface.from_config(ROBOT_CFG).config() == {"at_base_mm": -8.7, "slope_mm_per_m": -88.9,
                                                       "reach_mm": [94, 399], "margin_mm": 2.0,
                                                       "axis_mm": [38.8353, 0.0], "bearing_deg": [-92, 27]}
    assert ROBOT_CFG["tool_floor_mm"] == 5 and ROBOT_CFG["hand"]["across_roll_deg"] == 0
    assert ROBOT_CFG["hand"]["moving_jaw"] == "left"  # looked at on the arm, Oct 30
    hand = Hand(make(FakeFollower(start=pose(REST)))[0], ROBOT_CFG)
    assert hand.moving_axis is not None  # so picks aren't refused for it


def test_robot_yaml_s_tool_point_turned_half_a_turn_with_the_wrist_roll_s_zero():
    # Oct 30: the roll's zero went back half a turn, so every pose reads 180° less on it, and the four-touch test's
    # tips (Oct 29, [-0.0099, 0.0033, 0] then) have to come out where they did, or the desk's touches move
    before = load_chain({**BENCH_CFG, "tool": {**BENCH_CFG["tool"], "xyz": [-0.0099, 0.0033, 0]}}, ROOT)
    for q in np.random.default_rng(0).uniform(BENCH_CHAIN.lower, BENCH_CHAIN.upper, (200, 5)):
        now = BENCH_CHAIN.fk(q - [0, 0, 0, 0, math.pi])[:3, 3]
        assert 1000 * now == pytest.approx(1000 * before.fk(q)[:3, 3], abs=0.01)


def test_arm_open_takes_the_table_and_the_tool_floor_from_robot_yaml():
    robot = FakeFollower(start=pose(REST))
    arm = Arm.open("/dev/cu.fake", ROBOT_CFG, ROOT, make_robot=lambda *args: robot)
    assert arm.table == Surface.from_config(ROBOT_CFG) and arm.tool_floor == pytest.approx(0.005)


# The agent's arm tools


def bench(sim=False, answers=(), **kwargs):
    asked = []

    def confirm(question):
        asked.append(question)
        return answers[len(asked) - 1] if len(asked) <= len(answers) else True
    b = tools.Bench(ROOT, {}, None, None, sim=sim, robot_cfg={"hand": {"check_over": False}}, confirm=confirm,
                    **kwargs)
    return b, tools.build_toolbox(b), asked


def test_names_drop_their_article():
    assert tools.key("the Screwdriver") == tools.key("screwdriver") == "screwdriver"
    assert tools.key("that cup") == "cup" and tools.key("a") == "a"


def test_with_sim_locate_pick_up_and_put_down_pretend_from_the_recorded_desk():
    b, tb, _ = bench(sim=True)
    found = tb.dispatch("locate", {"thing": "the screwdriver"})
    assert found["ok"] and found["found"][0]["x_mm"] == 205.0 and b.seen["screwdriver"][0].x == 205.0
    assert tb.dispatch("put_down", {"thing": "tray"})["ok"] is False  # nothing held yet
    assert tb.dispatch("pick_up", {"thing": "screwdriver"}) == {"ok": True, "sim": True, "holding": "screwdriver",
                                                                 "x_mm": 205.0, "y_mm": -60.0}
    assert "already hold the screwdriver" in tb.dispatch("pick_up", {"thing": "screwdriver"})["error"]
    assert tb.dispatch("put_down", {"thing": "tray"})["put"] == "screwdriver"
    assert tb.dispatch("locate", {"thing": "hammer"})["found"] == []
    assert "no hammer" in tb.dispatch("pick_up", {"thing": "hammer"})["error"]


def test_without_arm_the_arms_tools_say_how_to_turn_it_on():
    b, tb, asked = bench()
    b.seen["screwdriver"] = (SCREWDRIVER, time.monotonic())
    assert "--arm" in tb.dispatch("pick_up", {"thing": "screwdriver"})["error"] and asked == []


def test_the_arm_moves_only_after_enter_and_powers_on_first():
    arm, robot, _ = make(FakeFollower(start=pose(REST)), table=TABLE)
    b, tb, asked = bench(answers=(False,), arm_port="fake")
    b.hand = Hand(arm, HAND)
    b.seen["screwdriver"] = (SCREWDRIVER, time.monotonic())
    result = tb.dispatch("pick_up", {"thing": "screwdriver"})
    assert not result["ok"] and "didn't press Enter" in result["error"] and not arm.powered
    assert "Pick up the screwdriver at x 205, y -60 mm?" in asked[0] and "motors come on first" in asked[0]
    robot.between["gripper"] = (1400, math.inf)
    result = tb.dispatch("pick_up", {"thing": "screwdriver"})
    assert result["ok"] and result["holding"] == "screwdriver" and len(arm.powered) == 6
    assert "motors come on first" in asked[1]  # the no left them off


def test_a_refused_move_comes_back_as_an_error_not_an_exception():
    arm, robot, _ = make(FakeFollower(start=pose(REST)), table=TABLE)
    b, tb, _ = bench(arm_port="fake")
    b.hand = Hand(arm, HAND)
    b.seen["box"] = (Thing("box", 600, 0, -30, length_mm=100, width_mm=40), time.monotonic())
    result = tb.dispatch("pick_up", {"thing": "box"})
    assert not result["ok"] and result["error"].startswith("refused, nothing moved for it: the arm can't put")


def test_park_takes_the_arm_home_and_turns_the_motors_off():
    arm, robot, _ = make(FakeFollower(start=pose(REST)), table=TABLE)
    said = []
    b, tb, _ = bench(arm_port="fake", say=said.append)
    b.hand = Hand(arm, HAND)
    b.seen["screwdriver"] = (SCREWDRIVER, time.monotonic())
    robot.between["gripper"] = (1400, math.inf)
    assert tb.dispatch("pick_up", {"thing": "screwdriver"})["ok"]
    b.park()
    assert np.degrees(arm.joints.q(robot.pos)) == pytest.approx(REST, abs=0.5)
    assert not any(robot.torque.values())
    assert said == ["The arm is going back to rest, then its motors go off."]  # it moves after the last reply


def test_a_move_whose_question_the_end_of_the_input_answered_says_the_session_is_ending():
    arm, robot, _ = make(FakeFollower(start=pose(REST)), table=TABLE)
    b, tb, asked = bench(answers=(None,), arm_port="fake")
    b.hand = Hand(arm, HAND)
    b.seen["screwdriver"] = (SCREWDRIVER, time.monotonic())
    result = tb.dispatch("pick_up", {"thing": "screwdriver"})
    assert result == {"ok": False, "error": "the session is ending, so the arm didn't move"} and not arm.powered


def test_after_ctrl_c_no_arm_move_starts_not_even_the_first():
    b, tb, asked = bench(arm_port="fake")
    b.seen["screwdriver"] = (SCREWDRIVER, time.monotonic())
    opened = []
    b.stop_arm()  # the hand isn't open yet
    result = tb.dispatch("pick_up", {"thing": "screwdriver"})
    assert not result["ok"] and "session is ending" in result["error"] and b.hand is None and asked == opened == []


def test_a_second_arm_tool_while_one_runs_is_refused_not_queued():
    b, tb, asked = bench(arm_port="fake")
    b.seen["screwdriver"] = (SCREWDRIVER, time.monotonic())
    with b.arm_lock:  # another tool's move holds the arm
        result = tb.dispatch("pick_up", {"thing": "screwdriver"})
    assert not result["ok"] and "call arm tools one at a time" in result["error"] and asked == []


def test_a_thing_is_located_again_once_stale_moved_or_wanted_at_another_height():
    b, tb, _ = bench(sim=True)
    elsewhere = Thing("screwdriver", 1, 2, -30, length_mm=176, width_mm=27)  # where it used to be
    b.seen["screwdriver"] = (elsewhere, time.monotonic())
    assert tb.dispatch("pick_up", {"thing": "screwdriver"})["x_mm"] == 1  # fresh: no locate
    assert "screwdriver" not in b.seen  # in the jaws now: located again before the next move
    tb.dispatch("put_down", {"thing": "tray"})
    b.seen["screwdriver"] = (elsewhere, time.monotonic() - tools.SEEN_S - 1)
    assert tb.dispatch("pick_up", {"thing": "screwdriver"})["x_mm"] == 205.0  # stale: located again
    tb.dispatch("put_down", {"thing": "tray"})
    b.seen["cup"] = (Thing("cup", 1, 2, -30, length_mm=82, width_mm=80), time.monotonic())  # located lying flat
    assert tb.dispatch("pick_up", {"thing": "cup", "height_mm": 95})["x_mm"] == 240.0  # standing: located again


def test_park_after_ctrl_c_turns_the_motors_off_without_moving():
    arm, robot, _ = make(FakeFollower(start=pose(REST)), table=TABLE)
    b, tb, _ = bench(arm_port="fake")
    b.hand = Hand(arm, HAND)
    b.seen["screwdriver"] = (SCREWDRIVER, time.monotonic())
    robot.between["gripper"] = (1400, math.inf)
    assert tb.dispatch("pick_up", {"thing": "screwdriver"})["ok"]
    where = dict(robot.pos)
    b.park(home=False)
    assert robot.pos == where and not any(robot.torque.values())


def summary(thing) -> dict:
    """What locate returns for a Thing."""
    return {"phrase": thing.name, "x_mm": thing.x, "y_mm": thing.y, "top_z_mm": thing.top_z,
            "heading_deg": thing.heading_deg, "length_mm": thing.length_mm, "width_mm": thing.width_mm,
            "height_mm": thing.height_mm}


def test_with_sim_move_thing_pretends_from_the_recorded_desk():
    b, tb, _ = bench(sim=True)
    result = tb.dispatch("move_thing", {"thing": "the screwdriver", "place": "tray", "place_height_mm": 20})
    assert result == {"ok": True, "sim": True, "moved": "screwdriver", "onto": "tray", "x_mm": 175.0, "y_mm": 125.0}
    assert "screwdriver" not in b.seen and "tray" in b.seen  # moved, so it's located again before the next move


def test_move_thing_finds_the_thing_and_the_place_in_one_photo(monkeypatch):
    calls = []

    def one_photo(bench, wants):
        calls.append(list(wants))
        scene = {"screwdriver": summary(SCREWDRIVER), "sticky note": summary(TRAY)}
        return {t: {"found": [scene[tools.key(t)]] if tools.key(t) in scene else []} for t, _ in wants}
    monkeypatch.setattr(tools, "_locate_all", one_photo)
    b, tb, asked = bench()  # no --arm: it finds them, then says the arm's tools are off
    result = tb.dispatch("move_thing", {"thing": "screwdriver", "place": "sticky note", "place_height_mm": 20})
    assert calls == [[("screwdriver", 0), ("sticky note", 20)]] and "--arm" in result["error"] and asked == []
    assert tb.dispatch("move_thing", {"thing": "screwdriver", "place": "cup"}) == {
        "ok": False, "error": "locate found no cup on the desk"}
    assert calls[-1] == [("cup", 0)]  # the screwdriver was found 0 s ago


def test_move_thing_picks_the_thing_up_and_puts_it_on_the_place_after_one_enter():
    arm, robot, _ = make(FakeFollower(start=pose(REST)), table=TABLE)
    b, tb, asked = bench(arm_port="fake")
    b.hand = Hand(arm, HAND)
    b.seen.update(screwdriver=(SCREWDRIVER, time.monotonic()), tray=(TRAY, time.monotonic()))
    robot.between["gripper"] = (1400, math.inf)  # the handle between the jaws
    result = tb.dispatch("move_thing", {"thing": "screwdriver", "place": "tray"})
    assert result["ok"] and result["moved"] == "screwdriver" and result["onto"] == "tray" and result["holding"] is None
    assert [s["step"] for s in result["steps"]] == ["over it", "just over it", "down to it", "back up", "over it",
                                                     "down to it", "back up"]
    assert b.hand.held is None and asked == ["Pick up the screwdriver at x 205, y -60 mm and put it on the tray at "
                                             "x 175, y 125 mm? The motors come on first."]


def test_a_move_whose_place_is_out_of_reach_is_refused_before_anything_is_asked():
    arm, robot, _ = make(FakeFollower(start=pose(REST)), table=TABLE)
    b, tb, asked = bench(arm_port="fake")
    b.hand = Hand(arm, HAND)
    box = Thing("box", 600, 0, -30, length_mm=100, width_mm=40)
    b.seen.update(screwdriver=(SCREWDRIVER, time.monotonic()), box=(box, time.monotonic()))
    result = tb.dispatch("move_thing", {"thing": "screwdriver", "place": "box"})
    assert result["error"].startswith("refused, nothing moved for it: the arm can't put") and asked == []
    assert not arm.powered and robot.sent == []


def test_a_put_refused_after_the_pick_says_the_jaws_still_hold_the_thing(monkeypatch):
    arm, robot, _ = make(FakeFollower(start=pose(REST)), table=TABLE)
    b, tb, asked = bench(arm_port="fake")
    b.hand = Hand(arm, HAND)
    b.seen.update(screwdriver=(SCREWDRIVER, time.monotonic()), tray=(TRAY, time.monotonic()))
    robot.between["gripper"] = (1400, math.inf)
    plan_put = Hand.plan_put

    def after_the_pick(self, place):  # it plans before anything moves, and is refused once the jaws hold it
        if self.held:
            raise Refused("the arm can't put the tool point there")
        return plan_put(self, place)
    monkeypatch.setattr(Hand, "plan_put", after_the_pick)
    result = tb.dispatch("move_thing", {"thing": "screwdriver", "place": "tray"})
    assert not result["ok"] and result["holding"] == b.hand.held == "screwdriver" and len(result["steps"]) == 4
    assert result["error"].startswith("picked up the screwdriver, but putting it on the tray was refused, so the jaws "
                                      "still hold it")


def test_a_move_whose_pick_misses_doesnt_try_the_put():
    arm, robot, _ = make(FakeFollower(start=pose(REST)), table=TABLE)
    b, tb, asked = bench(arm_port="fake")
    b.hand = Hand(arm, HAND)
    b.seen.update(screwdriver=(SCREWDRIVER, time.monotonic()), tray=(TRAY, time.monotonic()))
    result = tb.dispatch("move_thing", {"thing": "screwdriver", "place": "tray"})  # nothing between the jaws
    assert result["missed"] and result["put"] == "not tried" and b.hand.held is None
    assert [s["step"] for s in result["steps"]] == ["over it", "just over it", "down to it", "back up"]


def test_saying_no_over_the_place_says_the_jaws_still_hold_the_thing_not_that_it_moved():
    arm, robot, _ = make(FakeFollower(start=pose(REST)), table=TABLE)
    b, tb, asked = bench(arm_port="fake", answers=(True, True, False))
    b.robot_cfg = {"hand": {"check_over": True}}
    b.hand = Hand(arm, HAND)
    b.seen.update(screwdriver=(SCREWDRIVER, time.monotonic()), tray=(TRAY, time.monotonic()))
    robot.between["gripper"] = (1400, math.inf)
    result = tb.dispatch("move_thing", {"thing": "screwdriver", "place": "tray"})
    assert {k: v for k, v in result.items() if k != "steps"} == {"ok": False, "stopped": "over the tray, as you asked",
                                                                  "holding": "screwdriver"}
    assert len(result["steps"]) == 5 and asked[2] == "Over the tray. Go down and let go?"
    assert b.hand.held == "screwdriver"


def test_with_go_ahead_request_the_request_moves_the_arm_without_a_question():
    arm, robot, _ = make(FakeFollower(start=pose(REST)), table=TABLE)
    b, tb, asked = bench(arm_port="fake", go_ahead="request")
    b.robot_cfg, b.confirm = {"hand": {"check_over": True}}, None  # no keyboard, and no check-over: neither is asked
    b.hand = Hand(arm, HAND)
    b.seen.update(screwdriver=(SCREWDRIVER, time.monotonic()), tray=(TRAY, time.monotonic()))
    robot.between["gripper"] = (1400, math.inf)
    result = tb.dispatch("move_thing", {"thing": "screwdriver", "place": "tray"})
    assert result["ok"] and result["moved"] == "screwdriver" and b.hand.held is None and len(arm.powered) == 6


@pytest.mark.parametrize("go_ahead, says", [("enter", "The user presses Enter before the arm moves."),
                                            ("request", "The arm moves at once.")])
def test_the_arm_tools_tell_the_router_what_lets_the_arm_go(go_ahead, says):
    b, tb, _ = bench(go_ahead=go_ahead)
    told = {s["name"]: s["description"] for s in tb.schemas()}
    assert all(says in told[name] for name in ("pick_up", "put_down", "move_thing", "arm_home"))


@pytest.mark.parametrize("sim", [False, True])
def test_after_a_stop_no_arm_tool_moves_or_even_opens_the_arm_until_the_next_request(sim):
    b, tb, asked = bench(sim=sim, arm_port="fake", go_ahead="request")
    b.seen.update(screwdriver=(SCREWDRIVER, time.monotonic()), tray=(TRAY, time.monotonic()))
    b.halt()  # "Copilot, stop" before the arm was ever opened
    for name, args in (("pick_up", {"thing": "screwdriver"}), ("put_down", {"thing": "tray"}),
                       ("move_thing", {"thing": "screwdriver", "place": "tray"}), ("arm_home", {})):
        assert tb.dispatch(name, args) == tools.SAID_STOP
    assert b.hand is None and asked == []
    b.resume()
    if sim:
        assert tb.dispatch("pick_up", {"thing": "screwdriver"})["ok"]


def stop_before(arm, b, move: int) -> list:
    """Make the arm hear "Copilot, stop" just before its `move`th move_to, as from the ears' thread."""
    moves, move_to = [], arm.move_to

    def moving(q, **kwargs):
        moves.append(q)
        if len(moves) == move:
            b.halt()
        return move_to(q, **kwargs)
    arm.move_to = moving
    return moves


@pytest.mark.parametrize("move, holding", [(2, None), (4, "screwdriver")])  # just over it; back up with it gripped
def test_a_stop_partway_through_a_pick_holds_the_arm_and_says_what_the_jaws_have(move, holding):
    arm, robot, _ = make(FakeFollower(start=pose(REST)), table=TABLE)
    b, tb, _ = bench(arm_port="fake", go_ahead="request")
    b.hand = Hand(arm, HAND)
    b.seen["screwdriver"] = (SCREWDRIVER, time.monotonic())
    robot.between["gripper"] = (1400, math.inf)
    moves = stop_before(arm, b, move)
    result = tb.dispatch("pick_up", {"thing": "screwdriver"})
    assert result["ok"] is False and result["halted"] and "stopped before a move" in result["error"]
    assert result.get("holding") == holding and b.hand.held == holding and len(moves) == move
    assert all(robot.torque.values()) and "screwdriver" not in b.seen  # held there, and located again before a move


@pytest.mark.parametrize("move, holding", [(2, "screwdriver"), (3, None)])  # going down to the tray; back up from it
def test_a_stop_partway_through_a_put_says_whether_the_jaws_let_go(move, holding):
    arm, robot, _ = make(FakeFollower(start=pose(REST)), table=TABLE)
    b, tb, _ = bench(arm_port="fake", go_ahead="request")
    b.hand = Hand(arm, HAND)
    b.seen.update(screwdriver=(SCREWDRIVER, time.monotonic()), tray=(TRAY, time.monotonic()))
    robot.between["gripper"] = (1400, math.inf)
    assert tb.dispatch("pick_up", {"thing": "screwdriver"})["ok"]
    stop_before(arm, b, move)
    result = tb.dispatch("put_down", {"thing": "tray"})
    assert result["halted"] and result.get("holding") == holding and b.hand.held == holding


def test_park_after_a_stop_leaves_the_arm_where_it_stopped():
    arm, robot, _ = make(FakeFollower(start=pose(REST)), table=TABLE)
    said = []
    b, tb, _ = bench(arm_port="fake", go_ahead="request", say=said.append)
    b.hand = Hand(arm, HAND)
    b.seen["screwdriver"] = (SCREWDRIVER, time.monotonic())
    robot.between["gripper"] = (1400, math.inf)
    assert tb.dispatch("pick_up", {"thing": "screwdriver"})["ok"]
    b.halt()
    where = dict(robot.pos)
    b.park()
    assert robot.pos == where and not any(robot.torque.values())
    assert said == ["The arm was told to stop, so it stays where it is. Hold it: its motors are going off."]


def test_an_arm_resting_past_the_end_of_a_range_is_refused_before_the_question():
    # Oct 30: the gripper had been turned past the wrist roll's limit, and only Enter and the power-on found it
    arm, robot, _ = make(FakeFollower(start=pose([0, -20, 40, 30, -160])), table=TABLE)
    b, tb, asked = bench(arm_port="fake")
    b.hand = Hand(arm, HAND)
    b.seen["screwdriver"] = (SCREWDRIVER, time.monotonic())
    result = tb.dispatch("pick_up", {"thing": "screwdriver"})
    assert "past the end of its range: the wrist roll by 2.7°" in result["error"] and asked == []  # in whole ticks
    assert "the arm's own pose, not where anything on the desk is" in result["error"]
    assert not arm.powered and robot.sent == []


# Enter at the keyboard


def test_closing_the_keyboard_answers_the_waiting_question_no_and_every_one_after():
    said, asked = [], threading.Event()
    keyboard = Keyboard(say=lambda s: (said.append(s), asked.set()), wait_s=30)
    threading.Thread(target=lambda: (asked.wait(5), keyboard.close())).start()
    t = time.monotonic()
    assert keyboard.ask("Go?") is None and time.monotonic() - t < 2  # woken by close(), not the 30 s timeout
    t = time.monotonic()
    assert keyboard.ask("And now?") is None and time.monotonic() - t < 0.5 and keyboard.offer("\n") is False
    assert said == ["Go? Enter goes ahead; anything else stops."]  # no "No key", and the second never asked
    assert keyboard.closed


def test_a_line_typed_before_the_keyboard_closes_stays_the_answer():
    keyboard = Keyboard(say=lambda s: None, wait_s=5)

    def typed():
        while not keyboard.offer("no\n"):  # a no, then the end of the input at once
            threading.Event().wait(0.005)
        keyboard.close()
    threading.Thread(target=typed).start()
    assert keyboard.ask("Back to rest?", forever=True) is False


def answer_after(keyboard, line, delay=0.05):
    def later():
        while not keyboard.offer(line):
            threading.Event().wait(0.005)
    t = threading.Timer(delay, later)
    t.start()
    return t


def test_only_a_bare_enter_says_yes():
    said = []
    keyboard = Keyboard(say=said.append, wait_s=5)
    answer_after(keyboard, "\n")
    assert keyboard.ask("Pick up the screwdriver?") is True
    assert said == ["Pick up the screwdriver? Enter goes ahead; anything else stops."]
    answer_after(keyboard, "no\n")
    assert keyboard.ask("Again?") is False


def test_a_line_typed_with_no_question_waiting_is_a_request():
    assert Keyboard().offer("put the screwdriver in the tray\n") is False


def test_no_key_in_time_is_no():
    said = []
    assert Keyboard(say=said.append, wait_s=0.05).ask("Go?") is False and said[-1].startswith("No key")


def test_a_question_asked_forever_waits_past_the_timeout():
    keyboard = Keyboard(say=lambda s: None, wait_s=0.05)
    answer_after(keyboard, "\n", delay=0.3)
    assert keyboard.ask("Back to rest?", forever=True) is True


# Ending the session


class Typing:
    """stdin that types `lines`, then stays open, as a terminal does, until `done` is set."""

    def __init__(self, *lines):
        self.lines, self.done = lines, threading.Event()

    def __iter__(self):
        yield from self.lines
        self.done.wait(5)


def run_main(monkeypatch, run_turn, keyboard, stdin, can_go_home=True):
    parked = []
    bench = NS(park=lambda home=True: parked.append(home), stop_arm=lambda: None, hand=None,
               stopped=threading.Event(), can_go_home=lambda: can_go_home, resume=lambda: None)
    budget = NS(spent=0.0, limit=5.0, check=lambda: None, pop_warning=lambda: None)
    router = NS(label="Spark", model="spark", run_turn=run_turn)
    monkeypatch.setattr(agent, "build", lambda *a: ({"ears": {}, "coder": {}, "mouth": {}}, budget, SilentHud(), router,
                                                    bench, keyboard))
    monkeypatch.setattr(agent.coder, "compile_in_sandbox", lambda *a: None)
    monkeypatch.setattr("sys.stdin", stdin)
    return agent.main([]), bench, parked


def test_the_end_of_the_input_answers_a_waiting_question_no_and_the_arm_goes_home_after_the_turn(monkeypatch):
    keyboard, answers = Keyboard(say=lambda s: None, wait_s=30), []

    def run_turn(text):
        t = time.monotonic()
        answers.append((keyboard.ask("Pick up the screwdriver?"), time.monotonic() - t))
        return "Stopped."
    code, bench, parked = run_main(monkeypatch, run_turn, keyboard, io.StringIO("pick up the screwdriver\n"))
    assert code == 0 and answers[0][0] is None and answers[0][1] < 2  # not after the 30 s
    assert bench.stopped.is_set() and parked == [True]


@pytest.mark.parametrize("can_go_home, answer, home", [(True, "\n", True), (True, "no\n", False), (False, None, False)])
def test_the_budget_stop_asks_before_taking_the_arm_home_and_holds_it_until_an_answer(monkeypatch, can_go_home, answer,
                                                                                      home):
    said = []
    keyboard = Keyboard(say=said.append, wait_s=0.05)
    stdin = Typing("pick up the screwdriver\n")

    def run_turn(text):
        if answer:
            answer_after(keyboard, answer, delay=0.3)  # long after the 0.05 s a question usually waits
        raise BudgetExceeded("the budget is spent")
    code, bench, parked = run_main(monkeypatch, run_turn, keyboard, stdin, can_go_home)
    stdin.done.set()
    assert code == 1 and parked == [home]
    assert len(said) == (1 if can_go_home else 0) and not any(s.startswith("No key") for s in said)


@pytest.mark.parametrize("asked", [False, True])
def test_the_end_of_the_input_before_or_during_the_budget_stops_question_takes_the_arm_home(monkeypatch, asked):
    said = []
    keyboard = Keyboard(say=said.append, wait_s=0.05)
    stdin = Typing("pick up the screwdriver\n")

    def run_turn(text):
        if asked:
            threading.Timer(0.3, stdin.done.set).start()  # Ctrl-D while it waits, past the 0.05 s a question takes
        else:
            stdin.done.set()  # Ctrl-D during the turn
            t = time.monotonic()
            while not keyboard.closed and time.monotonic() - t < 5:
                time.sleep(0.01)
        raise BudgetExceeded("the budget is spent")
    code, bench, parked = run_main(monkeypatch, run_turn, keyboard, stdin)
    assert code == 1 and parked == [True] and len(said) == (1 if asked else 0)


@pytest.mark.parametrize("ending", ["Ctrl-C", "the end of the input"])
def test_after_ctrl_c_nothing_is_staged_or_flashed_but_after_the_end_of_the_input_it_is(monkeypatch, ending):
    b, tb, _ = bench()
    b.models = {"coder": {"auto_stage": True}}
    b.layout = {"firmware": {"upload_cmd": "pio run -t upload"}}
    staged, flashed = [], []
    monkeypatch.setattr(tools.coder, "ask_coder", lambda *a: {"ok": True, "diff": "d", "why": "w"})
    monkeypatch.setattr(tools.coder, "propose_patch", lambda *a: staged.append(a) or {"ok": True})
    monkeypatch.setattr(tools.dut, "flash", lambda *a: flashed.append(a) or {"ok": True})
    monkeypatch.setattr(tools.dut, "find_port", lambda fw: "/dev/fake")
    if ending == "Ctrl-C":
        b.stop_arm()
        assert "staged" not in tb.dispatch("ask_coder", {"task": "fix the pin", "files": ["pins.h"]})
        assert tb.dispatch("build_and_flash", {}) == {"ok": False,
                                                      "error": "the session is ending, so nothing was flashed"}
        assert staged == [] and flashed == []
    else:  # piped input ends before the first turn: no arm move starts, but the turn's patch is staged
        b.stopped.set()
        assert tb.dispatch("ask_coder", {"task": "fix the pin", "files": ["pins.h"]})["staged"] == {"ok": True}
        assert tb.dispatch("build_and_flash", {})["ok"] and len(staged) == len(flashed) == 1


@pytest.mark.parametrize("flags, prefetch", [([], True), (["--arm", "--no-prefetch"], False)])
def test_no_prefetch_skips_the_bench_facts_fetched_before_each_turn(monkeypatch, flags, prefetch):
    build, built = agent.build, []
    bench = NS(park=lambda home=True: None, stop_arm=lambda: None, hand=None, stopped=threading.Event(),
               can_go_home=lambda: False, resume=lambda: None)
    budget = NS(spent=0.0, limit=5.0, check=lambda: None, pop_warning=lambda: None)
    fake = NS(label="Spark", model="spark", run_turn=lambda text: "Done.")
    parts = ({"ears": {}, "coder": {}, "mouth": {}}, budget, SilentHud(), fake, bench, Keyboard(say=lambda s: None))
    monkeypatch.setattr(agent, "build", lambda *a: built.append(a) or parts)
    monkeypatch.setattr(agent.coder, "compile_in_sandbox", lambda *a: None)
    monkeypatch.setattr("sys.stdin", io.StringIO("pick up the screwdriver\n"))
    assert agent.main(flags) == 0 and built[0][5] is prefetch
    monkeypatch.setattr(agent.config, "meta_client", lambda models: NS())  # so building needs no key
    router = build(*built[0])[3]
    assert bool(router.prefetch) is prefetch  # without it, the router fetches nothing before a turn


def test_go_ahead_comes_from_robot_yaml_and_enter_overrides_it(monkeypatch):
    monkeypatch.setattr(agent.config, "meta_client", lambda models: NS())  # so building needs no key
    assert agent.build(True, False)[4].go_ahead == "request"
    assert agent.build(True, False, enter=True)[4].go_ahead == "enter"
    robot, load = agent.config.load_yaml("robot.yaml"), agent.config.load_yaml
    monkeypatch.setattr(agent.config, "load_yaml",
                        lambda name: {**robot, "hand": {"go_ahead": "voice"}} if name == "robot.yaml" else load(name))
    with pytest.raises(RuntimeError, match="hand.go_ahead is request or enter, not 'voice'"):
        agent.build(True, False)


def test_a_spoken_stop_holds_the_arm_drops_what_waits_and_the_next_request_moves_it_again(monkeypatch):
    lines, did, turns, voice = queue.Queue(), [], [], {}
    bench = NS(resume=lambda: did.append("resume"), halt=lambda: did.append("halt"), park=lambda home=True: None,
               stop_arm=lambda: None, hand=None, stopped=threading.Event(), can_go_home=lambda: False)
    budget = NS(spent=0.0, limit=5.0, check=lambda: None, pop_warning=lambda: None)

    def run_turn(text):
        turns.append(text)
        if len(turns) == 1:
            voice["on_text"]("put the screwdriver in the tray")  # said while the turn runs: it waits behind it
            voice["on_stop"]()  # then "Copilot, stop"
            lines.put("go home\n")  # and a typed request after that
            return "Stopped, with the screwdriver in the jaws."
        lines.put(None)  # the end of the input
        return "Back at rest. Want me to put the screwdriver away?"

    def listen(on_text, cfg, key, budget, hud, muted=None, on_stop=None, question=None):
        voice.update(on_text=on_text, on_stop=on_stop, question=question)
        lines.put("pick up the screwdriver\n")

    hud = SilentHud()
    router = NS(label="Spark", model="spark", run_turn=run_turn)
    monkeypatch.setattr(agent, "build", lambda *a: ({"ears": {}, "coder": {}, "mouth": {}}, budget, hud, router, bench,
                                                    Keyboard(say=lambda s: None)))
    monkeypatch.setattr(agent.coder, "compile_in_sandbox", lambda *a: None)
    monkeypatch.setattr(agent.config, "api_key", lambda: "key")
    monkeypatch.setattr(agent.ears, "listen_forever", listen)
    monkeypatch.setattr("sys.stdin", iter(lines.get, None))
    assert agent.main(["--voice", "--no-speak"]) == 0
    assert turns == ["pick up the screwdriver", "go home"] and did == ["resume", "halt", "resume"]
    assert ("note", "stopped: no arm move starts until the next request; dropped what was waiting: put the "
                    "screwdriver in the tray") in hud.events
    assert voice["question"].at is not None  # the last reply asked something, so a bare "yes" answers it


def test_the_wbc_script_ends_the_way_python_m_does():
    assert 'wbc = "wbc.agent.__main__:cli"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert callable(agent.cli)
