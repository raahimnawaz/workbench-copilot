"""Moves in mm, picking things up and putting them down (wbc/motion/hand.py), and the agent's arm tools, against the
arm tests' fake follower."""
import math
import threading

import numpy as np
import pytest

from test_arm import CHAIN, CLOSED, ROBOT_CFG, FakeFollower, make, pose
from wbc.agent import tools
from wbc.agent.approve import Keyboard
from wbc.config import ROOT
from wbc.motion import arm as driver
from wbc.motion.arm import Refused
from wbc.motion.hand import FLOOR_MARGIN_MM, HOVER_MM, Hand, Thing, line
from wbc.motion.kinematics import pose_to_task
from wbc.motion.trajectory import lowest

REST = [0, -20, 40, 30, 0]  # the arm tests' START: the tool point 64 mm above the table
TABLE = -0.017              # config/robot.yaml's table_mm
SCREWDRIVER = Thing("screwdriver", 205, -60, -30, heading_deg=15, length_mm=176, width_mm=27)
TRAY = Thing("tray", 175, 125, -10, heading_deg=5, length_mm=210, width_mm=120, height_mm=20)


def hand_on(table=TABLE, start=REST, **kwargs):
    """A Hand on a fake arm, powered on at `start` (URDF degrees)."""
    arm, robot, said = make(FakeFollower(start=pose(start)), table=table, **kwargs)
    hand = Hand(arm, {"hand": {"across_roll_deg": 0}})
    hand.power_on()
    robot.events.clear()
    robot.sent.clear()
    return hand, robot


def tool_at(q) -> tuple:
    x, y, z, pitch, _ = pose_to_task(CHAIN.fk(q))
    return 1000 * np.array([x, y, z]), math.degrees(pitch)


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


def test_a_pick_goes_down_to_the_drivers_floor_over_the_table_for_something_lying_flat():
    hand, _ = hand_on()
    over, down, up = hand.plan_pick(SCREWDRIVER)
    floor = 1000 * (TABLE + driver.TOOL_FLOOR) + FLOOR_MARGIN_MM
    assert down.xyz == (205, -60, floor) and over.xyz == up.xyz == (205, -60, floor + HOVER_MM)
    assert [line(s.jaws_deg - SCREWDRIVER.jaws_deg) for s in (over, down, up)] == pytest.approx([0, 0, 0], abs=1)


def test_with_the_table_at_the_bases_underside_the_pick_stops_higher():
    hand, _ = hand_on(table=0.0)
    assert hand.plan_pick(SCREWDRIVER)[1].xyz[2] == pytest.approx(1000 * driver.TOOL_FLOOR + FLOOR_MARGIN_MM)


def test_pick_up_goes_over_the_thing_opens_goes_down_grips_and_lifts_it():
    hand, robot = hand_on()
    robot.between["gripper"] = (1400, math.inf)  # the screwdriver's handle between the jaws
    result = hand.pick_up(SCREWDRIVER)
    assert result["ok"] and result["holding"] == "screwdriver" and hand.held == "screwdriver"
    assert [s["step"] for s in result["steps"]] == ["over it", "down to it", "back up"]
    at, pitch = tool_at(hand.arm.read().q)
    assert at == pytest.approx(np.array([205, -60, result["steps"][2]["z_mm"]]), abs=2)
    assert robot.goal["gripper"] == CLOSED  # still squeezing
    assert hand.arm.jaw().holding


def test_pick_up_on_nothing_is_a_miss_and_opens_the_jaws_again():
    hand, robot = hand_on()
    result = hand.pick_up(SCREWDRIVER)
    assert not result["ok"] and result["missed"] and hand.held is None
    assert hand.arm.jaw().opening > 90
    assert tool_at(hand.arm.read().q)[0][2] == pytest.approx(result["steps"][2]["z_mm"], abs=2)  # back up


def test_saying_no_over_the_thing_stops_there_with_the_jaws_as_they_were():
    hand, robot = hand_on()
    jaw = robot.pos["gripper"]
    asked = []
    result = hand.pick_up(SCREWDRIVER, confirm=lambda q: asked.append(q) or False)
    assert not result["ok"] and "over the screwdriver" in result["stopped"] and "Go down and grip it?" in asked[0]
    assert robot.pos["gripper"] == jaw
    assert tool_at(hand.arm.read().q)[0] == pytest.approx(np.array([205, -60, result["steps"][0]["z_mm"]]), abs=2)


def test_put_down_lowers_it_over_the_place_lets_go_and_goes_back_up():
    hand, robot = hand_on()
    robot.between["gripper"] = (1400, math.inf)
    hand.pick_up(SCREWDRIVER)
    robot.between.pop("gripper")
    result = hand.put_down(TRAY)
    assert result["ok"] and result["put"] == "screwdriver" and result["on"] == "tray" and hand.held is None
    assert result["steps"][1]["z_mm"] == pytest.approx(TRAY.top_z + 30)
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
    with pytest.raises(ValueError, match="the table has to be from -60 to 10 mm"):
        make(table=-17)


def test_robot_yaml_sets_the_table_and_the_jaws_line():
    assert ROBOT_CFG["table_mm"] == -17 and ROBOT_CFG["hand"]["across_roll_deg"] == 0


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
    assert found["ok"] and found["found"][0]["x_mm"] == 205.0 and b.seen["screwdriver"].x == 205.0
    assert tb.dispatch("put_down", {"thing": "tray"})["ok"] is False  # nothing held yet
    assert tb.dispatch("pick_up", {"thing": "screwdriver"}) == {"ok": True, "sim": True, "holding": "screwdriver",
                                                                 "x_mm": 205.0, "y_mm": -60.0}
    assert tb.dispatch("put_down", {"thing": "tray"})["put"] == "screwdriver"
    assert tb.dispatch("locate", {"thing": "hammer"})["found"] == []
    assert "no hammer" in tb.dispatch("pick_up", {"thing": "hammer"})["error"]


def test_without_arm_the_arms_tools_say_how_to_turn_it_on():
    b, tb, asked = bench()
    b.seen["screwdriver"] = SCREWDRIVER
    assert "--arm" in tb.dispatch("pick_up", {"thing": "screwdriver"})["error"] and asked == []


def test_the_arm_moves_only_after_enter_and_powers_on_first():
    arm, robot, _ = make(FakeFollower(start=pose(REST)), table=TABLE)
    b, tb, asked = bench(answers=(False,), arm_port="fake")
    b.hand = Hand(arm, {"hand": {"across_roll_deg": 0}})
    b.seen["screwdriver"] = SCREWDRIVER
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
    b.hand = Hand(arm, {"hand": {"across_roll_deg": 0}})
    b.seen["box"] = Thing("box", 600, 0, -30, length_mm=100, width_mm=40)
    result = tb.dispatch("pick_up", {"thing": "box"})
    assert not result["ok"] and result["error"].startswith("refused, nothing moved for it: the arm can't put")


def test_park_takes_the_arm_home_and_turns_the_motors_off():
    arm, robot, _ = make(FakeFollower(start=pose(REST)), table=TABLE)
    b, tb, _ = bench(arm_port="fake")
    b.hand = Hand(arm, {"hand": {"across_roll_deg": 0}})
    b.seen["screwdriver"] = SCREWDRIVER
    robot.between["gripper"] = (1400, math.inf)
    assert tb.dispatch("pick_up", {"thing": "screwdriver"})["ok"]
    b.park()
    assert np.degrees(arm.joints.q(robot.pos)) == pytest.approx(REST, abs=0.5)
    assert not any(robot.torque.values())


# Enter at the keyboard


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
