"""The table under the arm (wbc/motion/table.py), and the driver's floors measured from it."""
import importlib.util
import json
import math

import numpy as np
import pytest

from test_arm import CHAIN, MAP, ROBOT_CFG, FakeFollower, make, pose
from wbc.config import ROOT
from wbc.motion import arm as driver
from wbc.motion.arm import Refused
from wbc.motion.table import BEARING_SLACK, MIN_MARGIN, PAN_AXIS, Surface
from wbc.motion.trajectory import lowest

# Tonight's desk tags (Oct 29, 23:05), mm in the arm's frame
TAGS = {21: [381, -18, -38.0], 22: [261, 21, -29.5], 23: [33, -286, -35.5], 24: [262, -330, -43.9],
        25: [124, 43, -16.4]}
DESK = Surface.fit([np.array(p) / 1000 for p in TAGS.values()])
REST = [0, -20, 40, 30, 0]


def test_the_pan_axis_is_where_the_urdf_turns_the_base():
    axis = CHAIN.frames(np.zeros(5))[1][0][0][:2]
    assert axis == pytest.approx(PAN_AXIS, abs=1e-7)  # so turning only the base keeps every height level
    assert axis == pytest.approx(np.array(ROBOT_CFG["table"]["axis_mm"]) / 1000, abs=1e-7)


def test_the_fit_is_the_line_in_reach_from_the_pan_axis_through_the_touches_with_its_worst_miss_as_the_margin():
    assert DESK.config() == {"at_base_mm": -8.7, "slope_mm_per_m": -88.9, "reach_mm": [95, 399], "margin_mm": 2.0,
                             "axis_mm": [38.8353, 0.0], "bearing_deg": [-92, 27]}
    exact = Surface.fit([(0.15, 0, -0.020), (0, 0.25, -0.030), (-0.2, -0.2, -0.033284)], axis=(0, 0))  # -5 - 100 r
    assert exact.at_base == pytest.approx(-0.005, abs=1e-4) and exact.slope == pytest.approx(-0.1, abs=1e-3)
    assert exact.margin == MIN_MARGIN  # a perfect fit still keeps 2 mm
    from_origin = Surface.fit([np.array(p) / 1000 for p in TAGS.values()], axis=(0, 0))
    assert from_origin.margin > 3 * DESK.margin / 2  # the tags fit worse measured from the URDF's origin


@pytest.mark.parametrize("points, match", [
    ([(0.15, 0, -0.02), (0.3, 0, -0.03)], "3 touched points or more"),
    ([(0.15, 0, -0.02), (0, 0.16, -0.02), (0.17, 0, -0.021)], "spread over 100 mm"),
    ([(0.15, 0, -0.02), (0.25, 0, -0.03), (0.35, 0, -0.04), (0.2, 0, -0.005)], "200 mm out is .* off the line"),
    # a bad touch at the end of the reach, which pulls a line through every point toward itself
    ([(0.15, 0, -0.02), (0.20, 0, -0.025), (0.25, 0, -0.03), (0.40, 0, -0.065)], "400 mm out is .* off the line"),
])
def test_too_few_touches_too_little_reach_or_a_missed_touch_fits_nothing(points, match):
    with pytest.raises(ValueError, match=match):
        Surface.fit(points, axis=(0, 0))


@pytest.mark.parametrize("surface", [Surface(-0.006, -0.09, (0.13, 0.42), 0.0036),
                                     Surface(-0.030, 0.05, (0.13, 0.42), 0.002)])
def test_outside_the_touched_reach_the_table_is_never_lower_than_at_the_nearest_touch(surface):
    for r in (0.0, 0.05, 0.13):
        assert surface.desk(r, 0) >= surface.line(0.13) - 1e-12 and surface.desk(r, 0) >= surface.line(r) - 1e-12
    for r in (0.42, 0.5, 0.8):
        assert surface.desk(0, r) >= surface.line(0.42) - 1e-12 and surface.desk(0, r) >= surface.line(r) - 1e-12
    assert surface.desk(0.2, 0.1) == pytest.approx(surface.line(math.hypot(0.2, 0.1)))  # inside: the line
    assert surface.z(0.2, 0.1) == pytest.approx(surface.desk(0.2, 0.1) + surface.margin)


def at(bearing: float, r: float = 0.25) -> tuple:
    """(x, y) metres, `r` from the pan axis at `bearing` radians."""
    return DESK.axis[0] + r * math.cos(bearing), DESK.axis[1] + r * math.sin(bearing)


def test_at_a_bearing_no_touch_covered_the_table_is_the_highest_the_line_reached():
    for bearing in (DESK.bearing[1] + BEARING_SLACK + 0.05, math.radians(150), DESK.bearing[0] - BEARING_SLACK - 0.05):
        assert DESK.desk(*at(bearing)) == pytest.approx(DESK.line(DESK.reach[0]))  # 25 cm out, held at the nearest
        assert DESK.held(*at(bearing)) == pytest.approx(DESK.line(DESK.reach[0]) - DESK.line(0.25))  # touch's
    for bearing in DESK.bearing:
        assert DESK.desk(*at(bearing)) == pytest.approx(DESK.line(0.25)) and DESK.held(*at(bearing)) < 1e-12


def test_past_the_touched_bearings_the_table_rises_evenly_so_the_bases_own_error_never_steps_it():
    for edge, way in ((DESK.bearing[1], 1), (DESK.bearing[0], -1)):
        halfway = DESK.desk(*at(edge + way * BEARING_SLACK / 2, 0.29))
        assert halfway == pytest.approx((DESK.line(0.29) + DESK.line(DESK.reach[0])) / 2)
        heights = [DESK.desk(*at(edge + way * math.radians(d), 0.29)) for d in np.arange(-1, 12, 0.5)]
        assert all(b >= a - 1e-12 for a, b in zip(heights, heights[1:]))  # only ever rising, going round
        assert max(b - a for a, b in zip(heights, heights[1:])) < 0.001  # half a degree moves it under a millimetre


def test_robot_yaml_reads_a_table_a_level_table_or_neither():
    assert Surface.from_config({"table": DESK.config()}).config() == DESK.config()
    assert Surface.from_config({"table_mm": -17}) == Surface.flat(-0.017)
    assert Surface.from_config({}) == Surface.flat(0.0)


@pytest.mark.parametrize("kwargs, match", [
    ({"at_base": -0.2, "slope": 0.0}, "height at the base, -200 mm"),
    ({"at_base": -0.006, "slope": -0.9}, "slope, -900 mm per m"),
    ({"at_base": -0.006, "slope": 0.0, "margin": 0.05}, "margin, 50.0 mm"),
    ({"at_base": -0.006, "slope": -0.19, "reach": (0.1, 0.4)}, "height 400 mm out, -82 mm"),  # -90.1 typed as -190
    ({"at_base": -0.006, "slope": float("nan")}, "isn't a number"),
    ({"at_base": float("nan"), "slope": -0.09}, "isn't a number"),
])
def test_a_table_that_looks_like_a_mistake_is_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Surface(**kwargs)


def test_config_rounds_the_safe_way():
    s = Surface(-0.0087, -0.0889, (0.0949, 0.3991), 0.00201, PAN_AXIS, (-1.6, 0.47))
    c = s.config()
    assert c["margin_mm"] == 2.1 and c["reach_mm"] == [94, 400] and c["bearing_deg"] == [-92, 27]


def test_lowest_measures_each_point_from_the_table_under_it():
    q = np.radians([20, 10, 60, 20, 0])
    T, axes = CHAIN.frames(q)
    tool, axis = lowest(CHAIN, q, driver.FLOOR_JOINTS, DESK)
    assert tool == pytest.approx(T[2, 3] - DESK.z(T[0, 3], T[1, 3]))
    named = [CHAIN.names.index(m) for m in driver.FLOOR_JOINTS]
    assert axis == pytest.approx(min(axes[i][0][2] - DESK.z(axes[i][0][0], axes[i][0][1]) for i in named))
    assert lowest(CHAIN, q, driver.FLOOR_JOINTS, -0.017) == lowest(CHAIN, q, driver.FLOOR_JOINTS, Surface.flat(-0.017))


def under_the_tool(surface, x_mm, above_mm, start=REST):
    """Joint angles, solved from `start`, with the tool point straight down `above_mm` over `surface` at x_mm out."""
    from wbc.motion.kinematics import solve_task

    x = x_mm / 1000
    res = solve_task(CHAIN, x, 0, surface.z(x, 0) + above_mm / 1000, math.radians(90), 0, q0=np.radians(start))
    assert res.ok
    return res.q


def test_the_floors_follow_the_desk_so_the_tips_get_down_to_it_further_out():
    target = under_the_tool(DESK, 250, 7)  # 7 mm over the floors' table, 250 mm out: 34 mm under a level -17
    arm, robot, _ = make(FakeFollower(start=pose(REST)), table=DESK, tool_floor=0.005)
    arm.power_on()
    arm.move_to(target)
    assert lowest(CHAIN, arm.read().q, driver.FLOOR_JOINTS, DESK)[0] == pytest.approx(0.007, abs=0.002)
    flat, robot, _ = make(FakeFollower(start=pose(REST)), table=-0.017, tool_floor=0.005)
    flat.power_on()
    with pytest.raises(Refused, match="under its 5 mm floor"):
        flat.move_to(target)


def test_the_tool_floor_holds_over_the_desk_too():
    target = under_the_tool(DESK, 250, 3)  # 3 mm over the floors' table: under the 5 mm floor
    arm, robot, _ = make(FakeFollower(start=pose(REST)), table=DESK, tool_floor=0.005)
    arm.power_on()
    with pytest.raises(Refused, match=r"the tool point down to 3 mm above the table, under its 5 mm floor"):
        arm.move_to(target)


def test_from_under_its_floor_a_base_turn_still_cant_take_the_tool_point_any_lower():
    # A climb's goals, in whole ticks, may dip ROUNDING as they set off, but a turn's don't, so a turn under a floor
    # may go no lower at all: past the touched bearings, turning on round raises the table under the tool point
    q = under_the_tool(DESK, 1000 * DESK.axis[0] + 150, 3)  # 3 mm over the floors' table, 150 mm out
    q[0] -= math.radians(27.5)  # turned to just past the touched +27°: the pan turns clockwise, seen from above
    arm, robot, _ = make(FakeFollower(start=pose(np.degrees(q))), table=DESK, tool_floor=0.005)
    arm.power_on()
    start = arm.read()
    on, back = ({**start.raw, "shoulder_pan": start.raw["shoulder_pan"] + d} for d in (-1, 1))  # a tick either way

    def height(raw):
        return lowest(CHAIN, MAP.q(raw), driver.FLOOR_JOINTS, DESK)[0]
    assert driver.LEVEL < height(start.raw) - height(on) < driver.ROUNDING  # 43 µm a tick, 150 mm out
    with pytest.raises(Refused, match="lower than the 3 mm it starts at, which is already under its 5 mm floor"):
        arm.move_to(MAP.q(on))
    arm.move_to(MAP.q(back))  # back toward the touches it may stay under its floor


def desk_camera():
    spec = importlib.util.spec_from_file_location("desk_camera", ROOT / "scripts" / "desk_camera.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_desk_camera_table_prints_the_fit_for_robot_yaml(tmp_path):
    path = tmp_path / "desk_tags.json"
    path.write_text(json.dumps({"tags": {str(k): v for k, v in TAGS.items()}}), encoding="utf-8")
    said = []
    args = type("Args", (), {"file": path})()
    assert desk_camera().cmd_table(args, {}, say=said.append) == 0
    assert any(line.startswith("  tag 23: 286 mm out, z -35.5 mm,") for line in said)
    assert said[-1].endswith("table:\n  at_base_mm: -8.7\n  slope_mm_per_m: -88.9\n  reach_mm: [95, 399]\n"
                             "  margin_mm: 2.0\n  axis_mm: [38.8353, 0.0]\n  bearing_deg: [-92, 27]")
