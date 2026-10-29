"""The read-only registration script, against fake servos posed so the tool point lands where each touch should be."""
import importlib.util
import math

import numpy as np
import pytest

from test_carrier import BENCH, CHAIN, CORNERS, MAP, PARTS, ROBOT_CFG, TRUE
from test_check_joint_map import Clock, FakeLine
from test_joints import REST
from wbc.config import ROOT
from wbc.motion import carrier, sts
from wbc.motion.carrier import Registration, point_mm

_spec = importlib.util.spec_from_file_location("register_arm", ROOT / "scripts" / "register_arm.py")
register_arm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(register_arm)


def pose_at(spot) -> dict:
    """Raw positions that put the tool point on `spot` (arm frame, metres), the gripper near straight down."""
    for degrees in carrier.PITCHES:
        found = carrier._toward(CHAIN, np.asarray(spot, float), 0.0, math.radians(degrees), MAP.q(REST))
        if found is not None:
            return {**REST, **MAP.raw(found[0])}
    raise AssertionError(f"no pose reaches {spot}")


def touch(line, name, off=(0.0, 0.0, 0.0), reg=TRUE):
    """A reply that puts the tool point on a named point, or `off` metres away, then presses Enter."""
    raw = pose_at(reg.to_arm(point_mm(BENCH, PARTS, name)) + np.asarray(off))

    def act():
        line.pose(raw)
        return ""
    return act


def session(line, mode, names, path, replies=()):
    replies, said, record, clock = list(replies), [], {}, Clock()

    def ask(prompt):
        said.append(prompt)
        if not replies:
            raise EOFError
        reply = replies.pop(0)
        return reply() if callable(reply) else reply

    bus = sts.Bus("fake", serial_factory=lambda port, **kwargs: line)
    ok = register_arm.run(bus, mode, ROBOT_CFG, BENCH, PARTS, names, record, path, ask=ask,
                          say=lambda text="", **kwargs: said.append(text), sleep=clock.sleep, clock=clock, seconds=0.5)
    return ok, record, "\n".join(said)


def test_touching_the_four_tags_saves_a_registration_that_matches_the_sheet(tmp_path):
    line, path = FakeLine(), tmp_path / "registration.json"
    ok, record, said = session(line, "touch", CORNERS, path, [touch(line, n) for n in CORNERS])
    assert ok and "Saved to" in said and line.written() == []
    reg = Registration.load(path)
    for name in ("a1", "j63", "tag11"):  # anywhere on the sheet, within the joints' whole ticks
        mm = point_mm(BENCH, PARTS, name)
        assert np.linalg.norm(reg.to_arm(mm) - TRUE.to_arm(mm)) < 0.0015
    assert math.degrees(reg.yaw) == pytest.approx(-86.0, abs=0.5) and reg.rms_mm < 1.0
    assert len(record["touched"]) == 4


def test_a_touch_off_its_mark_isnt_saved_and_is_named(tmp_path):
    line, path = FakeLine(), tmp_path / "registration.json"
    replies = [touch(line, n, off=(0.0, 0.008, 0.0) if n == "tag12" else (0, 0, 0)) for n in CORNERS]
    ok, _, said = session(line, "touch", CORNERS, path, replies)
    assert not ok and not path.exists()
    assert "Not saved" in said and "tag12 is the furthest off" in said


def test_too_few_touches_save_nothing(tmp_path):
    line, path = FakeLine(), tmp_path / "registration.json"
    ok, record, said = session(line, "touch", CORNERS, path, [touch(line, "tag10"), "s", touch(line, "tag12"), "s"])
    assert not ok and not path.exists() and "2 points touched" in said


def test_q_stops_the_touches_and_keeps_what_was_done(tmp_path):
    line, path = FakeLine(), tmp_path / "registration.json"
    replies = [touch(line, n) for n in CORNERS[:3]] + ["q"]
    ok, record, said = session(line, "touch", CORNERS, path, replies)
    assert ok and "Stopped." in said and [t["name"] for t in record["touched"]] == CORNERS[:3]


def test_it_wont_register_with_a_motor_on(tmp_path):
    line = FakeLine(torque=("elbow_flex",))
    ok, _, said = session(line, "touch", CORNERS, tmp_path / "r.json")
    assert not ok and "Torque is on for the elbow" in said


def test_verify_passes_touches_on_their_holes_and_fails_ones_4_mm_off(tmp_path):
    path = tmp_path / "registration.json"
    TRUE.save(path)
    holes = ["a1", "j63", "a5"]
    line = FakeLine()
    ok, record, said = session(line, "verify", holes, path, [touch(line, n) for n in holes])
    assert ok and "within the 3 mm touch test" in said and line.written() == []
    assert all(r["across_mm"] < 1.0 and abs(r["dz_mm"]) < 1.0 for r in record["verify"])
    line = FakeLine()
    ok, record, said = session(line, "verify", holes, path, [touch(line, n, off=(0.004, 0, 0)) for n in holes])
    assert not ok and "over the 3 mm touch test" in said


def test_verify_says_how_far_the_board_top_is_from_its_setting(tmp_path):
    path = tmp_path / "registration.json"
    TRUE.save(path)
    line = FakeLine()
    ok, _, said = session(line, "verify", ["a1", "j63"], path, [touch(line, n, off=(0, 0, 0.003)) for n in ("a1", "j63")])
    assert "+3.0 mm from the board_top_mm height" in said


def test_verify_and_plan_without_a_registration_say_to_touch_first(tmp_path):
    for mode in ("verify", "plan"):
        ok, _, said = session(FakeLine(), mode, ["a5"], tmp_path / "none.json")
        assert not ok and "register_arm.py touch" in said


def test_watch_names_the_hole_under_the_tool_point(tmp_path):
    path = tmp_path / "registration.json"
    TRUE.save(path)
    line = FakeLine()
    line.pose(pose_at(TRUE.to_arm(point_mm(BENCH, PARTS, "c30") + [0, 0, 12])))
    ok, _, said = session(line, "watch", [], path)
    assert ok and "over c30" in said and "+12 mm from the board's top" in said and line.written() == []


def test_plan_says_where_point_at_would_go_and_moves_nothing(tmp_path):
    path = tmp_path / "registration.json"
    TRUE.save(path)
    line = FakeLine(torque=("shoulder_pan",))  # plan works with the motors on too
    line.pose(pose_at(TRUE.to_arm(point_mm(BENCH, PARTS, "j1") + [0, 0, 80])))
    ok, _, said = session(line, "plan", ["a5"], path)
    assert ok and "point_at a5: the gripper" in said and "first, further back" in said and line.written() == []


def test_plan_says_when_the_arm_starts_past_where_moves_may_take_it(tmp_path):
    path = tmp_path / "registration.json"
    TRUE.save(path)
    line = FakeLine()
    raw = pose_at(TRUE.to_arm(point_mm(BENCH, PARTS, "j1") + [0, 0, 80]))
    line.pose({**raw, "wrist_roll": 2047 + round(177.9 / 360 * 4096)})  # the Oct 28 L pose's roll
    ok, _, said = session(line, "plan", ["a5"], path)
    assert "The wrist roll is past where moves may take it now" in said


def test_main_refuses_names_that_arent_points_and_plan_wants_one(capsys):
    with pytest.raises(SystemExit):
        register_arm.main(["touch", "tag10", "k9", "--port", "/dev/cu.fake"])
    assert "isn't a tag or a hole" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        register_arm.main(["plan", "--port", "/dev/cu.fake"])


def test_main_sends_nothing_but_ping_and_read(monkeypatch, tmp_path):
    path = tmp_path / "registration.json"
    TRUE.save(path)
    line = FakeLine()
    line.pose(pose_at(TRUE.to_arm(point_mm(BENCH, PARTS, "j1") + [0, 0, 80])))
    real = sts.Bus
    monkeypatch.setattr(sts, "Bus", lambda port, **kwargs: real(port, serial_factory=lambda p, **k: line))
    assert register_arm.main(["plan", "a5", "--port", "/dev/cu.fake", "--file", str(path)]) == 0
    assert line.written() == []
