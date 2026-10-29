"""Quintic joint-space moves, against the real SO-101 URDF."""
import numpy as np
import pytest
import yaml

from wbc.config import ROOT
from wbc.motion.kinematics import load_chain
from wbc.motion.trajectory import PEAK, joint_path, lowest, min_seconds, quintic

JOINT_CAP = np.radians(30)  # rad/s at a move's peak, the driver's cap
TIP_CAP = 0.08              # m/s at a move's peak: the plan's 80 mm/s near the bench
HZ = 50.0                   # the driver's tick rate
REST = np.radians([4.66, -111.09, 92.11, -77.96, 89.03])  # the folded rest pose, read on Oct 28
L_POSE = np.zeros(5)  # upper arm up, forearm and gripper level pointing forward


@pytest.fixture(scope="module")
def chain():
    return load_chain(yaml.safe_load((ROOT / "config" / "robot.yaml").read_text()), ROOT)


def peak_speeds(chain, q0, path, hz) -> tuple:
    """The fastest any joint (rad/s) and the tool point (m/s) go from one goal to the next along a path."""
    rows = np.vstack([q0, path])
    tips = np.array([chain.fk(q)[:3, 3] for q in rows])
    return np.abs(np.diff(rows, axis=0)).max() * hz, np.linalg.norm(np.diff(tips, axis=0), axis=1).max() * hz


def test_a_quintic_runs_0_to_1_and_starts_and_stops_with_no_speed_or_acceleration():
    assert quintic(0.0) == 0 and quintic(0.5) == 0.5 and quintic(1.0) == 1
    s = np.linspace(0, 1, 1001)
    assert np.all(np.diff(quintic(s)) > 0)  # never goes back
    h = 1e-4
    for end in (0.0, 1.0):
        speed = (quintic(end + h) - quintic(end - h)) / (2 * h)
        acceleration = (quintic(end + h) - 2 * quintic(end) + quintic(end - h)) / h ** 2
        assert abs(speed) < 1e-6 and abs(acceleration) < 1e-3
    assert np.gradient(quintic(s), s).max() == pytest.approx(PEAK, rel=1e-5)  # 15/8 of its average, halfway


def test_a_path_has_one_row_per_tick_and_ends_exactly_on_its_target():
    q0, q1 = np.array([0.1, -0.2, 0.3, 0.0, 1.0]), np.array([0.3, 0.1, -0.4, 0.2, 1.0])
    path = joint_path(q0, q1, seconds=2.0, hz=HZ)
    assert path.shape == (100, 5) and np.array_equal(path[-1], q1)
    steps = np.diff(np.vstack([q0, path]), axis=0)
    assert np.all(steps[:, :4] * (q1 - q0)[:4] > 0)  # every moving joint heads for its target on every tick
    assert np.all(path[:, 4] == 1.0)  # and one that isn't moving stays exactly put
    assert np.abs(steps[[0, -1]]).max() < 0.001 * np.abs(steps[50]).max()  # it eases out of rest and into it


def test_a_path_never_runs_faster_than_asked():
    q0, q1 = np.zeros(5), np.full(5, 0.1)
    assert len(joint_path(q0, q1, 1.01, HZ)) == 51  # 50.5 ticks: rounded up, not to 50
    assert len(joint_path(q0, q1, 0.3, HZ)) == 15  # 15.000000000000002 ticks in floating point
    one = joint_path(q0, q1, 0.001, HZ)
    assert one.shape == (1, 5) and np.array_equal(one[0], q1)


def test_min_seconds_keeps_every_joint_at_its_cap_at_the_peak_and_never_over(chain):
    q0, q1 = L_POSE, np.radians([40, -25, 60, -30, 90])
    seconds = min_seconds(q0, q1, JOINT_CAP)
    assert seconds == pytest.approx(np.radians(90) * PEAK / JOINT_CAP)  # the wrist roll goes furthest: 5.625 s
    for hz in (HZ, 400):
        joint, _ = peak_speeds(chain, q0, joint_path(q0, q1, seconds, hz), hz)
        assert joint <= JOINT_CAP
    assert joint > 0.999 * JOINT_CAP  # the shortest such move: at 400 Hz its fastest tick is at the cap


@pytest.mark.parametrize("q0, q1", [
    (L_POSE, np.radians([90, 0, 0, 0, 0])),  # reaching out level, the base swings 90°: the tool point goes 55 cm
    (REST, np.radians([15, 20, 30, 40, 0])),  # from rest to reaching forward over the bench
    (L_POSE, np.radians([0, 0, 0, 80, 0])),  # the gripper tips 80° down at the wrist flex
], ids=["base swing", "rest to the bench", "gripper tipped down"])
def test_min_seconds_keeps_the_tool_point_under_its_cap(chain, q0, q1):
    seconds = min_seconds(q0, q1, JOINT_CAP, chain, TIP_CAP)
    assert seconds > min_seconds(q0, q1, JOINT_CAP)  # near the bench, the tool point sets the pace
    for hz in (HZ, 400):
        joint, tip = peak_speeds(chain, q0, joint_path(q0, q1, seconds, hz), hz)
        assert tip <= TIP_CAP and joint <= JOINT_CAP


def test_min_seconds_is_tight_when_the_tool_point_moves_at_one_speed_the_whole_way(chain):
    q0, q1 = L_POSE, np.radians([90, 0, 0, 0, 0])  # the base swinging the outstretched arm: a circle
    _, tip = peak_speeds(chain, q0, joint_path(q0, q1, min_seconds(q0, q1, JOINT_CAP, chain, TIP_CAP), 400), 400)
    assert 0.999 * TIP_CAP < tip <= TIP_CAP


def test_min_seconds_is_zero_for_a_move_that_goes_nowhere(chain):
    assert min_seconds(REST, REST, JOINT_CAP, chain, TIP_CAP) == 0.0
    assert min_seconds(REST, REST, JOINT_CAP) == 0.0


def test_min_seconds_refuses_caps_it_cant_keep(chain):
    with pytest.raises(ValueError, match="max_joint_speed has to be above 0"):
        min_seconds(L_POSE, REST, 0.0)
    with pytest.raises(ValueError, match="max_tip_speed has to be above 0"):
        min_seconds(L_POSE, REST, JOINT_CAP, chain, -0.08)
    with pytest.raises(ValueError, match="needs the chain"):
        min_seconds(L_POSE, REST, JOINT_CAP, max_tip_speed=TIP_CAP)


def test_lowest_reads_every_row_of_a_path(chain):
    down = np.radians([0, 0, 0, 80, 0])  # forearm level, the gripper tipped 80° down
    path = np.array([L_POSE, down, L_POSE])
    tool, axis = lowest(chain, path, ("elbow_flex", "wrist_flex", "wrist_roll"))
    T, axes = chain.frames(down)
    assert tool == pytest.approx(T[2, 3]) and tool < chain.fk(L_POSE)[2, 3] - 0.1  # 227 mm above the table to 76
    assert axis == pytest.approx(axes[4][0][2])  # the wrist roll's axis dips lowest, to 174 mm
    assert lowest(chain, path, ("elbow_flex",))[1] == pytest.approx(chain.frames(L_POSE)[1][2][0][2])  # 229 mm
    assert lowest(chain, down, ())[1] == np.inf  # a single pose is a path too
