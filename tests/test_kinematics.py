"""FK/IK round trips on the real SO-101 URDF: random joint angles → pose → IK → the same pose."""
import numpy as np
import pytest
import yaml

from wbc.config import ROOT
from wbc.motion.kinematics import (Chain, load_chain, pose_to_task, rot_x, rot_y, rot_z, rotation_error,
                                   solve_ik, solve_task, task_to_pose)


@pytest.fixture(scope="module")
def chain():
    return load_chain(yaml.safe_load((ROOT / "config" / "robot.yaml").read_text()), ROOT)


def sample_q(chain, rng, margin=0.9):
    mid, half = (chain.lower + chain.upper) / 2, (chain.upper - chain.lower) / 2
    return rng.uniform(mid - margin * half, mid + margin * half)


def bench_poses(chain, rng, count):
    """Random reachable poses in the bench workspace: in front of the base and above the table."""
    poses = []
    while len(poses) < count:
        q = sample_q(chain, rng)
        T = chain.fk(q)
        if T[0, 3] > 0.05 and T[2, 3] > 0.0:
            poses.append((q, T))
    return poses


def test_chain_matches_robot_yaml(chain):
    cfg = yaml.safe_load((ROOT / "config" / "robot.yaml").read_text())
    assert chain.names == cfg["joints"]


def test_jacobian_matches_finite_differences(chain):
    rng = np.random.default_rng(3)
    q = sample_q(chain, rng)
    J, T0 = chain.jacobian(q)
    eps = 1e-6
    for i in range(chain.n):
        dq = np.zeros(chain.n)
        dq[i] = eps
        T1 = chain.fk(q + dq)
        assert np.allclose((T1[:3, 3] - T0[:3, 3]) / eps, J[:3, i], atol=1e-5)
        assert np.allclose(rotation_error(T1[:3, :3], T0[:3, :3]) / eps, J[3:, i], atol=1e-5)


def test_round_trip_full_pose(chain):
    failures = []
    for q_true, T in bench_poses(chain, np.random.default_rng(42), 200):
        res = solve_ik(chain, T[:3, 3], T[:3, :3], rng=np.random.default_rng(7))
        T_ik = chain.fk(res.q)
        pos_mm = np.linalg.norm(T_ik[:3, 3] - T[:3, 3]) * 1000
        rot_deg = np.degrees(np.linalg.norm(rotation_error(T[:3, :3], T_ik[:3, :3])))
        if not (res.ok and pos_mm < 0.2 and rot_deg < 0.5):
            failures.append((np.round(q_true, 2), pos_mm, rot_deg))
    assert not failures, f"{len(failures)} of 200 poses failed, e.g. {failures[:3]}"


def test_round_trip_task_frame(chain):
    """The 5-number grasp API: (x, y, z, pitch, roll) round-trips, with yaw left to the solver.

    Uses grasp-like poses: 12-40 cm out, below 30 cm, tool tilted down 0-89°.
    """
    rng, poses = np.random.default_rng(11), []
    while len(poses) < 150:
        T = chain.fk(sample_q(chain, rng))
        x, y, z, pitch, roll = pose_to_task(T)
        if x > 0.05 and 0.12 < np.hypot(x, y) < 0.40 and 0 < z < 0.30 and 0 <= pitch < np.radians(89):
            poses.append((x, y, z, pitch, roll))
    failures = []
    for x, y, z, pitch, roll in poses:
        res = solve_task(chain, x, y, z, pitch, roll)
        T_ik = chain.fk(res.q)
        _, R_want = task_to_pose(np.cos(res.yaw), np.sin(res.yaw), 0.0, pitch, roll)  # pitch, roll at final yaw
        pos_mm = np.linalg.norm(T_ik[:3, 3] - (x, y, z)) * 1000
        rot_deg = np.degrees(np.linalg.norm(rotation_error(R_want, T_ik[:3, :3])))
        yaw_off = np.degrees(abs(np.angle(np.exp(1j * (res.yaw - np.arctan2(y, x))))))
        if not (res.ok and pos_mm < 0.3 and rot_deg < 0.5 and yaw_off <= 20.1):
            failures.append(((x, y, z, pitch, roll), pos_mm, rot_deg, yaw_off))
    # Known edge: near-top-down grasps with the wrist rolled ~180° close to the base can need more yaw
    # than YAW_RANGE allows. Allow 2% so a regression still fails loudly.
    assert len(failures) <= 3, f"{len(failures)} of 150 failed, e.g. {failures[:3]}"


def test_task_pose_conversion_round_trips():
    p, R = task_to_pose(0.25, -0.1, 0.05, 0.6, -1.2)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    assert np.allclose(pose_to_task(T), (0.25, -0.1, 0.05, 0.6, -1.2))


def test_positive_pitch_points_down():
    _, R = task_to_pose(0.2, 0.0, 0.1, np.radians(90), 0.0)
    assert np.allclose(R[:, 0], [0, 0, -1], atol=1e-9)


def test_rotation_error_near_180_degrees():
    R = rot_z(np.pi - 1e-6)
    err = rotation_error(R, np.eye(3))
    assert np.isclose(np.linalg.norm(err), np.pi, atol=1e-4)
    assert np.allclose(np.abs(err / np.linalg.norm(err)), [0, 0, 1], atol=1e-3)


URDF = """<robot name="toy">
  <link name="base"/><link name="a"/><link name="b"/><link name="tip"/><link name="jaw"/>
  <joint name="pan" type="revolute"><parent link="base"/><child link="a"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-2" upper="2"/></joint>
  <joint name="lift" type="revolute"><parent link="a"/><child link="b"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/><axis xyz="0 1 0"/><limit lower="-1.5" upper="1.5"/></joint>
  <joint name="mount" type="fixed"><parent link="b"/><child link="tip"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/></joint>
  <joint name="jaw" type="revolute"><parent link="tip"/><child link="jaw"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="0" upper="1"/></joint>
</robot>"""


def test_urdf_parsing_skips_joints_past_the_tip(tmp_path):
    path = tmp_path / "toy.urdf"
    path.write_text(URDF)
    toy = Chain.from_urdf(path, "base", "tip")
    assert toy.names == ["pan", "lift"]
    T = toy.fk([np.pi / 2, 0.0])
    assert np.allclose(T[:3, 3], [0, 0.2, 0.15])
    T = toy.fk([0.0, np.pi / 2])  # positive rotation about y swings +x down to -z
    assert np.allclose(T[:3, 3], [0, 0, -0.05])
