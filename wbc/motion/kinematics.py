"""Forward and inverse kinematics for the SO-101: 5 arm joints, with the gripper jaw left out of the chain.

The chain comes from the SO-101 URDF, so link lengths match the real arm. Units inside are metres and
radians; the skills layer converts to mm and degrees.

Grasp targets are (x, y, z, pitch, roll). The tool's approach axis is its x-axis after the `tool` offset
in config/robot.yaml, and positive pitch points the tool down. A 5-DOF arm can't choose its yaw: it
starts at the target's bearing from the base, atan2(y, x), and the solver adjusts it slightly, because
the point between the jaws sits a few mm off the plane the pan joint sweeps.

Poses behind the base need the arm folded back over itself, and random starts rarely find that branch.
The bench is in front of the arm, so pass q0 (the current joint angles) when you need continuity.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def transform(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    """URDF origin: rotate by fixed-axis roll, pitch, yaw, then translate."""
    T = np.eye(4)
    T[:3, :3] = rot_z(rpy[2]) @ rot_y(rpy[1]) @ rot_x(rpy[0])
    T[:3, 3] = xyz
    return T


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.asarray(axis, float)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def rotation_error(R_target: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotation vector, in the world frame, that turns R into R_target."""
    E = R_target @ R.T
    theta = np.arccos(np.clip((np.trace(E) - 1) / 2, -1.0, 1.0))
    v = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    if theta < 1e-7:
        return v / 2
    if np.pi - theta < 1e-4:  # near 180°, sin(theta) is ~0; read the axis off the symmetric part
        A = (E + np.eye(3)) / 2
        axis = A[:, int(np.argmax(np.diag(A)))]
        return axis / np.linalg.norm(axis) * theta
    return v * theta / (2 * np.sin(theta))


@dataclass
class Joint:
    name: str
    origin: np.ndarray   # 4x4: parent link frame to joint frame at q = 0
    axis: np.ndarray     # unit axis in the joint frame (unused for fixed joints)
    lower: float = -np.pi
    upper: float = np.pi
    revolute: bool = True


class Chain:
    def __init__(self, joints: list, tool: np.ndarray | None = None):
        self.joints = joints
        self.tool = np.eye(4) if tool is None else tool
        active = [j for j in joints if j.revolute]
        self.names = [j.name for j in active]
        self.lower = np.array([j.lower for j in active])
        self.upper = np.array([j.upper for j in active])

    @property
    def n(self) -> int:
        return len(self.names)

    @classmethod
    def from_urdf(cls, path, base_link: str, tip_link: str, tool_xyz=(0, 0, 0), tool_rpy=(0, 0, 0)) -> "Chain":
        root = ET.parse(path).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        path_joints, link = [], tip_link
        while link != base_link:
            j = by_child.get(link)
            if j is None:
                raise ValueError(f"no joint path from {base_link} to {tip_link}")
            path_joints.append(j)
            link = j.find("parent").get("link")
        joints = []
        for j in reversed(path_joints):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            kind = j.get("type")
            if kind == "fixed":
                joints.append(Joint(j.get("name"), transform(xyz, rpy), np.zeros(3), 0.0, 0.0, revolute=False))
                continue
            if kind not in ("revolute", "continuous"):
                raise ValueError(f"joint {j.get('name')}: unsupported type {kind}")
            ax = j.find("axis")
            axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()])
            lim = j.find("limit")
            lower, upper = -np.pi, np.pi
            if kind == "revolute" and lim is not None:
                lower, upper = float(lim.get("lower", -np.pi)), float(lim.get("upper", np.pi))
            joints.append(Joint(j.get("name"), transform(xyz, rpy), axis / np.linalg.norm(axis), lower, upper))
        return cls(joints, transform(tool_xyz, tool_rpy))

    def frames(self, q) -> tuple:
        """Tool pose, plus the world position and axis of every active joint."""
        q = np.asarray(q, float)
        T, axes, k = np.eye(4), [], 0
        for j in self.joints:
            T = T @ j.origin
            if j.revolute:
                axes.append((T[:3, 3].copy(), T[:3, :3] @ j.axis))
                R = np.eye(4)
                R[:3, :3] = axis_rotation(j.axis, q[k])
                T = T @ R
                k += 1
        return T @ self.tool, axes

    def fk(self, q) -> np.ndarray:
        return self.frames(q)[0]

    def jacobian(self, q) -> tuple:
        """Geometric Jacobian (linear rows, then angular rows) and the tool pose."""
        T, axes = self.frames(q)
        J = np.zeros((6, self.n))
        for i, (p, z) in enumerate(axes):
            J[:3, i] = np.cross(z, T[:3, 3] - p)
            J[3:, i] = z
        return J, T


@dataclass
class IKResult:
    q: np.ndarray
    ok: bool
    pos_err: float   # metres
    rot_err: float   # radians
    iters: int
    yaw: float | None = None  # final tool yaw when it was solved for (task-frame targets)


YAW_COST = 0.5  # < 1 makes yaw "expensive", so the solver moves joints first and yaw only as needed
YAW_RANGE = 0.35  # rad (~20°) either side of the bearing. On the real URDF the jaw point moves ±8 mm
                  # sideways with wrist roll, so typical grasps turn ~5°. Near a top-down pose, turning yaw
                  # turns the jaws, so the grasp skill should read IKResult.yaw and correct roll if it matters.


def _dls(chain: Chain, p, R, q, w_rot, lam, iters, tol_pos, tol_rot, yaw_task=None) -> IKResult:
    """Damped least squares. yaw_task = (yaw0, pitch, roll) also solves for yaw, fixing only pitch and roll."""
    W = np.diag([1, 1, 1, w_rot, w_rot, w_rot])
    yaw = None
    if yaw_task is not None:
        yaw, pitch, roll = yaw_task
        yaw_lo, yaw_hi = yaw - YAW_RANGE, yaw + YAW_RANGE
        yaw_col =np.array([0, 0, 0, 0, 0, -w_rot * YAW_COST])  # turning the target about world z
    for k in range(iters):
        if yaw is not None:
            R = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
        J, T = chain.jacobian(q)
        e_p = p - T[:3, 3]
        e_r = rotation_error(R, T[:3, :3])
        pos_err, rot_err = np.linalg.norm(e_p), np.linalg.norm(e_r)
        if pos_err < tol_pos and rot_err < tol_rot:
            return IKResult(q, True, pos_err, rot_err, k, yaw)
        Jw, e = W @ J, np.concatenate([e_p, w_rot * e_r])
        if yaw is not None:
            Jw = np.column_stack([Jw, yaw_col])
        dx = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam ** 2 * np.eye(6), e)
        dq = dx[:chain.n]
        step = np.linalg.norm(dq)
        if step > 0.3:
            dq *= 0.3 / step
        q = np.clip(q + dq, chain.lower, chain.upper)
        if yaw is not None:
            yaw = float(np.clip(yaw + YAW_COST * dx[chain.n], yaw_lo, yaw_hi))
    return IKResult(q, False, pos_err, rot_err, iters, yaw)


def solve_ik(chain: Chain, p_target, R_target, q0=None, *, w_rot: float = 0.2, lam: float = 0.01,
             iters: int = 150, tol_pos: float = 2e-4, tol_rot: float = np.radians(0.5),
             seeds: int = 8, rng: np.random.Generator | None = None, yaw_task=None) -> IKResult:
    """Damped least squares from q0, then from mid-range, then from random starts within the limits.

    w_rot weighs orientation against position: 0.2 m/rad makes 1° count like 3.5 mm.
    Defaults stop at 0.2 mm and 0.5°.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    p_target, R_target = np.asarray(p_target, float), np.asarray(R_target, float)
    starts = ([np.asarray(q0, float)] if q0 is not None else []) + [(chain.lower + chain.upper) / 2]
    while len(starts) < seeds:
        starts.append(rng.uniform(chain.lower, chain.upper))
    best = None
    for q in starts:
        res = _dls(chain, p_target, R_target, np.clip(q, chain.lower, chain.upper),
                   w_rot, lam, iters, tol_pos, tol_rot, yaw_task)
        if res.ok:
            return res
        if best is None or res.pos_err + w_rot * res.rot_err < best.pos_err + w_rot * best.rot_err:
            best = res
    return best


def task_to_pose(x: float, y: float, z: float, pitch: float, roll: float) -> tuple:
    R = rot_z(np.arctan2(y, x)) @ rot_y(pitch) @ rot_x(roll)
    return np.array([x, y, z]), R


def pose_to_task(T: np.ndarray) -> tuple:
    R, p = T[:3, :3], T[:3, 3]
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    return float(p[0]), float(p[1]), float(p[2]), float(pitch), float(roll)


def solve_task(chain: Chain, x: float, y: float, z: float, pitch: float, roll: float, **kwargs) -> IKResult:
    p, R = task_to_pose(x, y, z, pitch, roll)
    return solve_ik(chain, p, R, yaw_task=(np.arctan2(y, x), pitch, roll), **kwargs)


def load_chain(robot_cfg: dict, root) -> Chain:
    tool = robot_cfg.get("tool", {})
    return Chain.from_urdf(root / robot_cfg["urdf"], robot_cfg["base_link"], robot_cfg["tip_link"],
                           tool.get("xyz", (0, 0, 0)), tool.get("rpy", (0, 0, 0)))
