"""Joint-space moves, timed by a quintic so each one starts and ends at rest.

A move takes every joint along one straight line in joint space, from q0 to q1, so they all start and finish
together. Its pace is the quintic 10s³ - 15s⁴ + 6s⁵, with s the share of the move's time gone, 0 to 1: zero
velocity and zero acceleration at both ends, as the plan says, so the arm isn't jerked as a move starts or stops.
first_moves.py's half cosine starts at rest too, but its acceleration jumps from 0 the moment it starts. The quintic
pays for that with a higher top speed: halfway through it moves at 15/8 times its average speed, against the half
cosine's pi/2.

min_seconds sizes a move by that peak, keeping every joint under a speed cap and the tool point under another, for
moves near the bench. lowest says how low a path takes the tool point and the joint axes, so the driver can refuse a
move that would hit the table.
"""
from __future__ import annotations

import math

import numpy as np

PEAK = 15 / 8  # a quintic's top speed as a multiple of its average, reached halfway through


def quintic(s):
    """How far along a move is, 0 to 1, once a share s of its time has gone: 10s³ - 15s⁴ + 6s⁵. Takes arrays."""
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def joint_path(q0, q1, seconds: float, hz: float) -> np.ndarray:
    """The goals for a move from q0 to q1, one row of joint angles per tick: shape (n, len(q0)), with n = seconds
    times hz rounded up, at least 1, so the move never runs faster than `seconds` asks. Row k is quintic(k / n) of
    the way, so q0 itself isn't a row and the last row is exactly q1."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    n = max(1, math.ceil(seconds * hz - 1e-9))  # the slack keeps 0.3 s at 50 Hz (15.000000000000002) at 15 ticks
    path = q0 + np.outer(quintic(np.arange(1, n + 1) / n), q1 - q0)
    path[-1] = q1
    return path


def min_seconds(q0, q1, max_joint_speed: float, chain=None, max_tip_speed: float | None = None,
                samples: int = 50) -> float:
    """Seconds for the shortest quintic move from q0 to q1 that keeps every joint at or under max_joint_speed
    (rad/s) and, given max_tip_speed and the chain, the tool point at or under max_tip_speed (m/s), at the move's
    peak. The tip's bound is its fastest spot along the joint-space line, from `samples` points, times PEAK. 0.0 for
    a move that goes nowhere; ValueError for a cap that isn't above 0, or a tip cap without the chain."""
    if max_joint_speed <= 0:
        raise ValueError(f"max_joint_speed has to be above 0, not {max_joint_speed}")
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    dq = q1 - q0
    seconds = float(np.max(np.abs(dq), initial=0.0)) * PEAK / max_joint_speed
    if max_tip_speed is None:
        return seconds
    if chain is None:
        raise ValueError("a tip speed needs the chain, to find the tool point")
    if max_tip_speed <= 0:
        raise ValueError(f"max_tip_speed has to be above 0, not {max_tip_speed}")
    if samples < 2:
        raise ValueError(f"the line needs at least 2 samples, both ends, not {samples}")
    # metres the tool point moves per unit of progress along the line, at each sample: the Jacobian's linear rows
    along = np.linspace(0, 1, samples)
    fastest = max(np.linalg.norm(chain.jacobian(q0 + a * dq)[0][:3] @ dq) for a in along)
    return max(seconds, float(fastest) * PEAK / max_tip_speed)


def lowest(chain, path, joints: tuple) -> tuple[float, float]:
    """Along a path (rows of joint angles in radians), the lowest the tool point goes and the lowest any of the named
    joints' axes goes, as heights in metres above the base's underside, which sits on the table (inf with no
    joints named)."""
    named = [chain.names.index(m) for m in joints]
    tool = axis = np.inf
    for q in np.atleast_2d(path):
        T, axes = chain.frames(q)  # axes: (origin, direction) per joint, in chain.names order
        tool = min(tool, T[2, 3])
        axis = min([axis] + [axes[i][0][2] for i in named])
    return float(tool), float(axis)
