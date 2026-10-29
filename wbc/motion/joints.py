"""The joint map: each servo's raw position as the URDF's joint angle, and back.

lerobot-calibrate's Part 1 sets every servo's homing offset so its joint reads raw 2047 in the pose the arm is held
in, and that pose is the L: base facing forward, upper arm straight up, forearm and gripper level pointing forward.
That is so101_new_calib.urdf's q = 0 (forward kinematics there puts the upper arm up and the forearm and gripper
level along +x). So with 4096 ticks to a turn, a joint's URDF angle in radians is

    q = sign * (raw - 2047) * 2 pi / 4096

with each joint's sign in config/robot.yaml. All five are +1. On Oct 28 each joint was turned by hand with the motors
off, in a pose where the move reads clearly, and every time raw rose or fell as the URDF's angle does. The URDF's
positive way is the base clockwise seen from above (its pan axis points down), the shoulder lift forward, the elbow
folding, the wrist flex tipping the gripper down, and the wrist roll counterclockwise looking from the wrist toward
the jaw tips (its axis points from the jaw back toward the wrist). The L pose held by hand read within 2° of 0 on the
shoulder lift, elbow and wrist flex. That agrees with LeRobot's own RobotKinematics, which feeds LeRobot's degrees
straight into this URDF, and with the URDF's CAD, where every joint sits on its STS3215 the same way.
scripts/check_joint_map.py repeats the test. first_moves.py's answers on Oct 27 had the base, wrist flex and wrist
roll the other way round: they were read from the folded rest pose, where the arm's bulk sits behind the pan axis
and the gripper points up and back.

LeRobot's degrees aren't these. It puts 0 at the middle of each joint's calibrated range and divides by 4095, so at
raw 2047 its base reads -11.9°; lerobot_degrees and gripper_percent give its numbers, for comparison. The calibrated
ranges come from LeRobot's calibration file, and lerobot-calibrate also wrote them into each servo as its position
limits, so a joint can't be driven past either end. The real arm's stops go past the URDF's limits (at rest the
shoulder lift reads -111° against the URDF's -100°), so a joint's limits here are its calibrated range. The wrist
roll has no stop, so its range is the whole turn and its limits are the URDF's: its cable is the limit.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]  # servo ids 1-6
ARM = JOINTS[:5]  # the IK chain, in config/robot.yaml's order
NAMES = {"shoulder_pan": "base", "shoulder_lift": "shoulder lift", "elbow_flex": "elbow",
         "wrist_flex": "wrist flex", "wrist_roll": "wrist roll", "gripper": "gripper"}
CALIBRATION_POSE = 2047  # raw ticks every joint reads in lerobot-calibrate's Part 1 L pose: the URDF's q = 0
TICKS_PER_TURN = 4096    # raw ticks per turn of an STS3215's output shaft
LEROBOT_RES = 4095       # ticks LeRobot counts to 360° in its degrees: the STS3215's 4096 minus 1
CALIBRATION_ID = "wbc_follower"  # the --robot.id this arm was calibrated under (Oct 27): its file is wbc_follower.json


@dataclass(frozen=True)
class Servo:
    name: str
    id: int
    sign: int = 1          # +1: raw rising turns the joint the URDF's positive way
    range_min: int = 0     # raw ticks: the calibrated range, also written into the servo as its position limits
    range_max: int = 4095

    @property
    def full_turn(self) -> bool:
        """True for a joint calibrated over the whole turn because it has no stop (the wrist roll)."""
        return self.range_min == 0 and self.range_max == TICKS_PER_TURN - 1


def value_of(entry, key: str, default=None):
    """One value of a calibration entry, from LeRobot's JSON (a dict) or its MotorCalibration (attributes)."""
    return entry.get(key, default) if isinstance(entry, dict) else getattr(entry, key, default)


class JointMap:
    def __init__(self, servos: dict, urdf_lower=None, urdf_upper=None):
        """servos: {joint: Servo} for all six. urdf_lower and urdf_upper: the URDF's limits in radians, in ARM
        order, used only for a joint with no stop."""
        missing = [NAMES[m] for m in JOINTS if m not in servos]
        if missing:
            raise ValueError(f"the joint map needs all six servos, and has none for the {', '.join(missing)}")
        self.servos = dict(servos)
        self.urdf_lower = None if urdf_lower is None else np.asarray(urdf_lower, float)
        self.urdf_upper = None if urdf_upper is None else np.asarray(urdf_upper, float)

    @classmethod
    def from_config(cls, robot_cfg: dict, calibration: dict, chain=None) -> JointMap:
        """The map for this arm: ids and signs from config/robot.yaml's servos, ranges from LeRobot's calibration
        (a dict per joint as in its file, or its MotorCalibration records), and from `chain` the URDF's limits for
        a joint with no stop. The calibration's ids and drive modes are checked where it has them. Raises
        ValueError, naming the joint, for anything missing or that doesn't add up."""
        listed = robot_cfg.get("servos") or {}
        servos = {}
        for m in JOINTS:
            name = NAMES[m]
            if m not in listed:
                raise ValueError(f"the {name} is missing from the servos in config/robot.yaml")
            cfg = listed[m] or {}
            servo_id = cfg.get("id")
            if not isinstance(servo_id, int):
                raise ValueError(f"the {name} needs a servo id (a whole number) in config/robot.yaml, "
                                 f"not {servo_id!r}")
            sign = cfg.get("sign", 1 if m == "gripper" else None)  # the gripper isn't in the IK chain
            if sign not in (1, -1):
                raise ValueError(f"the {name} needs a sign of 1 or -1 in config/robot.yaml, not {sign!r}")
            if m not in calibration:
                raise ValueError(f"the {name} is missing from the calibration")
            cal = calibration[m]
            cal_id = value_of(cal, "id")
            if cal_id is not None and cal_id != servo_id:
                raise ValueError(f"the {name} is servo {servo_id} in config/robot.yaml but servo {cal_id} in the "
                                 "calibration")
            drive_mode = value_of(cal, "drive_mode", 0)
            if drive_mode != 0:
                raise ValueError(f"the {name}'s calibration has drive_mode {drive_mode}, which flips its direction "
                                 "in LeRobot, and this map assumes it isn't flipped")
            low, high = value_of(cal, "range_min"), value_of(cal, "range_max")
            if low is None or high is None:
                raise ValueError(f"the calibration has no range for the {name}")
            if not low < high:
                raise ValueError(f"the {name}'s calibrated range runs from {low} to {high}, and its minimum has to "
                                 "be below its maximum")
            servos[m] = Servo(m, servo_id, int(sign), int(low), int(high))
        if chain is None:
            return cls(servos)
        if list(chain.names) != ARM:
            raise ValueError(f"the chain's joints are {', '.join(chain.names)}, not the arm's {', '.join(ARM)}")
        return cls(servos, chain.lower, chain.upper)

    def angle(self, joint: str, raw: float) -> float:
        """The joint's URDF angle in radians at a raw position in ticks."""
        return self.servos[joint].sign * (raw - CALIBRATION_POSE) * 2 * math.pi / TICKS_PER_TURN

    def ticks(self, joint: str, angle: float) -> int:
        """The raw position for a URDF angle in radians, rounded to the nearest tick."""
        return int(round(CALIBRATION_POSE + self.servos[joint].sign * angle * TICKS_PER_TURN / (2 * math.pi)))

    def q(self, raw: dict) -> np.ndarray:
        """The five arm joints' URDF angles in radians, in ARM order, from raw positions {joint: ticks}."""
        return np.array([self.angle(m, raw[m]) for m in ARM])

    def raw(self, q) -> dict:
        """Raw positions {joint: ticks} for the five arm joints at URDF angles q (radians, in ARM order)."""
        q = np.asarray(q, float)
        if q.shape != (len(ARM),):
            raise ValueError(f"q needs one angle for each of the {len(ARM)} arm joints, not shape {q.shape}")
        return {m: self.ticks(m, angle) for m, angle in zip(ARM, q)}

    def lerobot_degrees(self, joint: str, raw: float) -> float:
        """What LeRobot 0.6.1 reports for an arm joint in degrees: 0 at the middle of its calibrated range and 4095
        ticks to 360°, not clamped to the range, and drive_mode left out."""
        s = self.servos[joint]
        mid = (s.range_min + s.range_max) / 2
        return (raw - mid) * 360 / LEROBOT_RES

    def gripper_percent(self, raw: float) -> float:
        """What LeRobot reports for the gripper at a raw position: clamped to its calibrated range, then 0 at
        range_min (closed, on this arm) to 100 at range_max."""
        s = self.servos["gripper"]
        bounded = min(s.range_max, max(s.range_min, raw))
        return ((bounded - s.range_min) / (s.range_max - s.range_min)) * 100

    def limits(self, joint: str) -> tuple[float, float]:
        """The joint's calibrated range as URDF angles (low, high) in radians, low first whatever its sign. A joint
        with no stop gets the URDF's limits instead (its cable is the limit), or ±pi without them."""
        s = self.servos[joint]
        if s.full_turn:
            if joint in ARM and self.urdf_lower is not None and self.urdf_upper is not None:
                i = ARM.index(joint)
                return float(self.urdf_lower[i]), float(self.urdf_upper[i])
            return -math.pi, math.pi
        ends = sorted((self.angle(joint, s.range_min), self.angle(joint, s.range_max)))
        return ends[0], ends[1]

    def outside(self, raw: dict) -> dict:
        """Every joint in `raw` (any of the six) that is past an end of its calibrated range, as {joint: degrees
        past the nearer end}. Empty when they're all inside."""
        past = {}
        for m in JOINTS:
            if m not in raw:
                continue
            s = self.servos[m]
            if not s.range_min <= raw[m] <= s.range_max:
                past[m] = max(s.range_min - raw[m], raw[m] - s.range_max) * 360 / TICKS_PER_TURN
        return past


def load_calibration(path) -> dict:
    """LeRobot's calibration file as lerobot-calibrate wrote it: {joint: {id, drive_mode, homing_offset, range_min,
    range_max}}, in raw ticks."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def calibration_dir() -> Path:
    """The folder LeRobot 0.6.1 reads an SO-101 follower's calibration from, for whoever runs this: robots/so_follower
    under $HF_LEROBOT_CALIBRATION, else $HF_LEROBOT_HOME/calibration, else $HF_HOME/lerobot/calibration, where HF_HOME
    defaults to $XDG_CACHE_HOME/huggingface or ~/.cache/huggingface. Each macOS account has its own, so on another
    account than the one that ran lerobot-calibrate, point at that one's folder instead."""
    cache = os.getenv("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
    hf_home = os.path.expandvars(os.path.expanduser(os.getenv("HF_HOME", os.path.join(cache, "huggingface"))))
    lerobot_home = Path(os.getenv("HF_LEROBOT_HOME", Path(hf_home) / "lerobot")).expanduser()
    calibration = Path(os.getenv("HF_LEROBOT_CALIBRATION", lerobot_home / "calibration")).expanduser()
    return calibration / "robots" / "so_follower"
