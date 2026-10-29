"""Check the arm's joint map by hand, with the motors off: which way each joint's URDF angle goes as you turn it.

wbc/motion/joints.py turns raw servo ticks into the URDF's joint angles, q = sign * (raw - 2047) * 2 pi / 4096, with
each joint's sign in config/robot.yaml. All five are +1: on Oct 28 each joint was turned by hand in a pose where the
move reads clearly, and each one turned the URDF's positive way as its raw position rose. The answers first_moves.py
recorded on Oct 27 had the base, wrist flex and wrist roll the other way round, because they were read from the folded
rest pose, where the arm's bulk sits behind the pan axis and the gripper points up and back. This repeats the Oct 28
test, so run it again after a recalibration or a servo swap.

Nothing is written to the servos: it only sends ping and read requests, through wbc/motion/sts.py, never LeRobot. It
reads each servo's position, whether its torque is on, and the position limits lerobot-calibrate wrote into it, which
are the calibrated range, so it needs no calibration file. The motors have to be off, so the arm is limp: support it
as you move it. It won't run with any motor on.

    python scripts/check_joint_map.py --port /dev/cu.usbmodem5B910297421 once
    python scripts/check_joint_map.py --port /dev/cu.usbmodem5B910297421 watch
    python scripts/check_joint_map.py --port /dev/cu.usbmodem5B910297421 check

once prints each joint's raw position, LeRobot's degrees and the URDF's, and where the tool point is. watch keeps
printing the URDF angles and the gripper's opening on one line, about 5 times a second, until Ctrl-C. check takes you
through five moves and the L pose, waiting for Enter at each step (s = skip, q = quit), and passes a move when it
turned its joint's URDF angle the way the URDF says that move goes. It passes only when every move and the L pose were
checked and right. --calibration PATH also compares a LeRobot calibration file with the servos' own limits. check
writes .wbc/logs/joint-map-check-<time>.json. It needs the arm extras (pip install -e '.[arm]'). Close any other
program using the port first.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wbc.config import ROOT, load_yaml  # noqa: E402
from wbc.motion import sts  # noqa: E402
from wbc.motion.joints import ARM, JOINTS, NAMES, TICKS_PER_TURN, JointMap, load_calibration  # noqa: E402
from wbc.motion.kinematics import load_chain, pose_to_task  # noqa: E402

# The moves check asks for, worded as they worked on Oct 28, and which way each one turns its joint's URDF angle. That
# way is a fact about the URDF, not this arm: its pan axis points down, so the base turning counterclockwise seen from
# above is minus; the upper arm tipping forward, the elbow folding and the gripper tipping down are plus; and its roll
# axis points from the jaw back toward the wrist, so rolling clockwise looking from the wrist toward the jaw tips is
# minus. tests/test_check_joint_map.py proves each one with forward kinematics.
MOVES = [
    ("shoulder_pan", "BASE. Put the arm anywhere comfortable, resting or held still.",
     "Turn the base about 30° COUNTERCLOCKWISE, as seen from above (looking down at the table).", -1),
    ("shoulder_lift", "SHOULDER. Turn the base so the arm faces straight out the way it reaches, and hold the upper "
                      "arm about upright.",
     "Tip the upper arm about 20° FORWARD, toward where the arm reaches.", 1),
    ("elbow_flex", "ELBOW. Hold the upper arm still, with the forearm sticking out roughly level.",
     "FOLD the elbow about 20°, closing the angle between the forearm and the upper arm.", 1),
    ("wrist_flex", "WRIST FLEX. Hold the forearm about level, pointing out in front, and straighten the wrist so the "
                   "gripper points straight out along the forearm.",
     "Keeping the forearm level, tip the gripper about 30° DOWN.", 1),
    ("wrist_roll", "WRIST ROLL. Keep the forearm level and the gripper pointing straight out along it.",
     "Look along the gripper from the wrist end toward the jaw tips, and roll it about 45° CLOCKWISE.", -1),
]
L_POSE = ("L POSE. Hold the arm in an L: the base facing straight forward, the upper arm straight up, the forearm "
          "level and pointing forward, and the gripper straight out along the forearm.")
L_TOLERANCE = 8.0  # degrees the shoulder lift, elbow and wrist flex may read from 0 in the L pose: within 2° on Oct 28
MIN_MOVE = 5.0     # degrees a move has to turn its joint to count
READS = 5          # reads of each servo per position, whose median is kept: a single reading can glitch
EVERY = 0.2        # seconds between watch's lines


class Stopped(Exception):
    """You chose to finish early."""


def listed(joints) -> str:
    return ", ".join(NAMES[m] for m in joints)


def choose(ask, prompt: str) -> str:
    """Ask, and return "s" for skip or "" for Enter. Raises Stopped on q."""
    reply = ask(prompt).strip().lower()
    if reply.startswith("q"):
        raise Stopped
    return "s" if reply.startswith("s") else ""


def wrapped(ticks: int) -> int:
    """A change in raw position the short way round, so a joint turned across the encoder's seam reads right."""
    return (ticks + TICKS_PER_TURN // 2) % TICKS_PER_TURN - TICKS_PER_TURN // 2


def degrees_turned(joint_map: JointMap, joint: str, before: int, after: int) -> float:
    """How far a joint's URDF angle went from one raw reading to another, in degrees."""
    return joint_map.servos[joint].sign * wrapped(after - before) * 360 / TICKS_PER_TURN


def servo_ids(robot_cfg: dict) -> dict:
    """{joint: servo id} from config/robot.yaml. ValueError naming the joint if one has none."""
    ids = {}
    for m in JOINTS:
        servo_id = ((robot_cfg.get("servos") or {}).get(m) or {}).get("id")
        if not isinstance(servo_id, int):
            raise ValueError(f"the {NAMES[m]} needs a servo id in config/robot.yaml")
        ids[m] = servo_id
    return ids


def read_positions(bus, ids: dict, reads: int = 1) -> dict:
    """Every servo's Present_Position in raw ticks, the median of `reads` reads each. STSError naming the joint when a
    servo doesn't answer."""
    positions = {}
    for m, i in ids.items():
        try:
            positions[m] = int(statistics.median(bus.read_u16(i, sts.FEEDBACK_START) for _ in range(reads)))
        except sts.STSError as e:
            raise sts.STSError(f"the {NAMES[m]} (servo {i}): {e}") from e
    return positions


def read_servos(bus, ids: dict) -> dict:
    """Whether each servo's torque is on, and the position limits lerobot-calibrate wrote into it, as {joint: {torque,
    range_min, range_max}}."""
    return {m: {"torque": bus.read_u8(i, sts.TORQUE_ENABLE), "range_min": bus.read_u16(i, sts.MIN_POSITION_LIMIT),
                "range_max": bus.read_u16(i, sts.MAX_POSITION_LIMIT)} for m, i in ids.items()}


def compare(calibration: dict, servos: dict) -> list:
    """Where a LeRobot calibration file's ranges differ from the servos' own limits, in plain sentences."""
    lines = []
    for m in JOINTS:
        cal = calibration.get(m)
        if not cal:
            lines.append(f"The calibration file has nothing for the {NAMES[m]}.")
            continue
        file_range = (cal.get("range_min"), cal.get("range_max"))
        servo_range = (servos[m]["range_min"], servos[m]["range_max"])
        if file_range != servo_range:
            lines.append(f"The {NAMES[m]}'s range is {file_range[0]}–{file_range[1]} in the calibration file but "
                         f"{servo_range[0]}–{servo_range[1]} in the servo.")
    if lines:
        lines.append("The servos' limits are what the arm obeys, and LeRobot's calibration check, which the arm driver "
                     "runs before anything moves, fails until the file matches them.")
    return lines


def tool_point(joint_map: JointMap, chain, raw: dict) -> str:
    """Where the tool point is, which way it is from the base, and where the gripper points, in a sentence."""
    T, axes = chain.frames(joint_map.q(raw))
    x, y, z, pitch, _ = pose_to_task(T)
    pan = axes[0][0]
    bearing = math.degrees(math.atan2(y - pan[1], x - pan[0]))  # positive toward +y, the arm's left
    side = ("straight ahead" if abs(bearing) < 0.5 else
            f"{abs(bearing):.0f}° {'left' if bearing > 0 else 'right'} of straight ahead")
    tilt = math.degrees(pitch)  # positive points the gripper down
    points = "level" if abs(tilt) < 0.5 else f"{abs(tilt):.0f}° {'down' if tilt > 0 else 'up'}"
    return (f"The tool point is at x {1000 * x:.0f}, y {1000 * y:.0f}, z {1000 * z:.0f} mm, {side} of the base, with "
            f"the gripper pointing {points}.")


def table(joint_map: JointMap, chain, raw: dict) -> list:
    """Each joint's raw position, LeRobot's reading and its URDF angle, and whether it's inside where moves may take
    it, then where the tool point is."""
    past = joint_map.past_limits(raw)
    lines = [f"  {'joint':<14}{'raw':>6}{'LeRobot':>10}{'URDF':>9}   range"]
    for m in JOINTS:
        if m == "gripper":
            lerobot, urdf = f"{joint_map.gripper_percent(raw[m]):.1f}%", ""
        else:
            lerobot = f"{joint_map.lerobot_degrees(m, raw[m]):+.1f}°"
            urdf = f"{math.degrees(joint_map.angle(m, raw[m])):+.1f}°"
        where = f"past its end by {past[m]:.1f}°" if m in past else "inside"
        lines.append(f"  {NAMES[m]:<14}{raw[m]:>6}{lerobot:>10}{urdf:>9}   {where}")
    return lines + ["  " + tool_point(joint_map, chain, raw)]


def watch(bus, ids: dict, joint_map: JointMap, say=print, sleep=time.sleep, clock=time.monotonic,
          seconds: float | None = None) -> None:
    """The URDF angles and the gripper's opening on one line, rewritten about every EVERY seconds, until Ctrl-C (or
    for `seconds`)."""
    t0 = clock()
    try:
        while seconds is None or clock() - t0 < seconds:
            raw = read_positions(bus, ids)
            angles = "  ".join(f"{NAMES[m]} {math.degrees(joint_map.angle(m, raw[m])):+6.1f}°" for m in ARM)
            past = joint_map.past_limits(raw)
            warning = f"   PAST ITS END: {listed(past)}" if past else ""
            say(f"\r{angles}  gripper {joint_map.gripper_percent(raw['gripper']):5.1f}%{warning}   ", end="",
                flush=True)
            sleep(EVERY)
    except KeyboardInterrupt:  # how watch ends
        say("")


def try_move(bus, ids: dict, joint_map: JointMap, joint: str, setup: str, action: str, way: int, ask, say) -> dict:
    """One move of the guided test, asked for again while it turns its joint less than MIN_MOVE."""
    while True:
        say(f"\n{setup}")
        if choose(ask, "  When it's set up and still, press Enter (s = skip, q = quit): ") == "s":
            return {"joint": joint, "skipped": True}
        before = read_positions(bus, ids, READS)
        say(f"  {action}")
        choose(ask, "  Hold it there, then press Enter (q = quit): ")
        after = read_positions(bus, ids, READS)
        turned = {m: degrees_turned(joint_map, m, before[m], after[m]) for m in JOINTS}
        others = [f"the {NAMES[m]} {v:+.1f}°" for m, v in turned.items() if m != joint and abs(v) >= MIN_MOVE]
        if others:
            say("  Also moved: " + ", ".join(others) + ".")
        if abs(turned[joint]) < MIN_MOVE:
            say(f"  The {NAMES[joint]} turned only {turned[joint]:+.1f}°, under the {MIN_MOVE:.0f}° this needs. "
                "Once more, a little further.")
            continue
        ok = turned[joint] * way > 0
        expected = "minus" if way < 0 else "plus"
        say(f"  The {NAMES[joint]}'s URDF angle went {turned[joint]:+.1f}° (raw {before[joint]} to {after[joint]}): "
            + ("the way the URDF says that move goes." if ok else
               f"that move should go {expected}, so its sign in config/robot.yaml looks wrong."))
        return {"joint": joint, "before": before, "after": after, "turned": round(turned[joint], 2),
                "expected": expected, "ok": ok}


def try_l_pose(bus, ids: dict, joint_map: JointMap, ask, say) -> dict:
    """The L pose held by hand: the shoulder lift, elbow and wrist flex should read within L_TOLERANCE of 0."""
    say(f"\n{L_POSE}")
    if choose(ask, "  Press Enter when it's held there (s = skip, q = quit): ") == "s":
        return {"skipped": True}
    raw = read_positions(bus, ids, READS)
    angles = {m: round(math.degrees(joint_map.angle(m, raw[m])), 2) for m in ARM}
    lift, elbow, wrist = (angles[m] for m in ("shoulder_lift", "elbow_flex", "wrist_flex"))
    ok = max(abs(lift), abs(elbow), abs(wrist)) <= L_TOLERANCE  # the base and the roll can't be judged by eye
    say(f"  The shoulder lift reads {lift:+.1f}°, the elbow {elbow:+.1f}° and the wrist flex {wrist:+.1f}°"
        + (f": within {L_TOLERANCE:.0f}° of the URDF's zero, so its zero is the L pose." if ok else
           f": not all within {L_TOLERANCE:.0f}° of the URDF's zero. If the arm really was in the L, the calibration's "
           "zero is off."))
    say(f"  The base reads {angles['shoulder_pan']:+.1f}° and the wrist roll {angles['wrist_roll']:+.1f}°, which the "
        "eye can't judge.")
    return {"raw": raw, "urdf_degrees": angles, "ok": ok}


def verdict(record: dict) -> tuple[list, bool]:
    """The summary table and what it means, with every move whether it ran or not, and whether the check passed:
    every move and the L pose checked, and each one right."""
    moves = {r["joint"]: r for r in record.get("moves", [])}
    lines = [f"\n{'joint':<14}{'turned':>9}{'should go':>11}   result"]
    for joint, *_ in MOVES:
        r = moves.get(joint)
        if r is None or r.get("skipped"):
            lines.append(f"{NAMES[joint]:<14}{'':>9}{'':>11}   {'not run' if r is None else 'skipped'}")
        else:
            lines.append(f"{NAMES[joint]:<14}{r['turned']:>+8.1f}°{r['expected']:>11}   "
                         f"{'right' if r['ok'] else 'WRONG'}")
    l_pose = record.get("l_pose")
    lines.append(f"{'L pose':<34}   " + ("not run" if l_pose is None else "skipped" if l_pose.get("skipped") else
                                         "right" if l_pose["ok"] else "OFF"))
    wrong = [m for m, r in moves.items() if not r.get("skipped") and not r["ok"]]
    tried = [m for m, r in moves.items() if not r.get("skipped")]
    unchecked = [NAMES[m] for m, *_ in MOVES if m not in tried]
    if l_pose is None or l_pose.get("skipped"):
        unchecked.append("L pose")
    l_off = bool(l_pose) and not l_pose.get("skipped") and not l_pose["ok"]
    if wrong:
        lines.append(f"\nThe joint map doesn't match the arm: the {listed(wrong)} turned the wrong way. Check "
                     "that each move was the one asked for, then the signs in config/robot.yaml, before anything "
                     "moves the arm by this map.")
    elif l_off:
        lines.append(("\nEvery move tried turned the URDF's way, but" if tried else "\nNo move was checked, and")
                     + " the L pose doesn't read near the URDF's zero.")
    elif len(unchecked) == len(MOVES) + 1:
        lines.append("\nNothing was checked, so this isn't a pass.")
    elif unchecked:
        lines.append(f"\nEvery move tried turned the URDF's way, but this isn't a full check, so it isn't a pass. "
                     f"Not checked: {', '.join(unchecked)}.")
    else:
        lines.append("\nThe joint map matches the arm: every move turned its joint the way the URDF says, and the L "
                     "pose reads near the URDF's zero.")
    return lines, not wrong and not l_off and not unchecked


def run_check(bus, ids: dict, joint_map: JointMap, record: dict, ask=input, say=print) -> bool:
    """The guided test: each of MOVES, then the L pose. Fills record (moves, l_pose) as it goes, so it survives q,
    Ctrl-C or a servo going silent, and prints the summary whichever way it ends. True only if every move and the L
    pose were checked and right."""
    record.setdefault("moves", [])
    say("All six servos answer and every motor is off, so the arm is limp. Support it as you move it.")
    try:
        for joint, setup, action, way in MOVES:
            record["moves"].append(try_move(bus, ids, joint_map, joint, setup, action, way, ask, say))
        record["l_pose"] = try_l_pose(bus, ids, joint_map, ask, say)
    except (Stopped, KeyboardInterrupt, EOFError):
        say("\nStopped.")
    except sts.STSError as e:
        say(f"\nStopped: {e}.")
    lines, ok = verdict(record)
    for line in lines:
        say(line)
    return ok


def run(bus, mode: str, robot_cfg: dict, record: dict, calibration: dict | None = None, ask=input, say=print,
        sleep=time.sleep, clock=time.monotonic, seconds: float | None = None) -> bool:
    """One session on an open bus: check that all six servos answer with their motors off, build the joint map from
    config/robot.yaml and the servos' own limits, then the mode. Fills `record`. True if it ran, and for check, if
    nothing read the wrong way."""
    ids = servo_ids(robot_cfg)
    silent = [m for m, i in ids.items() if not bus.ping(i)]
    if silent:
        say(f"No answer from the {listed(silent)}. Check that the servo power is on and the cable chain reaches "
            "every servo.")
        return False
    servos = read_servos(bus, ids)
    on = [m for m in JOINTS if servos[m]["torque"]]
    if on:
        say(f"Torque is on for the {listed(on)}. This check needs the motors off, so don't force the arm. Switch the "
            "servo power off and on again (they come back with torque off), then run this again.")
        return False
    whole = [m for m in JOINTS if m != "wrist_roll" and (servos[m]["range_min"], servos[m]["range_max"]) ==
             (0, TICKS_PER_TURN - 1)]
    if whole:
        say(f"The position limits on the {listed(whole)} are 0–{TICKS_PER_TURN - 1}, the whole turn, as a servo comes "
            "from the factory. Only the wrist roll, which has no stop, should be calibrated that way, so run "
            "lerobot-calibrate (a new servo?), then this again.")
        return False
    record["limits"] = {m: [s["range_min"], s["range_max"]] for m, s in servos.items()}
    try:
        chain = load_chain(robot_cfg, ROOT)
        joint_map = JointMap.from_config(robot_cfg, servos, chain)
    except ValueError as e:
        say(f"The joint map can't be built: {e}.")
        return False
    if calibration is not None:
        for line in compare(calibration, servos):
            say(line)
    if mode == "once":
        for line in table(joint_map, chain, read_positions(bus, ids, READS)):
            say(line)
        return True
    if mode == "watch":
        say("Watching (read only), in URDF degrees. Ctrl-C to stop.")
        watch(bus, ids, joint_map, say=say, sleep=sleep, clock=clock, seconds=seconds)
        return True
    return run_check(bus, ids, joint_map, record, ask=ask, say=say)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=["once", "watch", "check"],
                    help="once: one reading. watch: keep reading until Ctrl-C. check: the guided test")
    ap.add_argument("--port", required=True, help="the arm's servo adapter, e.g. /dev/cu.usbmodem5B910297421")
    ap.add_argument("--calibration", type=Path, metavar="PATH",
                    help="a LeRobot calibration file to compare with the servos' own limits, e.g. "
                         "~/.cache/huggingface/lerobot/calibration/robots/so_follower/wbc_follower.json")
    args = ap.parse_args(argv)
    calibration = None
    if args.calibration:
        try:
            calibration = load_calibration(args.calibration.expanduser())
        except (OSError, ValueError) as e:
            ap.error(f"can't read {args.calibration}: {e}")
        if not isinstance(calibration, dict) or not all(isinstance(c, dict) for c in calibration.values()):
            ap.error(f"{args.calibration} isn't a LeRobot calibration file, which has a range_min and range_max "
                     "for each joint")
    try:
        robot_cfg = load_yaml("robot.yaml")
        servo_ids(robot_cfg)
    except ValueError as e:
        print(f"config/robot.yaml: {e}")
        return 1
    try:
        bus = sts.Bus(args.port)
    except Exception as e:  # noqa: BLE001
        print(f"Can't open {args.port}: {e}")
        return 1
    record = {"port": args.port, "mode": args.mode}
    ok = False
    try:
        ok = run(bus, args.mode, robot_cfg, record, calibration=calibration)
    except sts.STSError as e:
        print(f"\nStopped: {e}")
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    finally:
        bus.close()
        if record.get("moves"):
            path = ROOT / ".wbc" / "logs" / f"joint-map-check-{datetime.now():%Y%m%d-%H%M%S}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1), encoding="utf-8")
            print(f"log: {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
