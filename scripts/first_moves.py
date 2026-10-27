"""Move each SO-101 joint a little under power for the first time, one joint at a time.

The motors come on holding the arm exactly where it is. Then each joint in turn moves 10° toward the
middle of its range and back, slowly, and you say which way it went: that's the sign the IK needs for
each joint. Every move waits for Enter, and the motors only go off after you say you're holding the arm,
because it drops when they do.

    python scripts/first_moves.py --port /dev/cu.usbmodem5B910297421

It needs LeRobot (uv pip install -e '.[lerobot]') and a calibration from lerobot-calibrate under the same
--id. Ctrl-C stops the moves, and the arm holds where it is until you release it. Each run writes
.wbc/logs/first-moves-<time>.json. Close any other program using the port first.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wbc.config import ROOT  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
NAMES = {"shoulder_pan": "base", "shoulder_lift": "shoulder lift", "elbow_flex": "elbow",
         "wrist_flex": "wrist flex", "wrist_roll": "wrist roll", "gripper": "gripper"}
# Asked after each move, with its two answers. All but the shoulder lift's read the same in any pose;
# that one needs the base facing forward.
QUESTIONS = {
    "shoulder_pan": ("Seen from above, which way did the base turn?", ("clockwise", "counterclockwise")),
    "shoulder_lift": ("Which way did the upper arm tip?", ("forward", "back")),
    "elbow_flex": ("Did the elbow fold or open?", ("fold", "open")),
    "wrist_flex": ("Did the wrist curl in (the way the elbow folds) or out?", ("in", "out")),
    "wrist_roll": ("Looking from the elbow to the gripper, which way did the gripper roll?",
                   ("clockwise", "counterclockwise")),
    "gripper": ("Did the gripper open or close?", ("open", "close")),
}
CALIBRATION_POSE = 2047  # every joint reads this raw position in lerobot-calibrate's Part 1 pose
TICKS_PER_DEGREE = 4096 / 360
FORWARD = 30.0   # degrees the base may be off forward and "forward or back" still makes sense
MAX_JUMP = 3.0   # degrees any joint may shift as torque comes on
MAX_STEP = 5.0   # LeRobot's max_relative_target: no goal goes more than this far from where the joint is


class Stopped(Exception):
    """You chose to finish early."""


def unit(joint: str) -> str:
    return "" if joint == "gripper" else "°"


def ramp(start: float, end: float, seconds: float, hz: float) -> list:
    """Goals from start to end, one per tick, easing in and out along a half cosine."""
    n = max(1, round(seconds * hz))
    return [start + (end - start) * (1 - math.cos(math.pi * k / n)) / 2 for k in range(1, n + 1)]


def toward_middle(joint: str, position: float, step: float) -> float:
    """The signed step toward the middle of a joint's range. LeRobot puts 0° at the middle of each joint's
    calibrated range, and the gripper's middle is 50 on its 0-100 scale."""
    middle = 50.0 if joint == "gripper" else 0.0
    return step if position <= middle else -step


def plus_means(joint: str, step: float, answer: int) -> str:
    """What a positive LeRobot move does, from the answer (0 or 1) to a move of sign `step`."""
    options = QUESTIONS[joint][1]
    return options[answer if step > 0 else 1 - answer]


def choose(ask, prompt: str, keys: str) -> str:
    """Ask until the reply starts with one of `keys`."""
    while True:
        reply = ask(prompt).strip().lower()
        if reply and reply[0] in keys:
            return reply[0]


def power_on(robot, sleep=time.sleep) -> dict:
    """Turn torque on without the arm moving, and return how far each joint shifted, in degrees.

    A servo drives to whatever its goal register holds the moment torque comes on. After calibration
    changed the homing offsets and the arm was moved by hand, that can be far from where the arm is, and
    LeRobot's connect() turns torque on without setting it. So this runs LeRobot's own configure() with its
    final torque-on held back, sets each goal to the present position and reads it back, and only then
    turns torque on."""
    bus = robot.bus
    enable = bus.enable_torque
    bus.enable_torque = lambda *args, **kwargs: None  # configure() ends by turning torque on: not yet
    try:
        robot.configure()
    finally:
        del bus.enable_torque  # back to the bus's own method
    present = bus.sync_read("Present_Position", normalize=False, num_retry=2)
    for motor, value in present.items():
        bus.write("Goal_Position", motor, value, normalize=False, num_retry=2)
    goals = bus.sync_read("Goal_Position", normalize=False, num_retry=2)
    wrong = [m for m in present if goals[m] != present[m]]
    if wrong:
        raise RuntimeError(f"the hold-here goal didn't take on {', '.join(wrong)}, so torque stayed off")
    enable(num_retry=2)
    sleep(0.3)
    now = bus.sync_read("Present_Position", normalize=False, num_retry=2)
    return {m: abs(now[m] - present[m]) / TICKS_PER_DEGREE for m in present}


def move(robot, joint: str, start: float, end: float, seconds: float, hz: float, still: dict, log: list,
         settle: float = 0.5, sleep=time.sleep, clock=time.monotonic) -> tuple:
    """Drive one joint from start to end, then hold it there for `settle` seconds. Logs every reading and
    returns the last one (all six joints) and how far any other joint strayed from `still`."""
    period = 1.0 / hz
    goals = ramp(start, end, seconds, hz) + [end] * max(1, round(settle * hz))
    drift = 0.0
    t0 = clock()
    for k, goal in enumerate(goals, 1):
        sent = robot.send_action({f"{joint}.pos": goal})[f"{joint}.pos"]
        obs = robot.get_observation()
        now = {m: obs[f"{m}.pos"] for m in JOINTS}
        drift = max([drift] + [abs(now[m] - still[m]) for m in JOINTS if m != joint])
        log.append({"t": round(clock() - t0, 3), "joint": joint, "goal": round(sent, 2),
                    **{m: round(v, 2) for m, v in now.items()}})
        wait = t0 + k * period - clock()
        if wait > 0:
            sleep(wait)
    return now, drift


def try_joint(robot, joint: str, step: float, seconds: float, hz: float, log: list, ask, say,
              sleep=time.sleep, clock=time.monotonic) -> dict | None:
    """Move one joint toward the middle of its range, ask which way it went, and bring it back. Returns the
    result, or None if you skip it."""
    obs = robot.get_observation()
    still = {m: obs[f"{m}.pos"] for m in JOINTS}
    start = still[joint]
    delta = toward_middle(joint, start, step)
    target = start + delta
    u = unit(joint)
    scale = " on its 0-100 scale" if joint == "gripper" else ""
    reply = ask(f"\nNext: the {NAMES[joint]}. It moves {delta:+.0f}{u}{scale}, from {start:+.1f}{u} to "
                f"{target:+.1f}{u}, over {seconds:g} s, then holds while you answer. "
                "Enter = go, s = skip, q = finish: ").strip().lower()
    if reply.startswith("q"):
        raise Stopped
    if reply.startswith("s"):
        return None

    def leg(a, b):
        return move(robot, joint, a, b, seconds, hz, still, log, sleep=sleep, clock=clock)

    question, options = QUESTIONS[joint]
    drift = 0.0
    while True:
        reached, d = leg(start, target)
        drift = max(drift, d)
        moved = reached[joint] - start
        if moved * delta <= 0 or abs(moved) < 0.3 * step:
            robot.send_action({f"{joint}.pos": reached[joint]})  # stop pushing toward a goal it can't reach
            say(f"The {NAMES[joint]} didn't follow: asked {delta:+.1f}{u}, it moved {moved:+.1f}{u}. "
                "Stopping the moves here.")
            return {"joint": joint, "start": round(start, 2), "asked": delta, "moved": round(moved, 2),
                    "ok": False}
        answer = choose(ask, f"{question}  1 = {options[0]}, 2 = {options[1]}, r = show me again: ", "12r")
        if answer != "r":
            break
        reached, d = leg(target, start)  # back to the start, then out again
        drift = max(drift, d)
    returned, d = leg(target, start)
    drift = max(drift, d)

    result = {"joint": joint, "start": round(start, 2), "asked": delta, "moved": round(moved, 2),
              "back_off": round(abs(returned[joint] - start), 2), "others_moved": round(drift, 2),
              "answer": options[int(answer) - 1], "plus_means": plus_means(joint, delta, int(answer) - 1),
              "ok": True}
    notes = []
    if abs(moved - delta) > 2:
        notes.append(f"missed its target by {abs(moved - delta):.1f}{u}")
    if result["back_off"] > 2:
        notes.append(f"came back {result['back_off']:.1f}{u} off")
    if drift > 2:
        notes.append(f"other joints moved up to {drift:.1f}")
    say(f"  It moved {moved:+.1f}{u} and came back within {result['back_off']:.1f}{u}. "
        f"Plus is {result['plus_means']}." + (f" Note: it {', '.join(notes)}." if notes else ""))
    return result


def wait_until_ready(robot, joints: list, ask, say) -> bool:
    """Show where the arm is and wait for the go-ahead. The shoulder lift's question needs the base facing
    forward, so until it does, this asks for that first."""
    bus = robot.bus
    while True:
        pos = bus.sync_read("Present_Position", num_retry=2)
        raw = bus.sync_read("Present_Position", normalize=False, num_retry=2)
        say("The arm now: " + ", ".join(f"{NAMES[m]} {pos[m]:+.1f}{unit(m)}" for m in JOINTS))
        off = (raw["shoulder_pan"] - CALIBRATION_POSE) / TICKS_PER_DEGREE
        if "shoulder_lift" in joints and abs(off) > FORWARD:
            reply = ask(f"The base is turned {abs(off):.0f}° from forward. Turn it to face forward, the way it "
                        "faced in calibration Part 1, then press Enter (go = start anyway, q = quit): ")
            reply = reply.strip().lower()
            if reply.startswith("q"):
                return False
            if reply != "go":
                continue
        reply = ask("The motors will come on and hold the arm exactly where it is. Rest it or hold it still, "
                    "then press Enter (q = quit): ")
        return not reply.strip().lower().startswith("q")


def release(robot, ask, say) -> None:
    """Torque off, once someone is holding the arm: it drops when the motors let go."""
    try:
        ask("\nHold the arm so it can't drop, then press Enter to turn the motors off: ")
    except (KeyboardInterrupt, EOFError):
        say("")
    try:
        robot.bus.disable_torque(num_retry=2)
        say("Motors off.")
    except Exception as e:  # noqa: BLE001
        say(f"Couldn't turn the motors off over the bus ({e}). Switch the servo power off.")


def summary(results: list) -> str:
    lines = [f"\n{'joint':<14}{'asked':>8}{'moved':>9}{'back off by':>13}{'others moved':>14}   plus is"]
    for r in results:
        u = unit(r["joint"])
        row = f"{NAMES[r['joint']]:<14}{r['asked']:>+7.0f}{u:1}{r['moved']:>+8.1f}{u:1}"
        if r["ok"]:
            row += f"{r['back_off']:>12.1f}{u:1}{r['others_moved']:>13.1f}    {r['plus_means']}"
        else:
            row += "   didn't follow"
        lines.append(row)
    return "\n".join(lines)


def run(robot, results: list, log: list, joints=JOINTS, step: float = 10.0, seconds: float = 2.0,
        hz: float = 50.0, ask=input, say=print, sleep=time.sleep, clock=time.monotonic) -> bool:
    """The whole session on a connected bus: checks, torque on, each joint, then release. Fills `results`
    and `log` as it goes, so they survive an early stop. True if every joint tried moved as asked."""
    if not robot.calibration:
        say("There's no calibration file for this id. Run lerobot-calibrate first.")
        return False
    if not robot.bus.is_calibrated:
        say("The servos' calibration doesn't match the file. Run lerobot-calibrate again.")
        return False
    try:
        if not wait_until_ready(robot, joints, ask, say):
            return False
    except (KeyboardInterrupt, EOFError):
        say("\nStopped before the motors came on.")
        return False

    ok = True
    try:
        shift = power_on(robot, sleep)
        worst = max(shift, key=shift.get)
        if shift[worst] > MAX_JUMP:
            say(f"The {NAMES[worst]} shifted {shift[worst]:.1f}° as the motors came on, which it shouldn't. "
                "Stopping before any moves.")
            return False
        say(f"Motors on and holding. The most any joint shifted was {shift[worst]:.1f}°.")
        for joint in joints:
            result = try_joint(robot, joint, step, seconds, hz, log, ask, say, sleep=sleep, clock=clock)
            if result is None:
                continue
            results.append(result)
            if not result["ok"]:
                ok = False
                break
    except Stopped:
        pass
    except (KeyboardInterrupt, EOFError):
        say("\nStopped. The arm is holding where it is.")
        ok = False
    except (RuntimeError, ConnectionError) as e:
        say(f"\nStopped: {e}")
        ok = False
    finally:
        release(robot, ask, say)
    if results:
        say(summary(results))
    return ok


def write_log(path: Path, meta: dict, results: list, log: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**meta, "results": results, "samples": log}, indent=1), encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", required=True, help="the arm's servo adapter, e.g. /dev/cu.usbmodem5B910297421")
    ap.add_argument("--id", default="wbc_follower", help="the --robot.id given to lerobot-calibrate")
    ap.add_argument("--joints", nargs="+", choices=JOINTS, default=JOINTS,
                    help="joints to move, in this order (default: all six)")
    ap.add_argument("--degrees", type=float, default=10.0,
                    help="how far each joint moves, at most 20 (the gripper moves this many points of its "
                         "0-100 scale; default 10)")
    ap.add_argument("--seconds", type=float, default=2.0, help="how long each move takes, at least 1 (default 2)")
    args = ap.parse_args(argv)
    if not 0 < args.degrees <= 20:
        ap.error("--degrees must be more than 0 and at most 20")
    if args.seconds < 1:
        ap.error("--seconds must be at least 1")
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError:
        print("LeRobot is missing: uv pip install -e '.[lerobot]'")
        return 1

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id, max_relative_target=MAX_STEP))
    try:
        robot.bus.connect()
    except Exception as e:  # noqa: BLE001
        print(f"Can't reach the arm on {args.port}: {e}")
        return 1
    results, log, ok = [], [], False
    try:
        ok = run(robot, results, log, joints=args.joints, step=args.degrees, seconds=args.seconds)
    finally:
        try:
            robot.disconnect()  # torque off again, in case the run ended early
        except Exception as e:  # noqa: BLE001
            print(f"Couldn't disconnect cleanly ({e}). If the motors are still on, switch the servo power off.")
        if results:
            path = write_log(ROOT / ".wbc" / "logs" / f"first-moves-{datetime.now():%Y%m%d-%H%M%S}.json",
                             {"port": args.port, "id": args.id, "degrees": args.degrees, "seconds": args.seconds},
                             results, log)
            print(f"log: {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
