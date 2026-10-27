"""Move each SO-101 joint a little under power for the first time, one joint at a time.

The motors come on one joint at a time, each holding exactly where it is, with torque capped at 50%. If a
joint moves more than 3° on its own as its torque comes on, or strays more than 5° from where it should be
during a move, every motor is cut at once. Then each joint in turn moves 10° toward the middle of its range
and back, slowly, and you say which way it went: that's the sign the IK needs for each joint. Every move
waits for Enter, and after a normal finish the motors only go off once you say you're holding the arm,
because it drops when they do.

    python scripts/first_moves.py --port /dev/cu.usbmodem5B910297421
    python scripts/first_moves.py --port /dev/cu.usbmodem5B910297421 --joints elbow_flex

Only the joints in --joints get torque, so hold the rest of the arm when you test a few. It needs LeRobot
(uv pip install -e '.[lerobot]') and a calibration from lerobot-calibrate under the same --id. Ctrl-C stops
the moves, and the arm holds where it is until you release it. Each run writes
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
FORWARD = 30.0    # degrees the base may be off forward and "forward or back" still makes sense
MAX_JUMP = 3.0    # degrees a joint may move on its own as torque comes on before every motor is cut
MAX_STRAY = 5.0   # degrees a joint may be from where it should be during a move before every motor is cut
MAX_STEP = 5.0    # LeRobot's max_relative_target: no goal goes more than this far from where the joint is
TORQUE_CAP = 50   # percent of full torque while this runs (each servo's Torque_Limit), unless --torque says


class Stopped(Exception):
    """You chose to finish early."""


class Tripped(Exception):
    """A joint moved when or where it shouldn't, and every motor was cut at once."""


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


def cut_torque(robot) -> None:
    """Every motor off at once. This is one broadcast write that servos don't answer, so a servo flagging
    overload can't refuse it the way it makes LeRobot's one-servo writes fail. Sent twice in case one is lost."""
    for _ in range(2):
        try:
            robot.bus.sync_write("Torque_Enable", {m: 0 for m in JOINTS}, normalize=False)
        except Exception:  # noqa: BLE001 - nothing more to try over the bus
            pass


def watch(robot, joint: str, powered: list, still: dict, shift: dict, samples: list, seconds: float,
          clock=time.monotonic) -> None:
    """Read the arm as fast as the bus allows for `seconds` after `joint`'s torque came on. If a powered joint
    moves more than MAX_JUMP from `still` (raw ticks), or the bus fails, cut every motor and raise Tripped."""
    bus = robot.bus
    t0 = clock()
    while clock() - t0 < seconds:
        try:
            now = bus.sync_read("Present_Position", normalize=False, num_retry=2)
            load = bus.sync_read("Present_Load", normalize=False, num_retry=2)
        except Exception as e:  # noqa: BLE001
            cut_torque(robot)
            raise Tripped(f"the bus failed as the {NAMES[joint]} came on ({e})") from e
        moved = {m: (now[m] - still[m]) / TICKS_PER_DEGREE for m in powered}
        samples.append({"t": round(clock() - t0, 4), "came_on": joint,
                        **{m: round(v, 2) for m, v in moved.items()}, **{f"{m}_load": load[m] for m in powered}})
        for m, v in moved.items():
            shift[m] = max(shift[m], abs(v))
        worst = max(moved, key=lambda m: abs(moved[m]))
        if abs(moved[worst]) > MAX_JUMP:
            cut_torque(robot)
            raise Tripped(f"the {NAMES[worst]} moved {moved[worst]:+.1f}° on its own, "
                          f"{1000 * (clock() - t0):.0f} ms after the {NAMES[joint]}'s torque came on "
                          f"(load {load[worst]:+d})")


def power_on(robot, joints: list, cap: int, samples: list, clock=time.monotonic) -> dict:
    """Turn torque on for `joints`, one at a time from the base out, without the arm moving. Returns how far
    each powered joint shifted, in degrees.

    A servo drives to whatever its goal register holds the moment torque comes on. After calibration changed
    the homing offsets and the arm was moved by hand, that can be far from where the arm is, and LeRobot's
    connect() turns torque on without setting it. So this runs LeRobot's own configure() with its final
    torque-on held back, caps the torque, sets each goal to the present position and reads it back, then
    turns the joints on one by one, watching the arm for 0.3 s after each."""
    bus = robot.bus
    enable = bus.enable_torque
    bus.enable_torque = lambda *args, **kwargs: None  # configure() ends by turning torque on: not yet
    try:
        robot.configure()
    finally:
        del bus.enable_torque  # back to the bus's own method
    for m in JOINTS:
        bus.write("Torque_Limit", m, cap * 10, normalize=False, num_retry=2)
    present = bus.sync_read("Present_Position", normalize=False, num_retry=2)
    for m, value in present.items():
        bus.write("Goal_Position", m, value, normalize=False, num_retry=2)
    goals = bus.sync_read("Goal_Position", normalize=False, num_retry=2)
    wrong = [m for m in present if goals[m] != present[m]]
    if wrong:
        raise RuntimeError(f"the hold-here goal didn't take on {', '.join(wrong)}, so torque stayed off")
    shift = {m: 0.0 for m in joints}
    on = []
    for joint in [m for m in JOINTS if m in joints]:
        enable(joint, num_retry=2)
        on.append(joint)
        watch(robot, joint, on, present, shift, samples, seconds=0.3, clock=clock)
    return shift


def move(robot, joint: str, start: float, end: float, seconds: float, hz: float, still: dict, powered: list,
         samples: list, settle: float = 0.5, sleep=time.sleep, clock=time.monotonic) -> tuple:
    """Drive one joint from start to end, then hold it there for `settle` seconds. Logs every reading and
    returns the last one (all six joints) and how far any other powered joint strayed from `still`. If the
    joint strays more than MAX_STRAY from its goal, or another powered joint from where it was, every motor is
    cut and Tripped raised."""
    period = 1.0 / hz
    goals = ramp(start, end, seconds, hz) + [end] * max(1, round(settle * hz))
    others = [m for m in powered if m != joint]
    u = unit(joint)
    drift = 0.0
    t0 = clock()
    for k, goal in enumerate(goals, 1):
        try:
            sent = robot.send_action({f"{joint}.pos": goal})[f"{joint}.pos"]
            obs = robot.get_observation()
        except Exception as e:  # noqa: BLE001
            cut_torque(robot)
            raise Tripped(f"the bus failed while the {NAMES[joint]} was moving ({e})") from e
        now = {m: obs[f"{m}.pos"] for m in JOINTS}
        samples.append({"t": round(clock() - t0, 3), "joint": joint, "goal": round(sent, 2),
                        **{m: round(v, 2) for m, v in now.items()}})
        if abs(now[joint] - goal) > MAX_STRAY:
            cut_torque(robot)
            raise Tripped(f"the {NAMES[joint]} was {now[joint] - goal:+.1f}{u} from where it should be, "
                          f"partway through its move")
        for m in others:
            drift = max(drift, abs(now[m] - still[m]))
            if abs(now[m] - still[m]) > MAX_STRAY:
                cut_torque(robot)
                raise Tripped(f"the {NAMES[m]} moved {now[m] - still[m]:+.1f}{unit(m)} while the "
                              f"{NAMES[joint]} was moving")
        wait = t0 + k * period - clock()
        if wait > 0:
            sleep(wait)
    return now, drift


def try_joint(robot, joint: str, powered: list, step: float, seconds: float, hz: float, samples: list, ask, say,
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
        return move(robot, joint, a, b, seconds, hz, still, powered, samples, sleep=sleep, clock=clock)

    question, options = QUESTIONS[joint]
    drift = 0.0
    while True:
        reached, d = leg(start, target)
        drift = max(drift, d)
        moved = reached[joint] - start
        if moved * delta < 0:
            cut_torque(robot)
            raise Tripped(f"the {NAMES[joint]} moved {moved:+.1f}{u} when asked for {delta:+.0f}{u}")
        if abs(moved) < 0.3 * step:
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


def wait_until_ready(robot, joints: list, cap: int, ask, say) -> bool:
    """Show where the arm is and wait for the go-ahead. Every joint that will get torque has to start inside
    its calibrated range, and the shoulder lift's question needs the base facing forward, so until both hold,
    this asks for them first."""
    bus, cal = robot.bus, robot.calibration
    while True:
        pos = bus.sync_read("Present_Position", num_retry=2)
        raw = bus.sync_read("Present_Position", normalize=False, num_retry=2)
        say("The arm now: " + ", ".join(f"{NAMES[m]} {pos[m]:+.1f}{unit(m)}" for m in JOINTS))
        outside = {m: max(cal[m].range_min - raw[m], raw[m] - cal[m].range_max) / TICKS_PER_DEGREE
                   for m in joints if not cal[m].range_min <= raw[m] <= cal[m].range_max}
        if outside:
            listed = ", ".join(f"the {NAMES[m]} by {past:.1f}°" for m, past in outside.items())
            reply = ask(f"Past the end of its calibrated range: {listed}. Move it in a little, toward the "
                        "middle, then press Enter (q = quit): ")
            if reply.strip().lower().startswith("q"):
                return False
            continue
        off = (raw["shoulder_pan"] - CALIBRATION_POSE) / TICKS_PER_DEGREE
        if "shoulder_lift" in joints and abs(off) > FORWARD:
            reply = ask(f"The base is turned {abs(off):.0f}° from forward. Turn it to face forward, the way it "
                        "faced in calibration Part 1, then press Enter (go = start anyway, q = quit): ")
            reply = reply.strip().lower()
            if reply.startswith("q"):
                return False
            if reply != "go":
                continue
        if set(joints) == set(JOINTS):
            prompt = (f"The motors will come on one at a time, at {cap}% torque, and hold the arm exactly "
                      "where it is. Rest it or hold it still, then press Enter (q = quit): ")
        else:
            prompt = (f"Only the {', '.join(NAMES[m] for m in joints)} will get torque, at {cap}%, so hold "
                      "the rest of the arm still. Press Enter when ready (q = quit): ")
        return not ask(prompt).strip().lower().startswith("q")


def release(robot, ask, say) -> None:
    """Every motor off, once someone is holding the arm: it drops when the motors let go."""
    try:
        ask("\nHold the arm so it can't drop, then press Enter to turn the motors off: ")
    except (KeyboardInterrupt, EOFError):
        say("")
    cut_torque(robot)
    try:
        on = [m for m, v in robot.bus.sync_read("Torque_Enable", normalize=False, num_retry=2).items() if v]
    except Exception:  # noqa: BLE001
        on = None
    say("Motors off." if on == [] else
        "Couldn't confirm every motor is off. If the arm is still stiff, switch the servo power off.")


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


def run(robot, record: dict, joints=JOINTS, step: float = 10.0, seconds: float = 2.0, hz: float = 50.0,
        cap: int = TORQUE_CAP, ask=input, say=print, sleep=time.sleep, clock=time.monotonic) -> bool:
    """The whole session on a connected bus: checks, torque on, each joint, then release. Fills `record`
    (results, samples, power_on, power_on_shift, stopped) as it goes, so it survives an early stop. True if
    every joint tried moved as asked."""
    for key, empty in (("results", []), ("samples", []), ("power_on", []), ("power_on_shift", None),
                       ("stopped", None)):
        record.setdefault(key, empty)
    if not robot.calibration:
        say("There's no calibration file for this id. Run lerobot-calibrate first.")
        return False
    if not robot.bus.is_calibrated:
        say("The servos' calibration doesn't match the file. Run lerobot-calibrate again.")
        return False
    try:
        if not wait_until_ready(robot, joints, cap, ask, say):
            return False
    except (KeyboardInterrupt, EOFError):
        say("\nStopped before the motors came on.")
        return False

    limits = robot.bus.sync_read("Torque_Limit", normalize=False, num_retry=2)
    ok, holding = True, True
    try:
        shift = power_on(robot, joints, cap, record["power_on"], clock=clock)
        record["power_on_shift"] = {m: round(v, 2) for m, v in shift.items()}
        say(f"Torque on at {cap}%, holding. The most any joint shifted as it came on was "
            f"{max(shift.values()):.1f}°.")
        for joint in joints:
            result = try_joint(robot, joint, joints, step, seconds, hz, record["samples"], ask, say,
                               sleep=sleep, clock=clock)
            if result is None:
                continue
            record["results"].append(result)
            if not result["ok"]:
                ok = False
                break
    except Stopped:
        pass
    except (KeyboardInterrupt, EOFError):
        say("\nStopped. The arm is holding where it is.")
        ok = False
    except Exception as e:  # noqa: BLE001 - Tripped, a bus error, or a bug: whatever it is, cut the motors
        cut_torque(robot)
        ok, holding, record["stopped"] = False, False, str(e)
        say(f"\nStopped: {e}. Every motor is off now, so the arm is limp.")
    finally:
        if holding:
            release(robot, ask, say)
        try:
            for m, value in limits.items():
                robot.bus.write("Torque_Limit", m, value, normalize=False, num_retry=2)
        except Exception:  # noqa: BLE001 - Torque_Limit is in RAM, so a power cycle restores it anyway
            pass
    if record["results"]:
        say(summary(record["results"]))
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", required=True, help="the arm's servo adapter, e.g. /dev/cu.usbmodem5B910297421")
    ap.add_argument("--id", default="wbc_follower", help="the --robot.id given to lerobot-calibrate")
    ap.add_argument("--joints", nargs="+", choices=JOINTS, default=JOINTS,
                    help="joints to power and move, in this order (default: all six)")
    ap.add_argument("--degrees", type=float, default=10.0,
                    help="how far each joint moves, at most 20 (the gripper moves this many points of its "
                         "0-100 scale; default 10)")
    ap.add_argument("--seconds", type=float, default=2.0, help="how long each move takes, at least 1 (default 2)")
    ap.add_argument("--torque", type=int, default=TORQUE_CAP,
                    help=f"torque cap in percent of full, 10-100 (default {TORQUE_CAP})")
    args = ap.parse_args(argv)
    if not 0 < args.degrees <= 20:
        ap.error("--degrees must be more than 0 and at most 20")
    if args.seconds < 1:
        ap.error("--seconds must be at least 1")
    if not 10 <= args.torque <= 100:
        ap.error("--torque must be between 10 and 100")
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
    record = {"port": args.port, "id": args.id, "joints": args.joints, "degrees": args.degrees,
              "seconds": args.seconds, "torque": args.torque}
    ok = False
    try:
        ok = run(robot, record, joints=args.joints, step=args.degrees, seconds=args.seconds, cap=args.torque)
    finally:
        cut_torque(robot)  # in case the run ended some way that skipped it
        try:
            robot.bus.disconnect(disable_torque=False)
        except Exception as e:  # noqa: BLE001
            print(f"Couldn't close the port cleanly ({e}).")
        if record.get("power_on") or record.get("results"):
            path = ROOT / ".wbc" / "logs" / f"first-moves-{datetime.now():%Y%m%d-%H%M%S}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1), encoding="utf-8")
            print(f"log: {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
