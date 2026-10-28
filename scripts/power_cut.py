"""Test the servo power cut-off: switch the arm's power strip off, check that the arm goes limp, and check that
it stays limp when the power comes back.

    python scripts/power_cut.py --port /dev/cu.usbmodem5B910297421 idle
    python scripts/power_cut.py --port /dev/cu.usbmodem5B910297421 hold

Run idle first. The motors stay off and nothing is written to the servos. You switch the strip off, and it
records how the servos' supply voltage falls until they stop answering. You switch it back on, and it checks
what they came back with: torque off, so the arm stays limp, and which settings the cut reset.

Then hold. The motors come on at 50% torque the way first_moves.py turns them on, with this arm's elbow at
P = 32, and hold the arm in a pose you set. Once it holds on its own, you put a hand under it and switch the
strip off, and it records when the arm let go. An arm that doesn't sink as you let go has no weight on its
motors and nothing to drop, so it says so before the cut. If a joint moves more than 5° while the voltage is
still normal, every motor is cut at once, and if the servos still answer 2 s after the voltage starts to
fall, it cuts them itself. Then the same checks as idle once the power is back.

Both wait as long as you take to switch the strip, so you can step away, and Ctrl-C stops either one. Each
run writes .wbc/logs/power-cut-<step>-<time>.json. It needs LeRobot (uv pip install -e '.[lerobot]') and,
for hold, a calibration from lerobot-calibrate under the same --id. Close any other program using the port
first.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from first_moves import (JOINTS, MAX_STRAY, NAMES, TICKS_PER_DEGREE, TORQUE_CAP, Tripped, choose,  # noqa: E402
                         cut_torque, parse_gains, power_on, release, wait_until_ready)
from wbc.config import ROOT  # noqa: E402

GAINS = {"elbow_flex": 32}  # this arm's elbow, a weaker 1:191 servo, can't hold its forearm up at LeRobot's 16
SAG = 0.3        # volts under the reading when the watch began that mean the supply is going
SILENT = 3       # failed reads in a row that mean the servos have lost power
DROP = 1.0       # degrees a held joint has to fall to count as the arm letting go
MAX_SINK = 15.0  # degrees the arm may sink as you let go of it and the motors take its weight
MIN_SINK = 0.5   # degrees it has to sink to show the motors carry weight: at P = 16 that's about 2% effort
GRACE = 2.0      # seconds the servos may still answer after the voltage starts to fall, in hold
POLL = 0.25      # seconds between pings while the power is off
BOOT = 0.5       # seconds to let the servos finish starting up once all six answer again
KEEP = 500       # readings kept from before the voltage started to fall
# Read before and after each cut. Present_Voltage is in tenths of a volt and Torque_Limit in tenths of a
# percent. Torque_Limit is the RAM copy of Max_Torque_Limit that a servo makes at power-up.
STATE = ["Present_Voltage", "Torque_Enable", "Torque_Limit", "Max_Torque_Limit", "P_Coefficient",
         "Goal_Position", "Present_Position", "Present_Load", "Present_Temperature", "Status", "Lock"]
# Status bits every Feetech table names the same way. Bits 1 and 4 mean different things in different tables.
FLAGS = {0: "voltage", 2: "overheat", 3: "current", 5: "overload"}
PORT_LOST = ("The arm's USB adapter went away too, so the strip seems to power the adapter or its USB hub as "
             "well. Only the servos' power supply should be on the strip: move the rest, then run this again.")


class PortLost(Exception):
    """The arm's USB adapter went away, not just the servos behind it."""


def listed(joints) -> str:
    return ", ".join(NAMES[m] for m in joints)


def read_state(bus) -> dict:
    """Every servo's STATE registers, raw, as {register: {joint: value}}."""
    return {name: bus.sync_read(name, normalize=False, num_retry=2) for name in STATE}


def read(bus, name: str, retries: int = 0) -> dict:
    """One register of every servo, raw. Servos that don't answer raise ConnectionError, and a port that has
    gone away raises PortLost."""
    try:
        return bus.sync_read(name, normalize=False, num_retry=retries)
    except ConnectionError:
        raise
    except OSError as e:
        raise PortLost(str(e)) from e


def flag_names(status: int) -> str:
    return ", ".join(FLAGS.get(bit, f"bit {bit}") for bit in range(8) if status >> bit & 1) or "none"


def table(state: dict) -> str:
    rows = [f"  {'joint':<14}{'volts':>6}{'torque':>8}{'cap':>6}{'P':>4}{'goal':>6}{'position':>10}{'temp':>7}"
            "   flags"]
    for m in JOINTS:
        rows.append(f"  {NAMES[m]:<14}{state['Present_Voltage'][m] / 10:>5.1f}V"
                    f"{'on' if state['Torque_Enable'][m] else 'off':>8}{state['Torque_Limit'][m] / 10:>5.0f}%"
                    f"{state['P_Coefficient'][m]:>4}{state['Goal_Position'][m]:>6}"
                    f"{state['Present_Position'][m]:>10}{state['Present_Temperature'][m]:>5}°C"
                    f"   {flag_names(state['Status'][m])}")
    return "\n".join(rows)


def wait_for_cut(robot, hold: dict | None = None, clock=time.monotonic) -> dict:
    """Read every servo's position and voltage as fast as the bus allows until they stop answering. Returns
    the readings from shortly before the voltage started to fall up to the last answer, with when the voltage
    started to fall (sag_at, None if it fell faster than the readings), when the servos last answered
    (last_at), and how many replies went missing with the servos answering again right after (glitches).
    Times are seconds from the start of the watch.

    With `hold`, the raw positions a powered arm is holding: if a joint moves more than MAX_STRAY while the
    voltage is still normal, every motor is cut at once and Tripped raised, and if the servos still answer
    GRACE seconds after the voltage started to fall, every motor is cut (cut_at) and the watch goes on."""
    bus = robot.bus
    start = read(bus, "Present_Voltage", retries=2)
    cut = {"volts_at_start": {m: v / 10 for m, v in start.items()}, "sag_at": None, "last_at": None,
           "cut_at": None, "glitches": 0}
    readings, missed = [], 0
    t0 = clock()
    while missed < SILENT:
        t = round(clock() - t0, 4)
        try:
            pos = read(bus, "Present_Position")
            volts = read(bus, "Present_Voltage")
        except ConnectionError:  # no reply, or a garbled one
            missed += 1
            continue
        cut["glitches"] += missed
        missed = 0
        cut["last_at"] = t
        readings.append({"t": t, **pos, **{f"{m}_v": volts[m] / 10 for m in JOINTS}})
        fall = max(start[m] - volts[m] for m in JOINTS) / 10
        if cut["sag_at"] is None and fall >= SAG:
            cut["sag_at"] = t
        elif cut["sag_at"] is not None and cut["cut_at"] is None and fall <= 0.1:
            cut["sag_at"] = None  # only a dip, and the supply held
        if cut["sag_at"] is None and len(readings) > 2 * KEEP:
            del readings[:-KEEP]
        if hold and cut["cut_at"] is None:
            moved = {m: (pos[m] - hold[m]) / TICKS_PER_DEGREE for m in JOINTS}
            worst = max(moved, key=lambda m: abs(moved[m]))
            if cut["sag_at"] is None and abs(moved[worst]) > MAX_STRAY:
                cut_torque(robot)
                raise Tripped(f"the {NAMES[worst]} moved {moved[worst]:+.1f}° with the power still on")
            if cut["sag_at"] is not None and t - cut["sag_at"] > GRACE:
                cut_torque(robot)
                cut["cut_at"] = t
    cut["readings"] = readings
    return cut


def let_go(readings: list, hold: dict) -> tuple:
    """When the held arm began its fall, as (time, joint): the earliest reading from which some joint stayed
    at least DROP degrees from its hold through the last reading. (None, None) if no joint had fallen that far
    by the time the servos went silent."""
    first = (None, None)
    for m in JOINTS:
        i = len(readings)
        while i and abs(readings[i - 1][m] - hold[m]) >= DROP * TICKS_PER_DEGREE:
            i -= 1
        if i < len(readings) and (first[0] is None or readings[i]["t"] < first[0]):
            first = (readings[i]["t"], m)
    return first


def describe(cut: dict, hold: dict | None = None) -> list:
    """The cut, in plain sentences."""
    readings, sag, last = cut["readings"], cut["sag_at"], cut["last_at"]
    if not readings:
        return ["The servos stopped answering before the first reading."]
    gaps = [b["t"] - a["t"] for a, b in zip(readings, readings[1:])]
    every = f" Readings came every {1000 * statistics.median(gaps):.0f} ms." if gaps else ""
    low = min(readings[-1][f"{m}_v"] for m in JOINTS)
    if sag is None:
        lines = [f"The servos went silent with the voltage still reading {low:.1f} V: it fell faster than the "
                 f"readings could follow.{every}"]
    else:
        lines = [f"The voltage started to fall {sag:.2f} s into the watch, and the servos last answered "
                 f"{1000 * (last - sag):.0f} ms later, at {low:.1f} V.{every}"]
    if hold:
        when, joint = let_go(readings, hold)
        if when is None:
            lines.append(f"No joint had fallen {DROP:.0f}° before the servos went silent.")
        else:
            ref, what = (sag, "the voltage did") if sag is not None else (last, "the servos last answered")
            gap = 1000 * (when - ref)
            lines.append(f"The arm started to fall {abs(gap):.0f} ms {'after' if gap >= 0 else 'before'} {what}, "
                         f"the {NAMES[joint]} first.")
        fell = {m: (readings[-1][m] - hold[m]) / TICKS_PER_DEGREE for m in JOINTS}
        moved = [f"{NAMES[m]} {fell[m]:+.1f}°" for m in JOINTS if abs(fell[m]) >= 0.5]
        lines.append("By the last reading: " + (", ".join(moved) if moved else "every joint within 0.5° of its hold")
                     + ".")
    if cut["cut_at"] is not None:
        lines.append(f"The servos still answered {GRACE:.0f} s after the voltage started to fall, so this cut "
                     "every motor itself.")
    if cut["glitches"]:
        lines.append(f"{cut['glitches']} replies went missing before the cut, with the servos answering again "
                     "right after.")
    return lines


def wait_for_power(bus, volts_before: dict, say, sleep=time.sleep) -> None:
    """Wait until all six servos answer again with their voltage back within SAG of volts_before, saying which
    answer while some don't, then give them BOOT seconds to finish starting up."""
    shown = []
    while True:
        try:
            back = [m for m in JOINTS if bus.ping(m) is not None]
            volts = read(bus, "Present_Voltage", retries=2) if len(back) == len(JOINTS) else {}
        except ConnectionError:  # they dropped out again between the pings and the read
            back, volts = [], {}
        except OSError as e:  # a ping, on a port that has gone away
            raise PortLost(str(e)) from e
        if volts and all(volts[m] / 10 >= volts_before[m] - SAG for m in JOINTS):
            sleep(BOOT)
            return
        if back != shown:
            if volts:
                say(f"  All six answer, but at only {min(volts.values()) / 10:.1f} V. Waiting for the full voltage.")
            elif back:
                say(f"  {len(back)} of 6 answer so far: {listed(back)}.")
        shown = back
        sleep(POLL)


def changes(before: dict, after: dict, held: bool) -> list:
    """What the servos came back with after the cut, compared with before it, in plain sentences."""
    on = [m for m in JOINTS if after["Torque_Enable"][m]]
    lines = [f"Torque came back ON for the {listed(on)}, so this turned it off again. A servo that powers up "
             "with torque on drives to its goal at once." if on else
             "Torque came back off on all six, so the arm stays limp when the power returns."]
    caps = [f"{NAMES[m]} {before['Torque_Limit'][m] / 10:.0f}% → {after['Torque_Limit'][m] / 10:.0f}%"
            for m in JOINTS if after["Torque_Limit"][m] != before["Torque_Limit"][m]]
    lines.append(f"The torque cap reset to each servo's stored maximum: {', '.join(caps)}. Anything that turns "
                 "torque on after a cut has to set its cap again first." if caps else
                 "The torque cap came back as it was.")
    gains = [f"{NAMES[m]} {before['P_Coefficient'][m]} → {after['P_Coefficient'][m]}"
             for m in JOINTS if after["P_Coefficient"][m] != before["P_Coefficient"][m]]
    lines.append(f"P gains changed: {', '.join(gains)}." if gains else
                 "P gains came back as they were (they're stored in EEPROM).")
    goals, pos = after["Goal_Position"], after["Present_Position"]
    if not any(goals.values()):
        lines.append("Every goal came back as 0, so torque must never come on before goals are written.")
    elif all(abs(goals[m] - pos[m]) <= 10 for m in JOINTS):
        lines.append("Every goal came back at its joint's position, so torque coming on would hold the arm "
                     "where it is.")
    else:
        lines.append("Goals came back as " + ", ".join(f"{NAMES[m]} {goals[m]}" for m in JOINTS) + ".")
    if not held:
        moved = {m: abs(pos[m] - before["Present_Position"][m]) / TICKS_PER_DEGREE for m in JOINTS}
        worst = max(moved, key=moved.get)
        lines.append(f"The {NAMES[worst]} moved {moved[worst]:.1f}° while the power was off." if moved[worst] >= DROP
                     else f"Nothing moved: every joint is within {moved[worst]:.1f}° of where it was.")
    flagged = [m for m in JOINTS if after["Status"][m]]
    if flagged:
        lines.append("Flags: " + ", ".join(f"{NAMES[m]} {flag_names(after['Status'][m])}" for m in flagged) + ".")
    lines.append(f"The supply is back at {min(after['Present_Voltage'].values()) / 10:.1f} V.")
    return lines


def put_back(bus, limits: dict) -> None:
    """Every Torque_Limit back to what it was before the run. It's in RAM, so a power cycle restores it too."""
    try:
        for m, value in limits.items():
            bus.write("Torque_Limit", m, value, normalize=False, num_retry=2)
    except Exception:  # noqa: BLE001
        pass


def run_idle(robot, record: dict, say=print, sleep=time.sleep, clock=time.monotonic) -> bool:
    """Step 1, with the motors off: you switch the strip off, then back on. Writes nothing to the servos unless
    one comes back with its torque on. Fills `record` (before, cut, after) as it goes. True if the servos lost
    power and all six came back limp."""
    bus = robot.bus
    record["before"] = before = read_state(bus)
    say("The servos now:\n" + table(before))
    on = [m for m in JOINTS if before["Torque_Enable"][m]]
    if on:
        say(f"\nTorque is on for the {listed(on)}. This step needs the motors off, with the arm resting.")
        return False
    say("\nOnly the servos' power supply should be plugged into the strip: not the Mac, a USB hub or an ESP32.")
    say("Switch the strip off now. I'll wait as long as it takes (Ctrl-C to stop).")
    try:
        record["cut"] = cut = wait_for_cut(robot, clock=clock)
        for line in describe(cut):
            say("  " + line)
        say("\nThe servos are off. Keep your hands clear of the arm and switch the strip back on (Ctrl-C to "
            "stop).")
        wait_for_power(bus, cut["volts_at_start"], say, sleep=sleep)
    except (KeyboardInterrupt, EOFError):
        say("\nStopped.")
        return False
    except PortLost:
        say("\n" + PORT_LOST)
        return False
    record["after"] = after = read_state(bus)
    on = [m for m in JOINTS if after["Torque_Enable"][m]]
    if on:
        cut_torque(robot)
    say("All six answer again:\n" + table(after))
    say("\nResults:\n  The strip cuts the servos' power.")
    for line in changes(before, after, held=False):
        say("  " + line)
    return not on


def run_hold(robot, record: dict, cap: int = TORQUE_CAP, gains: dict | None = None, ask=input, say=print,
             sleep=time.sleep, clock=time.monotonic) -> bool:
    """Step 2: the motors hold the arm at `cap` percent in a pose you set, and you switch the strip off, then
    back on. Fills `record` (power_on, hold, before, cut, after, let_go, stopped) as it goes. True if the arm
    let go when the power went and all six came back limp."""
    gains = GAINS if gains is None else gains
    if not robot.calibration:
        say("There's no calibration file for this id. Run lerobot-calibrate first.")
        return False
    if not robot.bus.is_calibrated:
        say("The servos' calibration doesn't match the file. Run lerobot-calibrate again.")
        return False
    bus = robot.bus
    say("Pose the arm so its weight hangs out in front of the shoulder, with room under the elbow for your hand: "
        f"the upper arm tipped well forward of upright. A level upper arm may be more than {cap}% can hold.")
    try:
        if not wait_until_ready(robot, JOINTS, cap, ask, say, face_forward=False):
            return False
    except (KeyboardInterrupt, EOFError):
        say("\nStopped before the motors came on.")
        return False

    limits = bus.sync_read("Torque_Limit", normalize=False, num_retry=2)
    record["power_on"] = []
    try:
        power_on(robot, JOINTS, cap, record["power_on"], gains=gains, clock=clock)
        on_at = bus.sync_read("Present_Position", normalize=False, num_retry=2)
        reply = ask(f"\nTorque on at {cap}%. Let go of the arm slowly: it sinks a little as the motors take its "
                    "weight. Press Enter once it holds on its own (q = turn the motors off): ")
        if reply.strip().lower().startswith("q"):
            release(robot, ask, say)
            put_back(bus, limits)
            return False
        record["hold"] = hold = bus.sync_read("Present_Position", normalize=False, num_retry=2)
        sank = {m: (hold[m] - on_at[m]) / TICKS_PER_DEGREE for m in JOINTS}
        record["sank"] = {m: round(v, 2) for m, v in sank.items()}
        worst = max(sank, key=lambda m: abs(sank[m]))
        if abs(sank[worst]) > MAX_SINK:
            raise Tripped(f"the {NAMES[worst]} sank {sank[worst]:+.1f}° as you let go, more than the motors hold "
                          f"at {cap}%")
        loaded = abs(sank[worst]) >= MIN_SINK
        if not loaded:
            reply = ask("\nIt didn't sink as you let go, so its motors are carrying almost no weight, and it may not "
                        "drop when the power goes. Press Enter to cut the power anyway, or q to turn the motors off "
                        "and pose it with more weight in front of the shoulder: ")
            if reply.strip().lower().startswith("q"):
                release(robot, ask, say)
                put_back(bus, limits)
                return False
        record["before"] = before = read_state(bus)
        where = f", at the {NAMES[worst]}" if abs(sank[worst]) >= 0.1 else ""
        say(f"Holding on its own. The most it sank as you let go was {abs(sank[worst]):.1f}°{where}.\n"
            + table(before))
        say("\nPut one hand flat under the elbow, a few centimetres below it, without touching it. Then switch "
            "the strip off with the other hand. I'm watching the arm until the power goes (Ctrl-C = stop without "
            "the cut).")
        record["cut"] = cut = wait_for_cut(robot, hold=hold, clock=clock)
    except (KeyboardInterrupt, EOFError):
        say("\nStopped before the cut. The arm is holding where it is.")
        release(robot, ask, say)
        put_back(bus, limits)
        return False
    except PortLost:
        say("\n" + PORT_LOST)
        return False
    except Exception as e:  # noqa: BLE001 - Tripped, a bus error, or a bug: whatever it is, cut the motors
        cut_torque(robot)
        record["stopped"] = str(e)
        say(f"\nStopped: {e}. Every motor is off now, so the arm is limp.")
        put_back(bus, limits)
        return False

    cut_torque(robot)  # unheard if the servos are off; if they aren't, the arm drops into your hand
    for line in describe(cut, hold):
        say("  " + line)
    try:
        say("\nThe arm is limp. Lower it to its rest pose, then switch the strip back on (Ctrl-C to stop).")
        wait_for_power(bus, cut["volts_at_start"], say, sleep=sleep)
    except (KeyboardInterrupt, EOFError):
        say("\nStopped.")
        return False
    except PortLost:
        say("\n" + PORT_LOST)
        return False
    record["after"] = after = read_state(bus)
    on = [m for m in JOINTS if after["Torque_Enable"][m]]
    cut_torque(robot)  # nothing should be on, and nothing may come on by itself
    say("All six answer again:\n" + table(after))
    showed = loaded or let_go(cut["readings"], hold)[0] is not None  # weight on the motors, or a fall seen
    say("\nResults:")
    if cut["cut_at"] is not None:
        say(f"  The servos kept answering {GRACE:.0f} s after the voltage started to fall, so this cut the motors "
            "itself. Check that the strip switches the servos' supply.")
    elif showed:
        say("  The strip stops the arm: its servos lost power and let go.")
    else:
        say("  The servos lost power, but the arm carried no weight, so this run couldn't show it letting go.")
    for line in changes(before, after, held=True):
        say("  " + line)
    if showed:
        try:
            felt = choose(ask, "\nDid the arm let go the moment you switched the strip off? 1 = at once, "
                               "2 = after a short pause, 3 = a second or more later: ", "123")
            record["let_go"] = {"1": "at once", "2": "after a short pause", "3": "a second or more later"}[felt]
        except (KeyboardInterrupt, EOFError):
            pass
    return not on and cut["cut_at"] is None and showed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["idle", "hold"],
                    help="idle: the motors stay off (run it first). hold: they hold the arm while the power goes")
    ap.add_argument("--port", required=True, help="the arm's servo adapter, e.g. /dev/cu.usbmodem5B910297421")
    ap.add_argument("--id", default="wbc_follower", help="the --robot.id given to lerobot-calibrate")
    ap.add_argument("--torque", type=int, default=TORQUE_CAP,
                    help=f"hold: torque cap in percent of full, 10-100 (default {TORQUE_CAP})")
    ap.add_argument("--gain", action="append", default=[], metavar="JOINT=P",
                    help="hold: a joint's P gain after LeRobot's setup, as in first_moves.py (default: "
                         "elbow_flex=32)")
    args = ap.parse_args(argv)
    try:
        gains = {**GAINS, **parse_gains(args.gain)}
    except ValueError as e:
        ap.error(str(e))
    if not 10 <= args.torque <= 100:
        ap.error("--torque must be between 10 and 100")
    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError:
        print("LeRobot is missing: uv pip install -e '.[lerobot]'")
        return 1

    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.id))
    try:
        robot.bus.connect()
    except Exception as e:  # noqa: BLE001
        print(f"Can't reach the arm on {args.port}: {e}")
        return 1
    record = {"step": args.step, "port": args.port, "id": args.id}
    ok = False
    try:
        if args.step == "idle":
            ok = run_idle(robot, record)
        else:
            record.update(torque=args.torque, gains=gains)
            ok = run_hold(robot, record, cap=args.torque, gains=gains)
    finally:
        if args.step == "hold":
            cut_torque(robot)  # in case the run ended some way that skipped it
        try:
            robot.bus.disconnect(disable_torque=False)
        except Exception as e:  # noqa: BLE001
            print(f"Couldn't close the port cleanly ({e}).")
        if "before" in record or "power_on" in record:
            path = ROOT / ".wbc" / "logs" / f"power-cut-{args.step}-{datetime.now():%Y%m%d-%H%M%S}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=1), encoding="utf-8")
            print(f"log: {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
