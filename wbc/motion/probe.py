"""Gripper probe: log what the gripper servo can feel, then measure each response against the noise at rest.

watch    Read only. Torque is left as it is, so you can move the jaws by hand and read off the open and
         closed positions.
squeeze  Closes slowly onto an object with a capped torque and holds it, so pulls, twists and pushes on
         the object show up in load and current. Torque is released when the run ends, however it ends.

Both start from the servo's own settings (read_settings). A squeeze stays inside the position limits that
lerobot-calibrate wrote, and under the torque where the servo's overload protection starts: LeRobot sets the
gripper's to 25%, and a squeeze that tripped it would change partway through a measurement. On this arm a
higher position is a more open gripper.
"""
from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

from . import sts

FIELDS = ["t", "position", "speed", "load", "current", "voltage", "temperature", "moving", "error", "goal", "mark"]
TORQUE_CEILING = 600  # per mille of max drive. The probe refuses more, so a wrong --closed can't crush anything.
MISSES = 5    # replies in a row that may go missing before a run stops. A single one is skipped.
CLOSED_ALREADY = 30  # ticks: jaws this near the closed goal have nothing between them (the board sat 70 away)
GAIN_RANGE = (8, 64)  # P gains a squeeze accepts. LeRobot's configure() sets 16: about 3.9 per mille of drive per tick
OVERLOAD = 5  # the overload bit, in a reply's status flags and in the servo's UNLOADING_CONDITION
# Read before every run, as (address, bytes)
SETTINGS = {
    "torque_on": (sts.TORQUE_ENABLE, 1),
    "min_position": (sts.MIN_POSITION_LIMIT, 2),
    "max_position": (sts.MAX_POSITION_LIMIT, 2),
    "torque_limit": (sts.TORQUE_LIMIT, 2),
    "max_torque_limit": (sts.MAX_TORQUE_LIMIT, 2),
    "p_gain": (sts.P_COEFFICIENT, 1),
    "protection": (sts.UNLOADING_CONDITION, 1),
    "overload_torque": (sts.OVERLOAD_TORQUE, 1),
    "protection_time": (sts.PROTECTION_TIME, 1),
    "protective_torque": (sts.PROTECTIVE_TORQUE, 1),
    "protection_current": (sts.PROTECTION_CURRENT, 2),
}


def read_settings(bus, servo_id: int) -> dict:
    """The servo's registers that decide what a squeeze may do, by their SETTINGS names. Writes nothing."""
    return {name: (bus.read_u16 if size == 2 else bus.read_u8)(servo_id, address)
            for name, (address, size) in SETTINGS.items()}


def describe_settings(settings: dict) -> list:
    """The settings, in plain sentences."""
    s = settings
    lines = ["Torque is ON, so the jaws will fight your hand. Switching the servo power off and on turns it off."
             if s["torque_on"] else "Torque is off, so the jaws move freely by hand."]
    lines.append(f"Position limits: {s['min_position']} (closed, 0%) to {s['max_position']} (open, 100%).")
    lines.append(f"Torque cap: {s['torque_limit'] / 10:.0f}% now, {s['max_torque_limit'] / 10:.0f}% at power-up. "
                 f"P gain {s['p_gain']}.")
    if s["protection"] >> OVERLOAD & 1:
        lines.append(f"Overload protection: an output above {s['overload_torque']}% for "
                     f"{s['protection_time'] / 100:.2f} s is cut to {s['protective_torque']}% (from Feetech's "
                     "manual, not yet seen on this arm).")
    else:
        lines.append("Overload protection is off.")
    lines.append(f"Protection is on for: {sts.flag_names(s['protection'])}. Current limit {s['protection_current']} "
                 "(raw units).")
    return lines


def percent_open(position: float, settings: dict) -> float:
    """LeRobot's 0-100 for the gripper: 0 at the closed end of its position limits, 100 at the open end."""
    low, high = settings["min_position"], settings["max_position"]
    return 100 * (position - low) / (high - low) if high > low else float("nan")


def check_jaws(settings: dict, position: int, closed: int) -> None:
    """Raise ValueError if the jaws, at `position`, aren't ready for a squeeze toward `closed`: past a limit,
    where torque coming on would snap them to it, or already closed, with nothing between them."""
    low, high = settings["min_position"], settings["max_position"]
    if not low <= position <= high:
        raise ValueError(f"the jaws are at {position}, past the end of their limits ({low} to {high}), so torque "
                         "coming on would snap them to the limit. Move them in a little first")
    if abs(position - closed) < CLOSED_ALREADY:
        raise ValueError(f"the jaws are at {position}, already about closed, so there's nothing between them to "
                         "squeeze. Open them around the object first")


def check_squeeze(settings: dict, closed: int, torque_limit: int, gain: int | None = None) -> None:
    """Raise ValueError if a squeeze toward `closed` would aim past a limit, push too hard, or trip the servo's
    overload protection."""
    low, high = settings["min_position"], settings["max_position"]
    if not low <= closed <= high:
        raise ValueError(f"closed={closed} is outside the servo's position limits, {low} to {high}")
    if gain is not None and not GAIN_RANGE[0] <= gain <= GAIN_RANGE[1]:
        raise ValueError(f"the gain must be {GAIN_RANGE[0]}-{GAIN_RANGE[1]}")
    ceiling = min(TORQUE_CEILING, settings["max_torque_limit"])
    if not 0 < torque_limit <= ceiling:
        raise ValueError(f"the torque limit must be 1-{ceiling}")
    trip = settings["overload_torque"] * 10
    if settings["protection"] >> OVERLOAD & 1 and torque_limit >= trip:
        raise ValueError(f"a torque limit of {torque_limit} isn't under {trip}, where the servo's overload "
                         f"protection starts: a squeeze that hard trips it after "
                         f"{settings['protection_time'] / 100:.2f} s and changes the grip mid-measurement")


def set_gain(bus, servo_id: int, gain: int, tries: int = 25) -> None:
    """Write the servo's P gain, an EEPROM register: unlock, write, read back, lock. Raises STSError if it didn't
    take."""
    bus.write_u8(servo_id, sts.LOCK, 0)
    try:
        bus.write_u8(servo_id, sts.P_COEFFICIENT, gain)
    except sts.STSError:
        pass  # saving to EEPROM can hold up the reply; the read-back is the check
    for _ in range(tries):
        try:
            got = bus.read_u8(servo_id, sts.P_COEFFICIENT)
            bus.write_u8(servo_id, sts.LOCK, 1)
            break
        except sts.STSError:
            time.sleep(0.02)
    else:
        raise sts.STSError(f"servo {servo_id} stopped answering after its gain was written")
    if got != gain:
        raise sts.STSError(f"servo {servo_id}'s gain reads {got}, not {gain}")


def release(bus, servo_id: int, tries: int = 3) -> bool:
    """Torque off, and read back that it is. True once confirmed."""
    for _ in range(tries):
        try:
            bus.write_u8(servo_id, sts.TORQUE_ENABLE, 0)
            if bus.read_u8(servo_id, sts.TORQUE_ENABLE) == 0:
                return True
        except sts.STSError:
            pass
    return False


def run_probe(bus, servo_id: int, mode: str = "watch", closed: int | None = None, torque_limit: int = 200,
              speed: float = 300.0, hz: float = 200.0, seconds: float | None = None, settings: dict | None = None,
              gain: int | None = None, rows: list | None = None, marks: list | None = None,
              poll_marks=lambda: [], clock=time.monotonic, sleep=time.sleep, on_sample=None) -> tuple:
    """Sample until Ctrl-C or `seconds`. Returns (rows, marks), where marks are (t, label). Pass your own `rows`
    and `marks` lists to keep what was logged if the run raises. A squeeze reads `settings` from the servo
    unless given them, and refuses to start (ValueError) if check_squeeze or check_jaws does. With `gain`, it
    runs at that P gain and puts the servo's own back afterwards.

    The servo's load is its own drive, P times how far the jaw is from its goal, so it changes only when the
    jaw moves: a pull along the jaw faces that leaves the gap alone doesn't show up at all."""
    if mode not in ("watch", "squeeze"):
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "squeeze" and closed is None:
        raise ValueError("squeeze needs closed=<ticks>; find it with watch mode first")
    rows = [] if rows is None else rows
    marks = [] if marks is None else marks

    goal, restore_limit, restore_gain = None, None, None
    t0 = clock()
    try:
        if mode == "squeeze":
            settings = settings or read_settings(bus, servo_id)
            check_squeeze(settings, closed, torque_limit, gain)
            position = bus.read_feedback(servo_id).position
            check_jaws(settings, position, closed)
            restore_limit, goal = settings["torque_limit"], float(position)
            if gain is not None and gain != settings["p_gain"]:
                restore_gain = settings["p_gain"]
                set_gain(bus, servo_id, gain)
            bus.write_u16(servo_id, sts.GOAL_POSITION, position)  # hold where it is, so torque-on doesn't jump
            bus.write_u16(servo_id, sts.TORQUE_LIMIT, torque_limit)
            if bus.read_u16(servo_id, sts.GOAL_POSITION) != position or \
                    bus.read_u16(servo_id, sts.TORQUE_LIMIT) != torque_limit:
                raise sts.STSError("the hold-here goal or the torque cap didn't take, so torque stayed off")
            bus.write_u8(servo_id, sts.TORQUE_ENABLE, 1)
        period = 1.0 / hz
        next_t = last = clock()
        missed = 0
        while True:
            now = clock()
            t = now - t0
            if seconds is not None and t >= seconds:
                break
            try:
                if goal is not None and goal != closed:  # ramp toward closed; the torque limit sets the squeeze
                    step = speed * (now - last)
                    goal = min(goal + step, closed) if closed > goal else max(goal - step, closed)
                    bus.write_u16(servo_id, sts.GOAL_POSITION, int(round(goal)))
                last = now
                fb = bus.read_feedback(servo_id)
            except sts.STSError:
                missed += 1
                if missed >= MISSES:
                    raise sts.STSError(f"the servo stopped answering ({MISSES} replies in a row went missing)")
                continue
            missed = 0
            labels = list(poll_marks())
            marks.extend((t, label) for label in labels)
            row = {"t": round(t, 4), "position": fb.position, "speed": fb.speed, "load": fb.load,
                   "current": fb.current, "voltage": fb.voltage, "temperature": fb.temperature,
                   "moving": int(fb.moving), "error": fb.error,
                   "goal": "" if goal is None else int(round(goal)), "mark": "|".join(labels)}
            rows.append(row)
            if on_sample:
                on_sample(row)
            next_t += period
            delay = next_t - clock()
            if delay > 0:
                sleep(delay)
            else:
                next_t = clock()  # running behind: don't try to catch up
    except KeyboardInterrupt:
        pass
    finally:
        if restore_limit is not None:  # a squeeze got as far as writing to the servo
            release(bus, servo_id)
            try:
                bus.write_u16(servo_id, sts.TORQUE_LIMIT, restore_limit)
            except sts.STSError:
                pass  # it's in RAM, so a power cycle restores it anyway
            if restore_gain is not None:
                try:
                    set_gain(bus, servo_id, restore_gain)
                except sts.STSError:
                    pass  # LeRobot's configure() sets every gain back to 16 on its next connect anyway
    return rows, marks


def rate(rows: list) -> str:
    span = rows[-1]["t"] - rows[0]["t"] if rows else 0
    return f"{len(rows)} samples over {span:.1f} s ({(len(rows) - 1) / span:.0f} Hz)" if span > 0 \
        else f"{len(rows)} samples"


def flags(rows: list) -> list:
    """A line naming every status flag the servo raised during the run, if it raised any."""
    flagged = [r for r in rows if r.get("error")]
    if not flagged:
        return []
    status = 0
    for r in flagged:
        status |= r["error"]
    return [f"The servo flagged {sts.flag_names(status)}, first at {flagged[0]['t']:.2f} s, in {len(flagged)} "
            f"of {len(rows)} readings."]


def jaws(rows: list, marks: list, settings: dict) -> list:
    """How far the jaws went, and where they were at each mark, in ticks and percent open."""
    if not rows:
        return []

    def at(position):
        return f"{position} ({percent_open(position, settings):.1f}% open)"

    positions = [r["position"] for r in rows]
    lines = [f"The jaws went from {at(min(positions))} to {at(max(positions))}."]
    for tm, label in marks:
        row = min(rows, key=lambda r: abs(r["t"] - tm))
        lines.append(f"{label!r} at {tm:.2f} s: {at(row['position'])}")
    return lines


def summarize(rows: list, marks: list, signals=("load", "current", "position"),
              baseline_s: float = 0.5, after_s: float = 1.0, quiet_s: float = 1.0) -> list:
    """For each mark: the biggest change within after_s, as a multiple of the noise while holding still."""
    if len(rows) < 2:
        return ["not enough samples"]
    lines = [rate(rows)] + flags(rows)
    if not marks:
        return lines + ["no marks: type a label and press Enter during a run to measure a response"]

    first = marks[0][0]
    quiet = [r for r in rows if first - quiet_s - 0.2 <= r["t"] < first - 0.2]
    noise = None
    if len(quiet) >= 10:
        noise = {s: statistics.pstdev(r[s] for r in quiet) for s in signals}
        lines.append("noise while still (std): " + ", ".join(f"{s} {noise[s]:.1f}" for s in signals))
    else:
        lines.append("too few still samples before the first mark: hold still for 2 s before your first action")

    for tm, label in marks:
        before = [r for r in rows if tm - baseline_s <= r["t"] < tm]
        after = [r for r in rows if tm <= r["t"] < tm + after_s]
        if not before or not after:
            continue
        parts = []
        for s in signals:
            base = statistics.median(r[s] for r in before)
            peak = max(after, key=lambda r: abs(r[s] - base))
            change = peak[s] - base
            if noise is None:
                verdict = ""
            elif noise[s] > 0:
                ratio = abs(change) / noise[s]
                verdict = f", {ratio:.0f}x noise, {'seen' if ratio >= 5 else 'lost in noise'}"
            else:
                verdict = ", seen" if change else ", no change"
            parts.append(f"{s} {change:+.0f} after {(peak['t'] - tm) * 1000:.0f} ms{verdict}")
        lines.append(f"{label!r} at {tm:.2f} s: " + "; ".join(parts))
    return lines


def write_csv(rows: list, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path
