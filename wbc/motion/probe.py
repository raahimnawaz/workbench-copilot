"""Gripper probe: log what the gripper servo can feel, then measure each response against the noise at rest.

watch    Read only. Torque is left as it is, so you can move the jaws by hand and read off the open and
         closed positions.
squeeze  Closes slowly onto an object with a capped torque and holds it, so pulls, twists and pushes on
         the object show up in load and current. Torque is released when the run ends, however it ends.
"""
from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

from . import sts

FIELDS = ["t", "position", "speed", "load", "current", "voltage", "temperature", "moving", "error", "goal", "mark"]
MAX_TORQUE_LIMIT = 600  # per mille of max drive. The probe refuses more, so a wrong --closed can't crush anything.


def run_probe(bus, servo_id: int, mode: str = "watch", closed: int | None = None, torque_limit: int = 300,
              speed: float = 300.0, hz: float = 200.0, seconds: float | None = None,
              poll_marks=lambda: [], clock=time.monotonic, sleep=time.sleep, on_sample=None) -> tuple:
    """Sample until Ctrl-C or `seconds`. Returns (rows, marks), where marks are (t, label)."""
    if mode not in ("watch", "squeeze"):
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "squeeze":
        if closed is None:
            raise ValueError("squeeze needs closed=<ticks>; find it with watch mode first")
        if not 0 < torque_limit <= MAX_TORQUE_LIMIT:
            raise ValueError(f"torque_limit must be 1-{MAX_TORQUE_LIMIT}")

    rows, marks = [], []
    goal, restore_limit = None, None
    t0 = clock()
    try:
        if mode == "squeeze":
            restore_limit = bus.read_u16(servo_id, sts.TORQUE_LIMIT)
            goal = float(bus.read_feedback(servo_id).position)
            bus.write_u16(servo_id, sts.GOAL_POSITION, int(goal))  # hold where it is, so torque-on doesn't jump
            bus.write_u16(servo_id, sts.TORQUE_LIMIT, torque_limit)
            bus.write_u8(servo_id, sts.TORQUE_ENABLE, 1)
        period = 1.0 / hz
        next_t = last = clock()
        while True:
            now = clock()
            t = now - t0
            if seconds is not None and t >= seconds:
                break
            if goal is not None and goal != closed:  # ramp toward closed; the torque limit sets the squeeze
                step = speed * (now - last)
                goal = min(goal + step, closed) if closed > goal else max(goal - step, closed)
                bus.write_u16(servo_id, sts.GOAL_POSITION, int(round(goal)))
            last = now
            fb = bus.read_feedback(servo_id)
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
        if mode == "squeeze":
            try:
                bus.write_u8(servo_id, sts.TORQUE_ENABLE, 0)
                if restore_limit is not None:
                    bus.write_u16(servo_id, sts.TORQUE_LIMIT, restore_limit)
            except sts.STSError:
                pass  # the original error, if any, is the one worth seeing
    return rows, marks


def summarize(rows: list, marks: list, signals=("load", "current", "position"),
              baseline_s: float = 0.5, after_s: float = 1.0, quiet_s: float = 1.0) -> list:
    """For each mark: the biggest change within after_s, as a multiple of the noise while holding still."""
    if len(rows) < 2:
        return ["not enough samples"]
    span = rows[-1]["t"] - rows[0]["t"]
    lines = [f"{len(rows)} samples over {span:.1f} s ({(len(rows) - 1) / span:.0f} Hz)" if span > 0
             else f"{len(rows)} samples"]
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
