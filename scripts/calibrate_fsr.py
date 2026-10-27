"""Force sensors: check the wiring, calibrate each sensor with known weights, and measure creep.

1. Flash the sampler onto the ESP32 (once):
       pio run -d firmware/fsr_sampler -t upload --upload-port /dev/cu.usbserial-XXXX
2. Find its port. The arm's servo adapter is listed too, so run --list with the ESP32 unplugged and again
   with it plugged in: the new line is the sampler.
       python scripts/calibrate_fsr.py --list
3. watch: live readings from all four channels. Press a sensor and its number should rise.
       python scripts/calibrate_fsr.py --port /dev/cu.usbserial-XXXX watch
4. calibrate: zero one sensor, then put known weights on it one at a time and type each total in grams.
   Saves the fit to config/fsr_cal.yaml.
       python scripts/calibrate_fsr.py --port /dev/cu.usbserial-XXXX calibrate --channel SD
5. creep: leaves a weight on the sensor for two minutes and reports how far the reading drifts.
       python scripts/calibrate_fsr.py --port /dev/cu.usbserial-XXXX creep --channel SD --grams 100

Channels: SD and SP are the static jaw's distal (tip) and proximal sensors, MD and MP the moving jaw's,
on GPIO 32, 33, 34 and 35. calibrate and creep write .wbc/logs/fsr-<mode>-<time>.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wbc.config import ROOT  # noqa: E402
from wbc.reflex import fsr_cal, fsr_link  # noqa: E402
from wbc.reflex.fsr_link import CHANNELS  # noqa: E402

CAL_PATH = ROOT / "config" / "fsr_cal.yaml"


def show_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def kohms(ohms: float) -> str:
    return "open" if ohms == float("inf") else f"{ohms / 1000:.1f} kΩ"


def list_ports() -> int:
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        print("The arm extras are missing: pip install -e '.[arm]'")
        return 1
    for p in lp.comports():
        ids = f"  [{p.vid:04X}:{p.pid:04X}]" if p.vid is not None else ""
        print(f"{p.device}  {p.description}{ids}")
    return 0


def open_sampler(port: str):
    try:
        sampler = fsr_link.Sampler(port).start()
    except Exception as e:
        print(f"Can't open {port}: {e}")
        return None
    if not sampler.wait_for(50, timeout=4.0):
        print(f"No frames from {port} in 4 s. Check that the sampler firmware is flashed and that this is the "
              f"ESP32's port, not the arm's servo adapter.")
        sampler.close()
        return None
    return sampler


def average_mv(sampler, ch: int, start: float, span: float, clock=time.monotonic, sleep=time.sleep) -> tuple:
    """Mean and spread of one channel over [start, start + span) in host time. Waits until the window ends."""
    end = start + span
    while clock() < end + 0.05:
        sleep(0.02)
    frames = sampler.window(start, end)
    if len(frames) < 0.5 * span * fsr_link.RATE_HZ:
        raise IOError(f"only {len(frames)} readings in {span:g} s: is the sampler still streaming?")
    values = np.array([f.mv[ch] for f in frames], dtype=float)
    return float(values.mean()), float(values.std())


def watch(sampler, cals: dict, clock=time.monotonic, sleep=time.sleep, seconds: float | None = None) -> None:
    print("Press each sensor and watch its number rise. Unwired channels drift around. Ctrl-C to stop.")
    if cals:
        print(f"Newtons use the unloaded reading from calibration, so they may be a little off zero: "
              f"{', '.join(cals)}")
    t_start = clock()
    try:
        while seconds is None or clock() - t_start < seconds:
            sleep(0.2)
            now = clock()
            frame = sampler.latest()
            parts = []
            for i, name in enumerate(CHANNELS):
                part = f"{name} {frame.mv[i]:4d} mV"
                if name in cals:
                    part += f" {cals[name].newtons(frame.mv[i]):5.2f} N"
                parts.append(part)
            p = sampler.parser
            print(f"\r{'   '.join(parts)}   | {len(sampler.window(now - 1.0, now))} Hz, "
                  f"dropped {p.dropped}, bad {p.bad_crc}   ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    print()


def calibrate(sampler, channel: str, cal_path: Path = CAL_PATH, supply_mv: float = fsr_cal.SUPPLY_MV,
              r_fixed: float = fsr_cal.R_FIXED_OHM, ask=input, clock=time.monotonic, sleep=time.sleep) -> tuple:
    """Zero, then one reading per weight, then the fit. Returns (calibration or None, marks)."""
    ch = CHANNELS.index(channel)
    marks = []
    ask(f"Take everything off sensor {channel}, then press Enter. ")
    t = clock()
    marks.append((t, "zero"))
    zero_mv, noise = average_mv(sampler, ch, t + 0.2, 1.0, clock, sleep)
    print(f"unloaded: {zero_mv:.0f} mV (noise {noise:.1f} mV)")
    if zero_mv > 500:
        print("That's high for an unloaded sensor. Check that nothing presses on it and that the "
              f"{r_fixed / 1000:g} kΩ resistor goes from the pin to GND.")

    print("For each weight: type the total grams on the sensor, put the weight on the puck and let go, then\n"
          "press Enter right away, so every reading is taken at the same moment after loading. Count the puck\n"
          "and anything under the weight. Go from light to heavy. A blank line finishes.")
    points = []
    while True:
        line = ask("grams> ").strip()
        if not line:
            break
        try:
            grams = float(line)
        except ValueError:
            print("Type a number of grams, or press Enter on a blank line to finish.")
            continue
        if grams <= 0:
            print("The weight has to be more than 0 g.")
            continue
        t = clock()
        marks.append((t, f"{grams:g} g"))
        mv, _ = average_mv(sampler, ch, t + 0.5, 1.0, clock, sleep)
        points.append((grams, mv))
        note = ""
        if mv < fsr_cal.ADC_GOOD_MV[0]:
            note = "  (the ESP32 reads poorly below 150 mV; a bigger fixed resistor lifts light loads)"
        elif mv > fsr_cal.ADC_GOOD_MV[1]:
            note = "  (the ESP32 reads poorly above 2.45 V; a smaller fixed resistor keeps heavy loads in range)"
        print(f"  {grams:g} g: {mv:.0f} mV, sensor {kohms(fsr_cal.sensor_ohms(mv, supply_mv, r_fixed))}{note}")

    if len(points) < 2:
        print("Need at least two weights to fit a calibration. Nothing was saved.")
        return None, marks
    ordered = sorted(points)
    if any(b[1] < a[1] for a, b in zip(ordered, ordered[1:])):
        print("A heavier weight read lower than a lighter one. Check that each weight sat centered on the puck.")
    try:
        cal = fsr_cal.fit(points, zero_mv, supply_mv, r_fixed)
    except ValueError as e:
        print(f"Can't fit a calibration: {e}. Nothing was saved.")
        return None, marks

    print(f"\n{'grams':>7} {'N':>7} {'mV':>6} {'sensor':>10} {'fit N':>7} {'error':>7}")
    for grams, n, mv, ohms, fitted, err in fsr_cal.fit_rows(cal):
        if mv <= cal.zero_mv:
            print(f"{grams:7g} {n:7.3f} {mv:6.0f} {kohms(ohms):>10}   not used: it read no higher than unloaded")
        else:
            print(f"{grams:7g} {n:7.3f} {mv:6.0f} {kohms(ohms):>10} {fitted:7.3f} {err:+6.1f}%")
    print(f"fit: F = {cal.k:.4g} × G^{cal.b:.3f}, with G in µS above unloaded (R² {cal.r2:.4f} in log-log)")
    if cal.b > 0:
        mid = cal.ohms_at(1.2)
        heaviest = max(fsr_cal.grams_to_newtons(g) for g, _ in points)
        guess = (f" Those are guesses from the fit: your heaviest weight was {heaviest:.2f} N."
                 if heaviest < 3.0 else "")
        print(f"Between 0.5 and 3 N the sensor would measure {kohms(cal.ohms_at(3.0))} to "
              f"{kohms(cal.ohms_at(0.5))}.{guess} The maker suggests a fixed resistor of a third to half the "
              f"sensor's resistance in the working range: about {kohms(mid / 3)} to {kohms(mid / 2)} (at 1.2 N). "
              f"You're using {kohms(r_fixed)}.")

    cals = fsr_cal.load(cal_path)
    cals[channel] = cal
    fsr_cal.save(cals, cal_path)
    print(f"saved {channel} to {show_path(cal_path)}")
    return cal, marks


def show_until(sampler, ch: int, end: float, cal, zero_mv: float, clock=time.monotonic, sleep=time.sleep) -> None:
    while clock() < end:
        sleep(min(0.5, max(end - clock(), 0.0)))
        frame = sampler.latest()
        value = f"   {cal.newtons(frame.mv[ch], zero_mv=zero_mv):.3f} N" if cal else ""
        print(f"\r  {max(end - clock(), 0):4.0f} s left   {frame.mv[ch]:4d} mV{value}   ", end="", flush=True)
    print()


def creep(sampler, channel: str, grams: float, seconds: float, after: float, cal=None,
          supply_mv: float = fsr_cal.SUPPLY_MV, r_fixed: float = fsr_cal.R_FIXED_OHM,
          clock=time.monotonic, sleep=time.sleep) -> tuple:
    """Unloaded for 3 s, loaded for `seconds`, unloaded for `after`. Returns (report or None, frames, marks)."""
    ch = CHANNELS.index(channel)
    marks = []
    t0 = clock()
    print(f"Take everything off sensor {channel} and don't touch it. Recording.")
    while clock() < t0 + 3.0:
        sleep(0.05)
    zero_mv, _ = average_mv(sampler, ch, t0 + 0.5, 2.0, clock, sleep)
    marks.append((clock(), "place"))
    print(f"Place the {grams:g} g weight now, gently, and let go. Leave it for {seconds:.0f} s.")
    show_until(sampler, ch, clock() + seconds, cal, zero_mv, clock, sleep)
    marks.append((clock(), "remove"))
    print(f"Lift the weight off now and leave the sensor bare for {after:.0f} s.")
    show_until(sampler, ch, clock() + after, cal, zero_mv, clock, sleep)

    frames = sampler.window(t0, clock() + 1.0)
    t = fsr_link.unwrap_seconds([f.t_us for f in frames])
    try:
        report = fsr_cal.analyze_creep(t, [f.mv[ch] for f in frames], cal=cal, grams=grams,
                                       supply_mv=supply_mv, r_fixed=r_fixed)
    except ValueError as e:
        print(f"Couldn't measure creep: {e}")
        return None, frames, marks
    print()
    for line in report.lines():
        print(line)
    if cal is None:
        print(f"Sensor {channel} isn't calibrated, so the drift is in conductance (µS), which tracks force "
              f"roughly in proportion. Calibrate it to get newtons.")
    return report, frames, marks


def write_log(frames: list, marks: list, mode: str, folder: Path = ROOT / ".wbc" / "logs") -> Path:
    path = folder / f"fsr-{mode}-{datetime.now():%Y%m%d-%H%M%S}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    t = fsr_link.unwrap_seconds([f.t_us for f in frames])
    pending = sorted(marks)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "seq", *CHANNELS, "mark"])
        for ti, f in zip(t, frames):
            labels = []
            while pending and pending[0][0] <= f.host_t:
                labels.append(pending.pop(0)[1])
            w.writerow([f"{ti:.4f}", f.seq, *f.mv, "|".join(labels)])
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    ap.add_argument("--port", help="the sampler ESP32, e.g. /dev/cu.usbserial-0001")
    ap.add_argument("--supply-mv", type=float, default=fsr_cal.SUPPLY_MV,
                    help="the ESP32's 3V3 rail in mV; measure it with a meter if you can (default 3300)")
    ap.add_argument("--r-fixed", type=float, default=fsr_cal.R_FIXED_OHM,
                    help="the divider's fixed resistor in ohms (default 10000)")
    sub = ap.add_subparsers(dest="mode")
    sub.add_parser("watch", help="live readings from all four channels")
    ca = sub.add_parser("calibrate", help="fit one sensor with known weights")
    ca.add_argument("--channel", choices=CHANNELS, required=True)
    cr = sub.add_parser("creep", help="leave a weight on one sensor and measure the drift")
    cr.add_argument("--channel", choices=CHANNELS, required=True)
    cr.add_argument("--grams", type=float, default=100, help="the weight's total grams (default 100)")
    cr.add_argument("--seconds", type=float, default=120, help="how long to leave it on (default 120)")
    cr.add_argument("--after", type=float, default=30, help="how long to record after lifting it off (default 30)")
    args = ap.parse_args(argv)

    if args.list:
        return list_ports()
    if not args.port or not args.mode:
        ap.error("give --port and a mode (watch, calibrate or creep); use --list to find the port")

    sampler = open_sampler(args.port)
    if sampler is None:
        return 1
    log = None
    try:
        if args.mode == "watch":
            watch(sampler, fsr_cal.load(CAL_PATH))
        elif args.mode == "calibrate":
            start = sampler.latest().host_t
            _, marks = calibrate(sampler, args.channel, CAL_PATH, args.supply_mv, args.r_fixed)
            log = write_log(sampler.window(start, float("inf")), marks, "calibrate")
        else:
            cal = fsr_cal.load(CAL_PATH).get(args.channel)
            _, frames, marks = creep(sampler, args.channel, args.grams, args.seconds, args.after, cal,
                                     args.supply_mv, args.r_fixed)
            log = write_log(frames, marks, "creep")
    except KeyboardInterrupt:
        print("\nstopped")
    except IOError as e:
        print(f"\n{e}")
        return 1
    finally:
        sampler.close()
    p = sampler.parser
    print(f"{p.frames} frames, {p.dropped} dropped, {p.bad_crc} bad CRC" + (f"; log: {show_path(log)}" if log else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
