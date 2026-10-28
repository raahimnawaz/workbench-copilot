"""Measure what the gripper servo can feel: position, load and current, logged a few hundred times a second.

1. Find the port, with the arm's servo power on:
       python scripts/gripper_probe.py --list
2. watch: read only. Move the jaws by hand (torque off) and press Enter to mark where they are, for example
   closed, fully open, and closed on each object. A typed label is optional; the live line hides what you type.
       python scripts/gripper_probe.py --port /dev/cu.usbmodem5B910297421 watch
3. squeeze: it waits while you open the jaws around an object, then on Enter closes slowly toward --closed
   with a capped torque, and holds. Let go, wait 2 s, then pull, twist or push the object, pressing Enter as
   you start each action. To stop, hold the object and press Ctrl-C: the gripper lets go.
       python scripts/gripper_probe.py --port /dev/cu.usbmodem5B910297421 squeeze --closed 1278

Both start by printing the gripper's own settings: its position limits from lerobot-calibrate, its torque cap,
and its overload protection. A squeeze refuses a --closed outside those limits, and a torque limit at or above
where the protection starts (LeRobot sets 25% on the gripper), so the grip can't change partway through.
Each run writes .wbc/logs/gripper-<time>.csv and prints where the jaws were at each mark, and for a squeeze,
how far each action rose above the noise. Close LeRobot first: only one program can use the servo bus at a time.
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wbc.config import ROOT  # noqa: E402
from wbc.motion import probe, sts  # noqa: E402


def list_ports() -> int:
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        print("The arm extras are missing: pip install -e '.[arm]'")
        return 1
    for p in lp.comports():
        print(f"{p.device}  {p.description}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    ap.add_argument("--port", help="servo bus adapter, e.g. /dev/cu.usbmodem5B910297421")
    ap.add_argument("--id", type=int, default=6, help="servo id (the SO-101 gripper is 6)")
    ap.add_argument("--hz", type=float, default=200)
    ap.add_argument("--seconds", type=float, help="stop after this long instead of waiting for Ctrl-C")
    sub = ap.add_subparsers(dest="mode")
    sub.add_parser("watch", help="read only")
    sq = sub.add_parser("squeeze", help="close onto an object with capped torque, then hold")
    sq.add_argument("--closed", type=int, required=True,
                    help="position to close toward: the closed end of the limits, or where watch saw the jaws touch")
    sq.add_argument("--torque-limit", type=int, default=200,
                    help=f"per mille of max drive, at most {probe.TORQUE_CEILING} and under where the servo's "
                         "overload protection starts (default 200)")
    sq.add_argument("--speed", type=float, default=300, help="closing speed in ticks/s (default 300)")
    sq.add_argument("--gain", type=int,
                    help=f"P gain for this squeeze, {probe.GAIN_RANGE[0]}-{probe.GAIN_RANGE[1]}, put back afterwards. "
                         "The push grows with the gain times how far the goal is past the object: at LeRobot's 16, "
                         "a board 37 ticks from closed got only 14%%, so 32 lets the torque limit set the squeeze")
    args = ap.parse_args(argv)

    if args.list:
        return list_ports()
    if not args.port or not args.mode:
        ap.error("give --port and a mode (watch or squeeze); use --list to find the port")

    try:
        bus = sts.Bus(args.port)
    except Exception as e:
        print(f"Can't open {args.port}: {e}")
        return 1
    rows, marks, settings, started = [], [], None, False
    try:
        if not bus.ping(args.id):
            print(f"Servo {args.id} didn't answer. Check that the servo power is on and the cable chain "
                  f"reaches it.")
            return 1
        settings = probe.read_settings(bus, args.id)
        print("The gripper's settings:")
        for line in probe.describe_settings(settings):
            print("  " + line)
        if args.mode == "squeeze":
            try:
                probe.check_squeeze(settings, args.closed, args.torque_limit, args.gain)
            except ValueError as e:
                print(f"Not squeezing: {e}.")
                return 1
            while True:  # wait for the object, however long that takes
                if input("\nOpen the jaws around the object and hold it there, fingers clear of the jaw faces. "
                         "Press Enter to close (q = quit): ").strip().lower().startswith("q"):
                    return 1
                try:
                    probe.check_jaws(settings, bus.read_feedback(args.id).position, args.closed)
                    break
                except ValueError as e:
                    print(f"Not yet: {e}.")

        labels = queue.Queue()

        def read_labels() -> None:
            for n, line in enumerate(sys.stdin, 1):
                labels.put(line.strip() or f"mark {n}")

        threading.Thread(target=read_labels, daemon=True).start()

        def poll_marks():
            out = []
            while not labels.empty():
                label = labels.get_nowait()
                if label.lower() == "q":
                    raise KeyboardInterrupt  # q and Enter stops a run the same way Ctrl-C does
                out.append(label)
            return out

        shown = [0.0]

        def show(row) -> None:
            pct = probe.percent_open(row["position"], settings)
            if row["mark"]:
                print(f"\r  marked {row['mark']!r} at {row['t']:.2f} s: position {row['position']} "
                      f"({pct:.1f}% open)".ljust(90))
            if row["t"] - shown[0] >= 0.2:
                shown[0] = row["t"]
                goal = f"  goal {row['goal']:5d}" if row["goal"] != "" else ""
                flagged = f"  FLAGS {sts.flag_names(row['error'])}" if row["error"] else ""
                print(f"\r t {row['t']:7.2f}  pos {row['position']:5d} ({pct:5.1f}% open){goal}  load {row['load']:+5d}"
                      f"  current {row['current']:+5d}  {row['temperature']} °C{flagged}   ", end="", flush=True)

        if args.mode == "squeeze":
            at_gain = f" at P gain {args.gain}" if args.gain is not None else ""
            print(f"Closing toward {args.closed} with torque limit {args.torque_limit}{at_gain}. Once it holds, let go "
                  "and keep your hands off for 2 s. Then press Enter as you start each action. To stop, hold the "
                  "object and press Ctrl-C (or q, Enter): the gripper lets go.")
        else:
            print("\nWatching (read only). Press Enter to mark where the jaws are. Ctrl-C (or q, Enter) to stop.")

        started = True
        probe.run_probe(
            bus, args.id, args.mode,
            closed=getattr(args, "closed", None), torque_limit=getattr(args, "torque_limit", 200),
            speed=getattr(args, "speed", 300), hz=args.hz, seconds=args.seconds, settings=settings,
            gain=getattr(args, "gain", None), rows=rows, marks=marks, poll_marks=poll_marks, on_sample=show)
    except sts.STSError as e:
        print(f"\nStopped: {e}")
    except ValueError as e:  # the jaws moved past a limit between the check above and the squeeze's own
        print(f"\nNot squeezing: {e}.")
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    finally:
        if args.mode == "squeeze" and started:
            print("\nThe gripper is limp." if probe.release(bus, args.id) else
                  "\nCouldn't confirm the gripper let go. If its jaws are still stiff, switch the servo power off.")
        bus.close()

    if not rows:
        return 1
    print()
    path = probe.write_csv(rows, ROOT / ".wbc" / "logs" / f"gripper-{datetime.now():%Y%m%d-%H%M%S}.csv")
    lines = probe.summarize(rows, marks) if args.mode == "squeeze" else [probe.rate(rows)] + probe.flags(rows)
    for line in lines + probe.jaws(rows, marks, settings):
        print(line)
    print(f"log: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
