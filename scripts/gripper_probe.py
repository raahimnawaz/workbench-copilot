"""Measure what the gripper servo can feel: position, load and current, logged a few hundred times a second.

1. Find the port, with the arm's servo power on:
       python scripts/gripper_probe.py --list
2. watch: read only. Move the jaws by hand (torque off) and note the open and closed positions.
       python scripts/gripper_probe.py --port /dev/tty.usbmodemXXXX watch
3. squeeze: put an object between the jaws. The gripper closes slowly toward --closed with a capped
   torque, then holds. Wait 2 s, then pull, twist or push the object. Type a label (e.g. "pull") and
   press Enter as you start each action. Ctrl-C ends the run and releases the gripper.
       python scripts/gripper_probe.py --port /dev/tty.usbmodemXXXX squeeze --closed 2150

Each run writes .wbc/logs/gripper-<time>.csv and prints how far each action rose above the noise.
Close LeRobot first: only one program can use the servo bus at a time.
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
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
    ap.add_argument("--port", help="servo bus adapter, e.g. /dev/tty.usbmodem58760431541")
    ap.add_argument("--id", type=int, default=6, help="servo id (the SO-101 gripper is 6)")
    ap.add_argument("--hz", type=float, default=200)
    ap.add_argument("--seconds", type=float, help="stop after this long instead of waiting for Ctrl-C")
    sub = ap.add_subparsers(dest="mode")
    sub.add_parser("watch", help="read only")
    sq = sub.add_parser("squeeze", help="close onto an object with capped torque, then hold")
    sq.add_argument("--closed", type=int, required=True, help="position to close toward, from watch mode")
    sq.add_argument("--torque-limit", type=int, default=300,
                    help=f"per mille of max drive, 1-{probe.MAX_TORQUE_LIMIT} (default 300)")
    sq.add_argument("--speed", type=float, default=300, help="closing speed in ticks/s (default 300)")
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
    try:
        if not bus.ping(args.id):
            print(f"Servo {args.id} didn't answer. Check that the servo power is on and the cable chain "
                  f"reaches it.")
            return 1

        labels = queue.Queue()

        def read_labels() -> None:
            for line in sys.stdin:
                labels.put(line.strip() or "mark")

        threading.Thread(target=read_labels, daemon=True).start()

        def poll_marks():
            out = []
            while not labels.empty():
                out.append(labels.get_nowait())
            return out

        shown = [0.0]

        def show(row) -> None:
            if row["t"] - shown[0] >= 0.2:
                shown[0] = row["t"]
                err = f"  ERROR 0x{row['error']:02x}" if row["error"] else ""
                print(f"\r t {row['t']:7.2f}  pos {row['position']:5d}  load {row['load']:+5d}  "
                      f"current {row['current']:+5d}  {row['temperature']} °C{err}   ", end="", flush=True)

        if args.mode == "squeeze":
            print(f"Closing toward {args.closed} with torque limit {args.torque_limit}. "
                  "Hold still 2 s, then type a label and press Enter as you start each action. Ctrl-C to stop.")
        else:
            print("Watching (read only). Type a label and press Enter to mark the log. Ctrl-C to stop.")

        rows, marks = probe.run_probe(
            bus, args.id, args.mode,
            closed=getattr(args, "closed", None), torque_limit=getattr(args, "torque_limit", 300),
            speed=getattr(args, "speed", 300), hz=args.hz, seconds=args.seconds,
            poll_marks=poll_marks, on_sample=show)
    finally:
        bus.close()

    print()
    path = probe.write_csv(rows, ROOT / ".wbc" / "logs" / f"gripper-{datetime.now():%Y%m%d-%H%M%S}.csv")
    for line in probe.summarize(rows, marks):
        print(line)
    print(f"log: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
