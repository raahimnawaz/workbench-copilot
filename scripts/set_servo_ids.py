"""Give each SO-101 servo its id (1-6) on an arm that's already assembled.

New STS3215 servos all leave the factory as id 1, so on a chained arm they all answer at once and nothing
is readable. LeRobot's lerobot-setup-motors fixes that one servo at a time, but it wants the board moved
to each servo in turn. This script grows the chain outward from the base instead, so the cable between
the board and the base servo never moves:

1. Only the base servo is connected (unplug the base -> shoulder lift cable at the shoulder lift end).
   It's parked on a spare id while the others join.
2. For each next servo: unplug the cable leading on from it (at the far end), plug in the cable that
   reaches it from the previous servo, and it joins as id 1 and gets its real id.
3. The base goes back to id 1.

    python scripts/set_servo_ids.py --port /dev/cu.usbmodemXXXX

It only writes ids, never moves a joint (torque stays off). If it stops partway, run it again: it
picks up where it left off.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wbc.motion import sts  # noqa: E402

JOINTS = [(1, "base (shoulder pan)"), (2, "shoulder lift"), (3, "elbow"), (4, "wrist flex"),
          (5, "wrist roll"), (6, "gripper")]
PARK_ID = 20  # where the base waits while each new servo joins at the factory id 1


def answering(bus) -> set:
    return {i for i in [*range(1, 7), PARK_ID] if bus.ping(i)}


def run(bus, ask=input, say=print) -> bool:
    ids = answering(bus)
    if set(range(1, 7)) <= ids:
        say("All six servos already answer at ids 1-6. Nothing to do.")
        return True
    if PARK_ID not in ids:
        if ids != {1}:
            say(f"Expected only the base servo, at id 1, but these ids answer: {sorted(ids) or 'none'}.\n"
                "Unplug the base -> shoulder lift cable at the shoulder lift end, so only the base is on "
                "the line, then run this again.")
            return False
        bus.set_id(1, PARK_ID)
        say(f"Base servo parked on id {PARK_ID} while the others join.")

    for k in range(2, 7):
        name, prev = JOINTS[k - 1][1], JOINTS[k - 2][1]
        if bus.ping(k):
            say(f"The {name} servo already answers as id {k}.")
            continue
        steps = ["Switch the servo power off."]
        if k < 6:
            steps.append(f"Unplug the cable from the {name} servo to the {JOINTS[k][1]} servo, "
                         f"at the {JOINTS[k][1]} end.")
        steps.append(f"Plug the loose cable from the {prev} servo into the {name} servo.")
        steps.append("Switch the power back on.")
        say(f"\nNext: the {name} servo (id {k}).\n" +
            "\n".join(f"  {i}. {s}" for i, s in enumerate(steps, 1)))
        while True:
            if ask("Press Enter when that's done (or type q to stop): ").strip().lower() == "q":
                say("Stopped. Run this again to carry on from here.")
                return False
            if bus.ping(1):
                break
            say("No servo answers at id 1 yet. Check that cable is pushed all the way in and the power is on, "
                "and that the cable leading on from it is unplugged (two new servos at once can't be told apart).")
        bus.set_id(1, k)
        say(f"The {name} servo is now id {k}.")

    bus.set_id(PARK_ID, 1)
    say("\nBase servo back on id 1. Every servo:")
    for sid, name in JOINTS:
        fb = bus.read_feedback(sid)
        flags = f", error flags 0x{fb.error:02x}" if fb.error else ""
        say(f"  id {sid}  {name:20s} position {fb.position:4d}  {fb.voltage:.1f} V  {fb.temperature} °C{flags}")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", required=True, help="the arm's servo adapter, e.g. /dev/cu.usbmodem5B910297421")
    args = ap.parse_args(argv)
    try:
        bus = sts.Bus(args.port)
    except Exception as e:
        print(f"Can't open {args.port}: {e}")
        return 1
    try:
        return 0 if run(bus) else 1
    except KeyboardInterrupt:
        print("\nStopped. Run this again to carry on from here.")
        return 1
    finally:
        bus.close()


if __name__ == "__main__":
    sys.exit(main())
