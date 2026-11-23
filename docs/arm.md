# The arm

The SO-101 driver, its safety checks, and the pick and put skills built on it. Back to the [README](../README.md).

## The arm

`wbc/motion/arm.py` is the driver, and the only code that talks to the servo bus while the agent runs. `Arm.open()` connects LeRobot's bus and nothing more: never `robot.connect()`, which can recalibrate the arm and always ends by turning torque on. `power_on()` brings the joints up one at a time from the base out, and `move_to(q)` moves all five together to URDF angles q.

- **Refused before anything moves:** a joint past an end of its calibrated range, the wrist roll near where its reading wraps, a servo at 65 °C, a target within 2° of a range's end, and a path that takes the jaw tips under their floor or the elbow and wrist axes under 30 mm above the desk. A refusal says which joint and which way to turn it.
- **Speed:** moves are quintic and peak at 45°/s on any joint and 180 mm/s at the jaw tips, and no goal goes more than 5° from where its joint is. Torque is capped at 50% on the arm joints and on the gripper (LeRobot's own ceiling for it; the gripper's overload threshold, which LeRobot sets to 25%, goes to 60% while it's powered, or a 50% squeeze would drop to 20% after 2 s), and a P of 32 on every joint but the wrist roll goes back on after LeRobot's setup at every power-on.
- **Cuts:** every motor is cut at once if a joint jumps 3° as its torque comes on, strays 5° from its goal during a move, sags 5 mm under a floor, or the bus fails. `cut()` works from any thread or a signal handler, every cut is read back servo by servo, and a cut latches until `clear_stop()`. `halt()`, the spoken stop, stops a move at its next tick and holds the arm there with its motors on.
- **Re-aim:** each joint stops short of its goal by its load over its P gain, so after a move's path `move_to()` sets each goal past its target by the shortfall, under the same caps and checks, until every joint is within 0.2°.
- **Gripper:** `grip()` closes on what's between the jaws and keeps squeezing under the cap, and says whether they stopped on something. `release()` opens them, stopping where something's in the way.

`wbc/motion/hand.py` builds the agent's arm skills on the driver. A pick goes straight down beside a thing's grip, 1 cm from its middle toward its narrow end (on a screwdriver, the handle near where it balances), with the jaws closing across its long axis, the fixed jaw beside it and the moving jaw closing it against the fixed one: over it, open only as wide as it needs, just over its top, down, grip, and back up. These jaws hold a handle in the gap above their tips, reading as shut as empty ones, so a pick goes back up either way and says when it can't tell. A put lines what's held up with the place's long axis, goes down to just over its top, opens as wide as the pick did, and closes the jaws again back up. A put into a container (`hand.containers` in `config/robot.yaml`, with its measured height) stands what's held in on end instead: lifted with the jaws turned across the arm, so turning the gripper up stands it on end over the opening, its narrow end first where the wrist roll can turn it that way, then down until that end is 15 mm inside the rim, and let go. The gripper reaches level low enough from about 25–28 cm out. Every pose a pick or a put needs is solved before the first one moves, so one the arm can't reach is refused with the arm where it was. Heights come from the desk as the arm's touches measured it (`wbc/motion/table.py`). Between requests, empty-handed, the arm waits in an idle pose with its motors on: the L pose with the gripper pointing straight down (`hand.idle_deg` in `config/robot.yaml`). The end of a session takes it back to where it sat when its motors came on, since the L pose can't hold itself up unpowered, and then the motors go off.

```bash
python scripts/check_joint_map.py --port /dev/cu.usbmodemXXXX check   # motors off: the joint map's signs, by hand
python scripts/register_arm.py --port /dev/cu.usbmodemXXXX touch       # motors off: register the arm to the carrier
```

- The joint map is q = sign · (raw − 2047) · 2π / 4096, with every sign +1: lerobot-calibrate's L pose is the URDF's zero. `check_joint_map.py` repeats the hand test, so run it again after a recalibration or a servo swap. It only reads, and won't run with a motor on.
- Keep to one program per serial port: LeRobot opens the port without locking it, so two would interleave their packets.
- In the URDF the gripper approaches along its z-axis, and `config/robot.yaml` turns the tool frame so the approach is x and the tool point is the closed jaws' tips.
