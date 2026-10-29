# Workbench Co-Pilot

[![tests](https://github.com/raahimnawaz/workbench-copilot/actions/workflows/tests.yml/badge.svg)](https://github.com/raahimnawaz/workbench-copilot/actions/workflows/tests.yml)

An AI lab partner that lives in your code editor and on your workbench. You talk to it hands free: say "Copilot," and your request. **Muse Spark** routes each request: **Muse Wonder** writes code, **SAM 3.1** reads the breadboard, **Muse Voice Transcribe** hears you, and an **SO-101** arm points, grasps and holds boards while you probe.

Built for the Meta Global AI Developer Hackathon (Indie track, Push the Frontier), Oct 26 – Nov 4, 2026.

## Built during the hackathon, and what it builds on

Everything in this repository was written during the hackathon, starting Oct 26, 2026, and the commit history dates every change. The one exception is the vendored URDF listed below.

| Built here | What it does |
|---|---|
| `wbc/agent/` | Router loop, tool registry, `ask_coder`, voice front end, budget guard, console HUD |
| `wbc/motion/kinematics.py` | URDF chain loader, forward kinematics, damped least-squares IK, the (x, y, z, pitch, roll) grasp frame |
| `wbc/motion/sts.py`, `wbc/motion/probe.py` | A small STS3215 servo client and the gripper feedback probe |
| `wbc/motion/joints.py`, `wbc/motion/trajectory.py` | The joint map from raw servo ticks to the URDF's joint angles, with signs checked by hand on the arm, and quintic joint moves timed under joint and tip speed caps |
| `wbc/motion/arm.py` | The arm driver, the one owner of the servo bus: it powers the arm on without a jump and moves all five joints together in URDF angles, refusing before anything moves and cutting every motor when a joint strays; it re-aims joints that stop short of their goals, and opens, closes and grips with the gripper |
| `scripts/set_servo_ids.py` | Gives each servo of an already-assembled arm its id, growing the chain from the base |
| `scripts/first_moves.py` | The arm's first powered moves: torque on without a jump, then each joint a little, recording which way it turns |
| `scripts/power_cut.py` | The power cut-off test: switching the servo supply off makes the arm let go, and it stays limp when the power returns |
| `scripts/check_joint_map.py` | A read-only check of the joint map with the motors off: which way each joint's URDF angle goes as you turn it by hand |
| `wbc/motion/carrier.py`, `scripts/register_arm.py` | Registering the arm to the carrier from points touched by hand with the motors off, the touch test, and the pose and two moves that point the gripper at a breadboard hole |
| `wbc/perception/desk.py`, `scripts/desk_camera.py` | The desk camera: tags touched by the arm map a second camera's view of the desk into the arm's frame, and `locate` finds a named thing there with SAM 3.1, giving its middle, size and heading, corrected for its height |
| `wbc/reflex/fsr_link.py`, `wbc/reflex/fsr_cal.py` | The force sensor link (500 Hz frames with a CRC), per-sensor calibration fit and creep analysis |
| `wbc/perception/` | The SAM 3.1 client (one noun phrase per request, sent in parallel; masks decoded and drawn), camera capture, and the board reader: the carrier's tags, the top-down board view, the hole grid fit, wire tracing, the netlist and its check against `pins.h` |
| `scripts/sam_probe.py`, `scripts/read_board.py`, `scripts/make_carrier.py` | What SAM 3.1 finds in a still; the board reader on a still; the printable carrier sheet |
| `config/`, `fixtures/`, `scripts/`, `tests/` | Model routing config, the recorded bench for `--sim`, the smoke test, gripper probe and force sensor tools, 269 tests |
| `firmware/dut_demo/` | The Act 1 demo board firmware, with the wiring bug planted on purpose |
| `firmware/fsr_sampler/` | ESP32 firmware that samples the four jaw sensors at 500 Hz and streams them over USB |

| Reused | License | Used for |
|---|---|---|
| `hardware/so101_new_calib.urdf`, from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) | Apache-2.0 | The arm's kinematic chain (unchanged copy) |
| [LeRobot](https://github.com/huggingface/lerobot) | Apache-2.0 | Servo calibration and the Feetech bus (from Oct 27) |
| Muse Spark, Muse Wonder, Muse Voice Transcribe, SAM 3.1 on the Meta Model API | Meta API terms | Routing, coding, speech, segmentation |
| [meta-sam-parser](https://pypi.org/project/meta-sam-parser/) 0.1, from Meta | SAM License | Reading SAM 3.1's replies: the token grammar and the mask codec (installed from PyPI, not vendored) |
| [AprilTag](https://april.eecs.umich.edu/software/apriltag)'s tag36h11 family, as built into OpenCV's ArUco module | BSD-2-Clause | The carrier's tags |
| Muse Glimmer 30B weights | Apache-2.0 | The local backup router |
| openai, numpy, PyYAML, httpx, sounddevice, pyserial, pytest, OpenCV, Pillow | Apache-2.0, BSD-3, MIT, BSD-3, MIT, BSD-3, MIT, Apache-2.0, MIT-CMU | Python dependencies |
| [PlatformIO](https://platformio.org) and the [Arduino core for ESP32](https://github.com/espressif/arduino-esp32) | Apache-2.0, LGPL-2.1 | Building and flashing both firmware projects |

## Status, Oct 29 (Day 4)

| Piece | State |
|---|---|
| Router loop on Muse Spark (Responses API) | Runs live. On Oct 28 a spoken Act 1 on the recorded bench took about 63 s in four router rounds. Code now fetches the bench facts first and stages checked patches itself, and the same turn takes 25 s in two rounds. |
| Backup router on Muse Glimmer (Chat Completions) | Built and tested against a fake API. |
| `ask_coder` on Muse Wonder | Runs live: Wonder wrote the Act 1 fix in 3–7 s on Oct 28. The fix compiles in a sandbox copy in about 8 s once its build cache is warm (about 14 s cold). |
| Voice front end | The request format was checked against the live API on Oct 26. The wake phrase and the loudness gate that cuts the mic into utterances are tested on synthetic audio, and on Oct 28 they worked on the live mic (the Razer, run from Terminal.app). |
| Budget guard | Built and tested. Spend persists in `.wbc/spend.json`. |
| FK/IK | Tested on the real SO-101 URDF. 200 of 200 bench poses round-trip to within 0.2 mm and 0.5°, at about 33 ms per solve. |
| Joint map (Oct 28) | Checked by hand on the real arm with the motors off. Raw rising turns all five joints the URDF's positive way, and the arm's L pose reads within 2° of the URDF's zero on the lift, elbow and wrist flex. 46 offline tests cover the map and the timed moves, and run those five moves through FK so a sign can't flip back unnoticed. |
| Arm driver and joint map check (Oct 29) | Built and tested offline: the driver against a fake LeRobot follower (power-on order, every refusal, the cuts on a jump, a stray, a sag under a floor or a bus failure, Ctrl-C, a cut or close from another thread or a signal handler, the stop latch, the cut's read-back), the check against fake servos with the Oct 28 hand-test numbers. The driver's first powered run (Oct 29, small moves from the rest pose) was clean; its re-aim and gripper control are tested offline only. |
| STS3215 client and gripper probe | Run on the arm: `set_servo_ids.py` gave the six servos ids 1–6 (Oct 27), and the gripper probe logged position, load and current while a board in the jaws was pulled, pushed and twisted (Oct 28). |
| Force sensor sampler and tools (Oct 27) | The firmware runs on the ESP32: 500.0 Hz, no dropped frames and no CRC errors. On a real sensor, creep measured +6.8% over 3 s and +28% over 60 s under 51 g. Calibration is rough so far (coins up to 51 g, errors up to 27%), and readings at 34 g and 51 g came out about the same. That's an open question to settle with heavier weights before the jaw pads. Deferred to the end of the build on Oct 29. |
| SAM 3.1 client (Oct 28) | Built to Meta's documented request and reply format and tested offline on replies written with Meta's own mask encoder. Called live since Oct 28: "wire" finds each flat wire (scores 0.61–0.88) and "breadboard" the board, in 1.1–1.6 s a request, while "jumper wire" finds nothing. |
| Board reader (Oct 28) | Built and tested on a synthetic bench photo: the carrier's tags, the top-down view, the hole grid fit, wire tracing, the netlist and the check against `pins.h`. On the first real photo (Oct 28) it found all three wires, and with a 1 mm tip inset all six ends land on their true holes, though with a thin margin. On the wired LED board (Oct 28 evening) it traced the LED's jumper from a5 and reported it on GPIO 4 while `pins.h` says 5: the planted bug, found on the real bench. |
| Act 1 demo board (Oct 28–29) | On Oct 28, an LED: the reader traced its jumper on the real board and reported GPIO 4 against `pins.h`'s 5. On Oct 29 it became a force sensor circuit: the firmware prints a reading every 20 ms and the pin it reads, `read_sensor` reads a few seconds of them live, and `build_and_flash` uploads the fixed firmware, so the co-pilot can check its own fix under your finger. Built and tested offline; the sensor isn't wired yet. The camera keeps the sharpest frame from 1.5–3 s after opening. |
| Desk camera and `locate` (Oct 29) | Built and tested offline on a synthetic camera: the table maps through the touched tags within 0.01 mm, a badly touched tag is named, and an upright cup's middle lands within 3 mm given its height. Also tested on tags rendered into a frame and found by the real detector. Not yet run on the real desk camera, and not yet a router tool. |
| Pointing, probe hold, web HUD, spoken replies | Not started; next in the Oct 29 plan. The tools return a clear "not wired yet" error, or use the recorded bench with `--sim`. |

**The plan as of Oct 29** ([blueprint](https://claude.ai/artifact/1f5AMxFTVycA2C25DkqWwm)): Act 1 becomes debugging a force sensor circuit on the carrier board, where the co-pilot reads live samples, finds the wire on the wrong GPIO, fixes `pins.h`, points at the wire with the arm, and checks the reading itself after you flash. Act 2 is the arm holding your multimeter's ground probe on the rail. Act 3, a stretch, is a skill the co-pilot writes for itself. Replies are spoken. The force-sensing jaw pads and the slip catch move to the end.

## Setup (Apple Silicon)

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e '.[voice,dev]'
cp .env.example .env        # then put your META_API_KEY in .env
pytest
python scripts/smoke_test.py
```

The smoke test makes one small call to each hosted model and reports which model ids your key can see. Run it before anything else.

## Run

```bash
python -m wbc.agent --sim
```

Then type `My force sensor isn't reading anything. Check my wiring against the firmware.` With `--sim`, `read_breadboard` returns the recorded Act 1 scene from `fixtures/`, `read_sensor` answers a press only once firmware reading GPIO 4 has been "flashed", and `point_at` and `build_and_flash` pretend. That exercises the real router and the real Wonder coder with no hardware attached.

- `--voice` also listens on the bench mic (`ears.input_device`, the Razer Seiren Mini). Say "Copilot," and your request, or "Copilot." and then the request. Every utterance is transcribed, and only the ones addressed to the co-pilot go to the router. `--hear` is a mic check: it prints what it heard and routes nothing. The first time, macOS asks to let your terminal use the microphone.
- `--backup` routes with Muse Glimmer on your own machine. Serve it with any OpenAI-compatible server that supports tool calls (llama.cpp's `llama-server`, LM Studio or Ollama), then set `backup.base_url` and `backup.model` in `config/models.yaml`. The 4-bit build fits easily in 64 GB.

## Servo ids on an assembled arm

New STS3215 servos all leave the factory as id 1. On a chained arm they all answer at once, so nothing reads cleanly. Their identical ping replies sometimes overlap well enough to pass, which makes the fault look like a flaky cable. LeRobot's `lerobot-setup-motors` fixes this one servo at a time, but it moves the board to each servo in turn. `scripts/set_servo_ids.py` grows the chain outward from the base instead, so the board → base cable never moves:

```bash
uv pip install -e '.[arm]'
python scripts/set_servo_ids.py --port /dev/cu.usbmodemXXXX
```

1. Start with only the base servo connected: unplug the base → shoulder lift cable at the shoulder lift end.
2. The script parks the base on id 20. Then, for each next servo, it tells you to unplug the cable leading on from that servo (at the far end) and plug in the one reaching it. The servo joins alone as id 1 and gets its real id (2–6).
3. At the end the base returns to id 1, and the script lists all six.

Before changing an id, it checks that five full reads at id 1 come back clean, which only happens with one servo on the line. Power the servos off while moving cables. If it stops partway, run it again and it carries on. It only writes ids; torque stays off.

## First powered moves

After `lerobot-calibrate`, `scripts/first_moves.py` turns the motors on and moves each joint a little, one at a time, so you see every joint drive and say which way it turns. Install LeRobot first (`uv pip install -e '.[lerobot]'`).

```bash
python scripts/first_moves.py --port /dev/cu.usbmodemXXXX
```

- A servo drives to its goal register the moment torque comes on. After calibration and moving the arm by hand, that goal can be far from where the arm is, and LeRobot's `connect()` turns torque on without setting it. So the script runs LeRobot's `configure()` with its final torque-on held back, caps torque at 50% (`--torque`), sets each goal to the present position and reads it back. Then it turns the joints on one at a time from the base out, watching the arm for 0.3 s after each.
- If a joint moves more than 3° on its own as its torque comes on, or strays more than 5° from where it should be during a move, every motor is cut at once. The cut is one broadcast write that servos don't answer, so a servo flagging overload can't refuse it, the way it makes LeRobot's own torque-off fail.
- Every joint that gets torque has to start inside its calibrated range, and the script asks you to move in any that aren't. Only the joints in `--joints` get torque, so hold the rest of the arm when you test a few.
- `configure()` sets every joint's P gain to 16, a value chosen for the follower's 1:345 servos. `--gain elbow_flex=32` raises one joint's gain after that, reads it back before any torque comes on, and leaves it on the servo until LeRobot's setup runs again. This arm's elbow is a leader-type 1:191 servo that couldn't lift its folded forearm at 16.
- Each joint moves 10° toward the middle of its range over 2 s (`--degrees`, `--seconds`), holds while you say which way it went, then comes back. LeRobot's `max_relative_target` keeps every goal within 5° of where the joint is.
- Every step waits for Enter. Ctrl-C stops the moves and the arm holds where it is. After that, or a normal finish, the motors go off only once you say you're holding the arm, because it drops when they do.
- Each run writes `.wbc/logs/first-moves-<time>.json`: every reading while each joint came on and moved, what each joint was asked and did, and which way plus turns it.

## Power cut-off test

The servos' power supply runs through a switched power strip, which is the arm's emergency stop. `scripts/power_cut.py` checks that switching it off makes the arm let go, and that the arm stays limp when the power comes back. Only the servo supply should be plugged into that strip.

```bash
python scripts/power_cut.py --port /dev/cu.usbmodemXXXX idle
python scripts/power_cut.py --port /dev/cu.usbmodemXXXX hold
```

- `idle` comes first. The motors stay off and nothing is written to the servos. You switch the strip off, and the script records how the supply voltage falls until the servos stop answering. You switch it back on, and it checks what they came back with: torque off on all six, and which settings the cut reset.
- `hold` turns the motors on at 50% torque the way `first_moves.py` does, with the elbow at P = 32, and holds a pose you set once you've let go and it has settled. With a hand under the arm, you switch the strip off, and it records every joint and the supply voltage until the servos go silent. If the arm doesn't sink as you let go, its motors carry no weight and there's nothing to drop, so the script says so before the cut and that run doesn't count. If a joint moves more than 5° while the voltage is still normal, every motor is cut at once. If the servos still answer 2 s after the voltage starts to fall, the script cuts them itself.
- Both wait as long as you take, so you can step away. Ctrl-C stops either one, and in `hold` the motors go off only once you say you're holding the arm.
- The supply keeps its output up for a moment after the switch, which the servos can't report, so `hold` asks whether the arm let go at once.
- If the USB adapter disappears along with the servos, the strip powers the adapter or its hub too, and the script says so.
- Each run writes `.wbc/logs/power-cut-<step>-<time>.json`: the servo registers before and after, and every reading around the cut.

What it found on this arm (Oct 27–28): the strip cuts only the servo supply. Once the supply voltage starts to fall, the servos stop answering within about 10 ms, with or without torque on, before a loaded joint has moved even a degree. A loaded elbow let go at the click and fell onto its stop. The servos come back limp, with every goal at 0 and the torque cap back at each servo's stored maximum, so code that turns torque on after a cut has to set its cap again. Folded poses near upright hold themselves up by gear friction, so only a joint carrying weight drops.

## Joint map and arm driver

`wbc/motion/joints.py` turns each servo's raw position into the URDF's joint angle: q = sign · (raw − 2047) · 2π / 4096. lerobot-calibrate's Part 1 makes every joint read 2047 in the L pose (base facing forward, upper arm straight up, forearm and gripper level and pointing forward), which is the URDF's zero. Every sign is +1. That was checked by hand on Oct 28 with the motors off, each joint turned in a pose where the move reads clearly: raw rises as the base turns clockwise seen from above, the upper arm tips forward, the elbow folds, the gripper tips down, and the wrist rolls counterclockwise looking from the wrist toward the jaw tips. The answers `first_moves.py` recorded on Oct 27 had three of those backwards, because they were read from the folded rest pose. `scripts/check_joint_map.py` repeats the hand test, so run it again after a recalibration or a servo swap:

```bash
python scripts/check_joint_map.py --port /dev/cu.usbmodemXXXX once    # each joint raw, in LeRobot's degrees and the URDF's, and the tool point
python scripts/check_joint_map.py --port /dev/cu.usbmodemXXXX watch   # the URDF angles and the gripper's opening, live
python scripts/check_joint_map.py --port /dev/cu.usbmodemXXXX check   # the guided test: five moves and the L pose
```

- It only reads, through `wbc/motion/sts.py` and never LeRobot, and it won't run with a motor on. The calibrated ranges come from the limits lerobot-calibrate wrote into the servos, so it needs no calibration file; `--calibration PATH` compares one with them.
- `check` waits for Enter at each step and passes a move when it turned its joint's URDF angle the way the URDF says that move goes. A move under 5° is asked for again. The L pose passes when the shoulder lift, elbow and wrist flex read within 8° of zero (they read within 2° on Oct 28). The check passes only when every move and the L pose were checked and right; q, Ctrl-C or a servo going silent still prints the summary. It writes `.wbc/logs/joint-map-check-<time>.json`.
- It refuses a servo other than the wrist roll whose limits are the whole turn, 0–4095, as a new servo comes from the factory: run lerobot-calibrate first.

`wbc/motion/arm.py` is the driver, and the only code that should talk to the servo bus while the agent runs. `Arm.open()` connects LeRobot's bus and nothing more: never `robot.connect()`, which can recalibrate the arm and always ends by turning torque on. `power_on()` brings the joints up the way `first_moves.py` does, and `move_to(q)` moves all five together to URDF angles q. Writing a joint's goal turns its torque on, so each goal goes in only as that joint's turn comes, and a write whose reply flags the input voltage, a reading glitch on these servos, goes again, up to 3 times.

- Refused before anything is written. `power_on()` refuses a joint past an end of its calibrated range, or the wrist roll past the URDF's limits, which protect its cable (it says which way to turn the gripper back), and a servo at 65 °C (a warning from 55 °C); heat is checked only there. `move_to()` refuses a start past those ends, a target within 2° of the end of a joint's range, and a path that takes the tool point under 10 mm, or the elbow and wrist axes under 30 mm, above the table. A move that starts under a floor has to climb back over it without going lower on the way, or only turn the base, so small moves can't walk the arm down.
- Moves are quintic and peak at 30°/s on any joint and 80 mm/s at the tool point, and no goal goes more than 5° from where its joint is. Torque is capped at 50% on the arm joints and 24% on the gripper, under its 25% overload start, and the elbow's P of 32 goes back on after LeRobot's setup at every power-on.
- Every motor is cut at once if a joint jumps 3° as its torque comes on, strays 5° from its goal during a move, goes 5 mm under a floor, or the bus fails. The floors hold for the plan, not the arm: joints within the 5° stray can put the tool point 17–41 mm under the plan, so they aren't a clearance guarantee.
- `cut()` works from any thread or a signal handler, and a move or power-on running elsewhere stops at its next tick. Every cut is read back, servo by servo, and when it can't be confirmed the message says to switch the strip off. A cut latches, whether from `cut()`, `close()` or a trip: `power_on()` refuses until `clear_stop()`, so a power-on queued behind a stop can't turn the motors back on. Call it once someone has checked the arm.
- Ctrl-C reaches only Python's main thread: a move there holds the arm where it is, and one on another thread stops with `cut()`. Ctrl-C while LeRobot awaits a reply leaves scservo_sdk's port marked busy, which would fail every later request, the cut included, so the driver clears it, and drops the late reply, before it talks to the bus again.
- Each move first checks that every powered joint still has its torque on at its cap. After a power cut the servos come back limp and uncapped, so the move is refused until `power_on()` runs again. If one servo resets on its own, the move is refused and the rest keep holding: hold the arm, `cut()` and `clear_stop()`, then power on again.
- After a move's path, `move_to()` re-aims. Each joint stops short of its goal by its load over its P gain (gravity and gearbox friction) and stays there, so it moves each goal past its target by the shortfall, ramped under the same speed caps with every check a move has, until every joint is within 0.2°, for up to 3 rounds. A goal never goes more than 4° past its target or past its joint's range, and a round that would aim the arm under a floor isn't run. `move_to(q, aim=False)` skips it.
- `grip()` closes the gripper on what's between its jaws and keeps squeezing, under its 24% cap, and says whether they stopped on something. `release(opening)` opens them to a share of their range, at up to 60°/s, which the capped jaw can follow. If something's in the way it stops them where they are, says so, and returns once they've come to rest. `jaw()` reads them. While it grips, jaws closing further because what they held slipped out don't trip a move, and `hold()` keeps the squeeze. Ctrl-C during a grip stops the jaws without squeezing, in case it's a finger.
- `close()` cuts, then, once the cut reads back, puts each servo's torque cap back and locks every servo's EEPROM again (LeRobot's setup leaves it unlocked), and disconnects without LeRobot's torque-off. Keep to one `Arm` per port, and close it before another program uses the port (`check_joint_map.py`, `first_moves.py`): LeRobot opens the port without locking it, so their packets would interleave.
- LeRobot keeps its calibration file per macOS account. From another account, pass the calibrating account's folder: `Arm.open(port, load_yaml("robot.yaml"), ROOT, calibration_dir="/Users/<account>/.cache/huggingface/lerobot/calibration/robots/so_follower")`.

Its first powered run (Oct 29, small moves from the rest pose) was clean, and showed every joint holding 1–4° short of its goal, which the re-aim makes up. On Oct 27 the elbow lagged its goal by up to 3° at about 8°/s, so a fast elbow move under load may trip the 5° stray cut. If it does, lower `MAX_JOINT_SPEED` or `MAX_TIP_SPEED` before raising `MAX_STRAY`.

### Pointing at a hole

To point at a hole, the arm has to know where the carrier lies. `scripts/register_arm.py touch`, with the motors off, asks you to put the tool point (the closed jaws' tips) on the sheet's four tag centers, or any tags and holes you name, and fits the turn about the vertical and the shift from carrier mm to the arm's frame (`wbc/motion/carrier.py`). The table is level, so four numbers are enough, and the fit measures the base's heading rather than trusting "facing forward" by eye. It saves `.wbc/registration.json` when the points sit within 2 mm rms and 3 mm worst. `verify` is the touch test without power: touch holes, and it says how far the tool point is from each across the sheet and in height, passing at a mean of 3 mm or less.

```bash
python scripts/register_arm.py --port /dev/cu.usbmodemXXXX touch               # tag10 tag11 tag12 tag13
python scripts/register_arm.py --port /dev/cu.usbmodemXXXX verify a1 j63 a5    # the touch test, motors off
python scripts/register_arm.py --port /dev/cu.usbmodemXXXX watch               # the hole under the tool point, live
python scripts/register_arm.py --port /dev/cu.usbmodemXXXX plan a5             # what point_at would do, moving nothing
```

`carrier.point_at(arm, registration, point_mm(bench, parts, "a5"))` points the gripper at a hole from 30 mm back along its own line: as steep as the arm can, straight down where it can reach that way, and up to 50° from it where it can't. It moves to 60 mm further back on that line first, then in. It only calls `move_to()`, so every refusal and cut stays in force, and it never powers the arm on or clears a stop. Like the driver, it's tested offline only.

## Gripper probe: what can servo feedback feel?

`scripts/gripper_probe.py` reads the gripper servo's position, load and current about 200 times a second, so you can measure whether servo feedback alone can see a grip, a pull or a slip. Install the arm extra first (`uv pip install -e '.[arm]'`), close LeRobot, and turn the servo power on.

```bash
python scripts/gripper_probe.py --list
python scripts/gripper_probe.py --port /dev/tty.usbmodemXXXX watch
python scripts/gripper_probe.py --port /dev/tty.usbmodemXXXX squeeze --closed 2150
```

- `watch` only reads. With torque off, move the jaws by hand and note the open and closed positions.
- `squeeze` closes slowly toward `--closed` with the torque capped (300‰ by default, 600‰ at most), then holds. Hold still for 2 s, then type a label such as `pull` and press Enter as you start each action. Ctrl-C ends the run and releases the gripper.
- Each run writes `.wbc/logs/gripper-<time>.csv` and prints, for every label, how far load, current and position moved and how that compares with the noise while holding still.

The current register's units aren't confirmed for your servos, so compare the raw numbers with each other rather than converting them to mA.

## Force sensors: calibration and creep

Each jaw gets two thin-film force sensors, read by an ESP32 running `firmware/fsr_sampler` at 500 Hz. Each sensor is half of a voltage divider: one leg to 3V3, the other to its pin, and a 10 kΩ resistor from that pin to GND.

| Sensor | Pin |
|---|---|
| SD, static jaw, distal (tip) | GPIO 32 |
| SP, static jaw, proximal | GPIO 33 |
| MD, moving jaw, distal | GPIO 34 |
| MP, moving jaw, proximal | GPIO 35 |

```bash
uv pip install -e '.[arm,firmware]'
pio run -d firmware/fsr_sampler -t upload --upload-port /dev/cu.usbserial-XXXX
python scripts/calibrate_fsr.py --list          # the ESP32 is the CP210x or CH340 line, not the arm's adapter
python scripts/calibrate_fsr.py --port /dev/cu.usbserial-XXXX watch
python scripts/calibrate_fsr.py --port /dev/cu.usbserial-XXXX calibrate --channel SD
python scripts/calibrate_fsr.py --port /dev/cu.usbserial-XXXX creep --channel SD --grams 100
```

- `watch` shows all four channels live, with the frame rate and any dropped or damaged frames. An unloaded sensor reads about 140 mV, which is the lowest voltage the ESP32's ADC reports.
- `calibrate` zeroes one sensor, then takes a reading for each known weight you put on it, and fits `F = k · G^b`, where G is the sensor's conductance above unloaded. The fit for each channel goes to `config/fsr_cal.yaml`. Press through a small puck about the size of the 8 mm sensing area, because load that spreads past it reads low.
- `creep` records 3 s unloaded, then the weight for 120 s, then 30 s unloaded again. It reports how far the reading drifts after 3, 10, 30, 60 and 120 s, the drift rate compared with the slip detector's 3 N/s cue, and what's left after you lift the weight off.
- Both write `.wbc/logs/fsr-<mode>-<time>.csv`.

## Seeing the bench with SAM 3.1

SAM 3.1 on the Meta Model API takes one short noun phrase and one image per request, and returns a box and a mask for every match. `wbc/perception/sam_client.py` sends one request per phrase, in parallel, and decodes the masks with Meta's `meta-sam-parser`. Each request is billed as one image, $2.50 per 1,000. `scripts/sam_probe.py` shows what SAM finds on your bench. Run it from Terminal.app, because macOS only lets a terminal it has asked use the camera:

```bash
uv pip install -e '.[eyes]'
python scripts/sam_probe.py cameras                  # free: which cameras macOS and OpenCV see
python scripts/sam_probe.py capture --camera 0       # free: a 1920x1080 still in .wbc/vision/
python scripts/sam_probe.py segment .wbc/vision/photo-<time>.png --phrase "wire"   # $0.0025 per phrase
python scripts/sam_probe.py replay .wbc/vision/sam-<time>.json   # free: parse and draw a saved reply again
```

- Name the thing, not an instruction: `"wire"`, not `"find the wires"`, and use the plainest word. On the bench, `"jumper wire"` matched nothing at all, where `"wire"` found every wire. A list or an "and" doesn't segment two things, so pass `--phrase` once per thing.
- No match is a normal, empty result, not an error.
- `segment` draws every mask over the photo in `.wbc/vision/sam-<time>.png`, lists each match's box, area and confidence, and keeps SAM's whole reply in the run's `.json`, so `replay` can parse it again without paying.
- `--crop X,Y,W,H` sends only part of the photo. The open SAM 3 model works on 1008 × 1008 pixel images, so a crop of just the breadboard may show thin wires at more detail than the whole 1920 × 1080 frame.

## Reading the breadboard

`read_breadboard` turns one photo into each jumper's two holes, the netlist, and every place it disagrees with `pins.h`:

1. **Tags.** The breadboard lies on a printed carrier sheet with four AprilTags (tag36h11, ids 10–13, 20 mm). Their corners map the photo onto the sheet in millimeters, so a bumped camera or a refocus doesn't move the hole grid, and there's no camera calibration to keep.
2. **Top-down view.** Code cuts out the board area at 5.5 px/mm, about 970 × 360 px. That's under the 1008 px SAM works at, so thin wires keep their detail, and SAM's masks come back in board millimeters.
3. **SAM 3.1 and the hole grid.** SAM segments `wire` in the view. Meanwhile code finds where the board really lies: its holes are small dark squares, and it shifts and turns the grid, by up to 1.2 mm and 1°, until the predicted holes sit on the darkest spots. That's under half a pitch, because a grid a whole pitch off fits just as well. It scales the grid too: the board's top stands about 10 mm above the sheet the tags are printed on, so from 30 cm the camera sees it about 2% larger than life, enough to put the end rows a millimeter off. Only the right scale lines up all 63 rows.
4. **Wires.** Each mask is thinned to its center line and followed out to both tips. A flat jumper's legs go into its holes at its tips, so each tip snaps to the nearest hole within 1.2 mm, and a tip farther from every hole is flagged. A mask that branches is flagged as two wires that may touch.
5. **Netlist and `pins.h`.** Each row's a–e and f–j strips and each rail is a node, wires join nodes, and the ESP32's and the sensor's pins sit in holes given in `config/bench.yaml`. Plain code then follows each `pins.h` signal from its part's pin to the GPIO its net reaches, and reports any mismatch, its wire and hole, and the line to fix.

**The Act 1 board** is an ESP32 DevKit reading one force sensor. The DevKit straddles the center channel, which leaves only column a free beside its pins, and two flat wires heading the same way along one column lie on top of each other, where the camera can't see the lower one. So the board carries one signal wire: a jumper from a5 (GPIO 4) to a26, where the sensor's pin and a 10 kΩ resistor to ground meet, while `pins.h` reads GPIO 15. Short jumpers take the sensor's other pin to the 3.3 V rail and the resistor to the ground rail. Press the sensor and the readings don't move; after the fix they rise to about 1.9 V. Watch them yourself with `.venv/bin/pio device monitor -b 115200`. The 7-wire SPI display planned first is shelved.

It reads flat pre-cut jumpers. Loose jumper leads stand about 15 mm off the board, and from a camera 30 cm up their tips look up to 4 mm away from where they go in, more than a hole.

```bash
python scripts/make_carrier.py          # .wbc/vision/carrier.pdf: print at 100%, check the 100 mm bar
python scripts/read_board.py check      # free: the tags, and the fitted hole grid drawn over the board
python scripts/read_board.py read       # $0.0025: the whole read, drawn in .wbc/vision/board-<time>.png
```

- Both take a still from the camera in `bench.yaml` and keep it as `.wbc/vision/photo-<time>.png`, or read one with `--photo`. Run them from Terminal.app. The C920 refocuses every time it's opened, so the capture waits 1.5 s for exposure, then keeps the sharpest of the next 1.5 s of frames (the variance of the Laplacian).
- `config/bench.yaml` holds the camera, the carrier's tags, each board's place on the sheet, where each module's headers start, and which part pin each `pins.h` signal drives. `config/parts.yaml` holds the breadboard's geometry and the module pinouts. Change those and print the carrier again; never edit the sheet.
- `reader.tip_inset_mm` moves each wire end back from the tip of its mask, for jumpers whose masks stop short of their holes or run past them. Set it from the golden photos' residuals.
- Each read saves the drawing, the top-down view and the result, with SAM's whole reply, as `.wbc/vision/board-<time>.*`.

## Finding things on the desk

A second C920 looks straight down at the desk from about 71 cm, centered about 20 cm in front of the arm's base, and sees the arm's whole reach: about 97 × 55 cm at 0.5 mm per pixel. The board camera keeps its own job over the breadboard. `wbc/perception/desk.py` turns what the desk camera sees into the arm's frame, so the co-pilot can find "the cup" or "the screwdriver" and know where it is.

1. **Tags, touched once.** Six AprilTags (tag36h11, ids 20–25, 50 mm) lie flat around the arm's reach. With the motors off, you put the gripper's tips on each tag's center, and forward kinematics records where it is in the arm's frame. Every photo is then mapped through the tags it shows, so a bumped camera changes nothing. A moved tag needs touching again.
2. **Top-down view and SAM 3.1.** Code cuts out the table around the arm at 1 px/mm and asks SAM for the thing by name, one request per phrase.
3. **Height.** One camera can't see height, and something standing up looks further from the point under the camera than it is. The tags also place the camera itself, so given a thing's height, its outline's near edge (its base, on the table) and far edge (its top) put its middle back where it is. A cup's middle lands within 3 mm with its height given. Without it, the cup reads 15 mm off from straight above.

```bash
python scripts/desk_camera.py tags                                    # .wbc/vision/desk-tags.pdf: print at 100%, check the 100 mm bar
python scripts/desk_camera.py --port /dev/cu.usbmodemXXXX touch       # motors off: touch each tag's center with the gripper's tips
python scripts/desk_camera.py check                                   # free: how well the tags agree, and where the camera is
python scripts/desk_camera.py locate cup --height 95                  # $0.0025: where the cup is in the arm's frame
```

- Run `check` and `locate` from Terminal.app, for the camera. They find the desk camera by its tags, so macOS renumbering cameras doesn't matter, and keep each photo in `.wbc/vision/`. `locate --replay PATH` reads a saved reply again for free.
- `config/desk.yaml` holds the camera, the tags, the area SAM sees and the focal length used to place the camera.

## How the router works

Muse Spark is the only model that talks to you and the only one that decides. Every other model sits behind a tool:

- `ask_coder` sends a task, the bench facts and up to 4 files to Muse Wonder. The diff it returns must pass `git apply --check`, and, if `coder.compile_cmd` is set, a PlatformIO build in a sandbox copy. Only then does the router see it, as a `patch_id`.
- `propose_patch(patch_id)` applies the diff to the working tree and opens VS Code at the first changed line. It never commits or flashes.
- Tool calls in one round run in parallel, so the arm can point while Wonder writes.
- Every model id and price lives in `config/models.yaml`.
- **Budget stop:** every paid call checks tracked spend first. Nothing more goes out once it reaches `budget.limit_usd` ($140, just under the $150 credit), and the HUD warns once when it passes `budget.warn_usd` ($100). Tracked spend lives in `.wbc/spend.json` and uses the estimated prices in `models.yaml`. Calls made elsewhere, such as the playground or another machine, aren't counted, so keep the spend alert on the Meta Billing page as the backstop.

## Kinematics notes

- In the URDF the gripper approaches along its z-axis. `config/robot.yaml` rotates the tool frame so the approach is x.
- The point between the jaws moves ±8 mm sideways as the wrist rolls, so a 5-joint arm can't always face exactly along the target's bearing. `solve_task` lets yaw move up to 20° and reports the result in `IKResult.yaw`. Near a top-down grasp, a yaw change also turns the jaws, so the grasp skill should correct roll by it when jaw angle matters.
- Poses behind the base need the arm folded back over itself, and random starts rarely find that solution. Pass `q0`, the current joint angles, when you need continuity.
- LeRobot reports joints in degrees, with 0° at the middle of each joint's calibrated range. That isn't the calibration pose when a joint's range is lopsided: on Oct 27 the base's 0° was about 12° off forward. `scripts/first_moves.py` records which way plus turns each joint. Check both against a ruler before trusting IK on the real arm.

## Check on Day 1

- Model ids and prices: run `scripts/smoke_test.py` and compare with Meta's pricing page. The Spark 1.3, Wonder and Voice Transcribe prices in `models.yaml` come from third-party write-ups.
- The SAM 3.1 request format: found in Meta's docs and built on Oct 28 (see "Seeing the bench with SAM 3.1"). Access to the gated `facebook/sam3.1` weights is requested, for an offline backup.
