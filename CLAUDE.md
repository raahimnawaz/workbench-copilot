# workbench-copilot: notes for Claude

Entry for the Meta Global AI Developer Hackathon (Indie track, Push the Frontier). The build runs Oct 26 – Nov 4, 2026, and **submissions close Nov 4 at 9:00 PM ET** with a public repo and a 3-minute video. The plan of record is the blueprint artifact at https://claude.ai/artifact/1f5AMxFTVycA2C25DkqWwm. Read it with the Artifact tool, starting with the Status sheet, before changing it.

## Working rules
- Every change goes on its own branch with a PR into `main`. Push, open the PR and merge without asking. Ask before destructive git operations.
- **Paid API calls:** say what will run and roughly what it costs, then wait for a yes. Only run paid calls when they're needed. Offline tests are free.
- Never read, print or commit `.env`, which holds `META_API_KEY`.
- The demo may only show work built during the event. Keep the README's built-vs-reused table accurate, and add any reused code with its license.

## Commands (Python 3.12 via uv, which lives in `~/.local/bin`)
```bash
uv venv --clear --python 3.12 && uv pip install -e '.[voice,dev,arm,firmware]'   # also after moving the folder: .venv stores absolute paths
.venv/bin/python -m pytest -q                                   # 112 offline tests; CI runs them on every push
.venv/bin/python scripts/smoke_test.py                          # PAID, about $0.01: one call to each hosted model
echo "Zero response on SPI. Check my wiring against the firmware." | .venv/bin/python -m wbc.agent --sim   # PAID, about $0.04
.venv/bin/python scripts/gripper_probe.py --list                # free: find the servo bus port
.venv/bin/python scripts/first_moves.py --port /dev/cu.usbmodem5B910297421   # the arm's first powered moves (needs the lerobot extra)
.venv/bin/pio run -d firmware/fsr_sampler -t upload --upload-port /dev/cu.usbserial-0001   # flash the force sensor sampler
.venv/bin/python scripts/calibrate_fsr.py --port /dev/cu.usbserial-0001 watch   # free: also calibrate / creep --channel SD
```
- A `--sim` run applies Wonder's real patch to `firmware/dut_demo/include/pins.h` (`OLED_CS` 5 → 13). **Restore it afterwards** with `git checkout -- firmware/dut_demo/include/pins.h`. That bug is the Act 1 demo: the CS wire goes to GPIO 13 while `pins.h` says 5.
- Tracked spend lives in `.wbc/spend.json` ($0.049 on Oct 27). The hard stop is $140 and the warning $100 (`budget` in `config/models.yaml`).
- PlatformIO is installed in the venv (the `firmware` extra), and both firmware projects compile. `ask_coder` runs its compile check only when `pio` is on PATH, so activate the venv first.

## State on Oct 27
- Router (`wbc/agent/loop.py`): Muse Spark 1.3 on the Responses API with parallel tool calls. `ask_coder` sends code changes to Muse Wonder and returns only diffs that apply (and compile, when `pio` is present). A Chat Completions router for Muse Glimmer on this Mac is coded (`--backup`) but has never run.
- Verified live on Oct 26: every model id is in the account's model list; Spark tool calls, Wonder diffs and Voice Transcribe all work; a `--sim` Act 1 run worked end to end.
- **Open problem:** that Act 1 turn took 81 s, 75 of them in three Spark calls, against a 55 s slot. The router now asks for `effort: low` and logs each call's time. It hasn't been re-timed.
- FK/IK (`wbc/motion/kinematics.py`) is tested on the real URDF: 200/200 bench poses within 0.2 mm and 0.5°.
- `scripts/gripper_probe.py` hasn't run on the arm yet. `wbc/motion/sts.py` has (Oct 27): `scripts/set_servo_ids.py` gave the freshly built arm's servos ids 1–6, all on 5.3 V with no error flags. On the real servos, the first ping after an id write goes unanswered while the servo saves to EEPROM, and servos sharing an id still pass pings about a third of the time, so `set_id` retries and the script checks with full reads.
- LeRobot 0.6.1 is installed (the `lerobot` extra, kept out of CI). The arm's port is `/dev/cu.usbmodem5B910297421`. It's calibrated (Oct 27, `--robot.id=wbc_follower`; read back from the servos, it matches the file). `lerobot-calibrate` writes each recorded range into the servos as position limits, so a joint left unswept can't move under power. LeRobot's 0° is the middle of each joint's recorded range, not the calibration pose: about 12° off forward for the base, under 5° for the others. `scripts/first_moves.py` turns joints on one at a time at 50% torque, cuts every motor at once if one moves on its own, then moves each joint 10° and back, recording which way plus turns it.
- **Open problem (Oct 27), the elbow flip: the elbow servo (id 3) is faulty.** Whenever its torque comes on, it drives itself toward opening, its effort climbing from zero while it still sits at its goal, until it hits the far stop and trips overload. That happened in every pose tried (folded, mid-range, the calibration pose), with its homing offset at 0 as well as at −1831, and in the calibration pose against gravity. Every byte of its settings matches the base, wrist and roll servos, which hold. `first_moves.py --joints elbow_flex` now catches it: torque cut 126 ms in, at −3.2°. A direction test settled it. Told to open 9° it overshot, and told to fold 9° it opened anyway, at full capped effort both times. So it drives open whatever its goal is: it's broken, not wired backward. It moves freely by hand. Next: replace it with an STS3215 7.4 V 1:345 (the shoulder lift's type; the other five are leader-arm 1:191 and 1:147 servos), give it id 3 and recalibrate.
- The shoulder lift servo (id 2) has different factory settings from the other five: Max_Voltage_Limit 8.0 V vs 12.0 V, Max_Temperature_Limit 70 vs 80 °C, Maximum_Velocity_Limit 68 vs 250, Phase 12 vs 76. The arm may have mixed servo types, so read the labels before changing the supply.
- Force sensors (Oct 27): `firmware/fsr_sampler` is flashed on the ESP32 at `/dev/cu.usbserial-0001` and streams 500.0 Hz with no dropped or bad frames. `scripts/calibrate_fsr.py` (`watch`, `calibrate`, `creep`, with `wbc/reflex/fsr_link.py` and `fsr_cal.py`) is tested against a fake sensor and has run on real sensors.
- **Force sensor bench work is paused (Oct 27), with an open problem we may need to go back to:**
  - Wired: SD (GPIO 32) and SP (GPIO 33), each with 10 kΩ. MD and MP aren't wired.
  - Coins pressing on a bare sensor barely register; pressing through a small bolt as the puck works. The jaw pads need a puck.
  - Creep, measured on SD with 51 g: +6.8% over 3 s, +17% over 10 s, +28% over 60 s, then straight back to the floor on removal. Decision: the compliant hold locks the jaw position instead of regulating force. This result is solid.
  - Calibration is rough only. With 9 quarters (up to 51 g) the fits miss by up to 27% (SD) and 20% (SP), and SP doesn't register 11.3 g. Those fits are in the local `config/fsr_cal.yaml`, which isn't committed.
  - **Open problem:** on both sensors, 34 g and 51 g read about the same (about 1650 mV, sensor near 10 kΩ). Placement repeats within about 6% (51 g three times: 1634, 1727, 1658 mV), and a hard finger press reaches about 1 kΩ, so this flat spot is real. Either part of the load goes around the sensor as weight grows, or the sensor stops responding through a small puck above about 35 g. If it's the second, the gripper can't feel its 60–150 g grip, so **resolve this before building the jaw pads**. It needs heavier known weights: a cheap kitchen scale (press test at 50, 100, 200, 300 g while watching mV), or measured water.
  - Keep the 10 kΩ resistors. 47 kΩ or 100 kΩ would push grip-range readings to the top of the ADC. Revisit (possibly 4.7 kΩ) only after the heavier test.
- Not started: SAM 3.1 client and breadboard netlist reader, web HUD, slip detector, motion skills, compliant hold.

## Next, in order
0. Force sensors are paused (see above). Before the jaw pads (Oct 29): get a kitchen scale, run the heavier press test to settle the flat spot above about 35 g, wire MD and MP, then calibrate all four in the pads.
1. Re-time Act 1 with `effort: low` (paid, ask first). If it's still slow, cut router rounds and compare `muse-spark-1.1`.
2. Arm bring-up: calibrated Oct 27, but the elbow servo is faulty (see "the elbow flip" above): replace it, give it id 3 and recalibrate. Then `scripts/first_moves.py`, test the power cut-off (a switched power strip), then the gripper probe: `watch` to find the closed position, then `squeeze` with labeled pulls.
3. Gravity model of the arm, once the scripted moves have logged joint loads (about half a day, before Oct 31). It predicts each joint's load from the pose, using the link masses already in `hardware/so101_new_calib.urdf`, plus a small fitted correction for friction and the units of Present_Load. It's a fitted physics model, not a neural network: there's little data, and it has to be checkable. Three uses:
   - Weigh a gripped object: measured minus predicted load at the shoulder and elbow. The regrasp force estimate in step 8 needs this.
   - Collision stops with a tighter threshold, because the part of the load that comes from the pose is removed.
   - Check a pose before moving: can the arm hold it at the hold mode's 45% torque limit? On 5 V the servos make less torque than their 7.4 V rating.
4. Voice with the Razer mic (`--voice`). There's no foot pedal, so set `ears.pedal_key` to a key on the keyboard (for example `alt_r`).
5. Oct 28: mount the C920 about 30 cm over the breadboard, print the AprilTags, wire the demo board (pins below), flash it, and build the SAM 3.1 reader to ≥ 98% of wire endpoints on 20 photos. Touch test ≤ 3 mm.
6. Oct 29: web HUD; force-sensor pads and the ESP32 sampler; slip detector on recorded traces.
7. Oct 30: Act 1 end to end, 10 runs under 60 s, then the backup video.
8. Oct 31 – Nov 2: grip PI loop, slip catch and regrasp, compliant hold. ROS 2 nodes come only after Act 1 works (see `../ros2-gate`).
   - Regrasp force estimate, replacing the fixed 1.5 N once both gates below pass. A two-jaw grip needs a squeeze of at least W / (2μ) to hold weight W with friction μ, and at least 3·W·d / (4·μ·r) to stop an off-center grasp from pivoting (d is the distance from the grasp to the center of mass, r the pad's contact radius). At d = 20 mm and r = 4 mm the second is about 7 times the first, so moving the grasp to the center of mass beats squeezing harder, as Act 2 already plans.
   - Each input gets a first guess and a measurement. W: Spark's guess from the photo, then the gravity model after the lift. d: the middle of the SAM mask, then which way the board pivots in the slip (the twist cue, or SAM 3.1 video, which tracks objects across frames in one call at $0.20 per 1,000 frames; its request format isn't on Meta's model page yet). μ: Spark's guess from the materials, narrowed by whether the gentle grasp held.
   - Code computes the force and clamps it at 3 N, and force still rises only on a confirmed slip. Spark picks the grasp and says why, and the HUD shows predicted against felt weight.
   - Gates: the flat spot above about 35 g is settled (the grip range is 60–150 g), and the gripper probe shows the servos can weigh a 50 g board.
9. Nov 3 – 4: rehearse, record, submit by 9:00 PM ET, post the video on X.

## Facts
- **Hardware:**
  - SO-101: 5 arm joints plus the gripper, servo IDs 1–6, gripper = 6. STS3215 servos on a 1 Mbps bus.
  - Three classic ESP32 DevKits: the demo board, the FSR sampler (ADC1 pins, GPIO 32–39), and a spare. An Arduino Uno spare.
  - Logitech C920 (1080p), Razer Seiren Mini mic, UMLIFE SSD1306 0.96" 7-pin SPI OLEDs.
  - Usdrwam TFS-100 thin-film force sensors: 10 mm head, 8 mm sensing area, 20 g–2 kg, two pins at 2.54 mm.
- **Demo board pins** (`pins.h`): SCK 18, MOSI 23, CS 5 (wired to 13 on purpose), DC 16, RES 17.
- **Meta Model API** (`https://api.meta.ai/v1`, OpenAI SDK):
  - Router `muse-spark-1.3` and coder `muse-wonder-1.0` (accepts `reasoning={"effort": "low"}`).
  - Voice: POST `/v1/asr/transcribe` as multipart, with a JSON `request` part (`model`, `audioEncoding: "WAV"`, `keywords`) and an `audio` part. It returns `{"transcript": ...}`. Keyword hints work best as acronyms only.
  - SAM 3.1 (`sam-3.1`), documented at https://dev.meta.ai/docs/sam/segmenting but not in the code yet: POST `/v1/responses` with one `input_text` noun phrase and one `input_image` (data URL). One concept per request. It returns boxes and masks as a token string; read it with `meta-sam-parser` 0.0.6+. The `facebook/sam3.1` weights are gated, and access has been requested.
- **Force sensor wiring:** each sensor runs from 3V3 to its pin, with 10 kΩ from the pin to GND. SD, SP, MD, MP = GPIO 32, 33, 34, 35. The ESP32 reads nothing below about 142 mV, so an unloaded sensor reads that floor. The sampler is a CP2102 board (`10C4:EA60`). The arm's servo adapter is a CH343 (`1A86:55D3`) at `/dev/cu.usbmodem5B910297421`.
- **IK:** the tool frame is turned −90° about y so the approach axis is x. Yaw may move up to ±20° from the target's bearing (`IKResult.yaw`). Reach is about 460 mm at table height. Poses behind the base need `q0`.
- **Servos:** LeRobot's bus API moved (`FeetechMotorsBus` is deprecated in favor of `SerialMotorsBus`), so `wbc/motion/sts.py` talks to one servo directly. Registers: Torque_Enable 40, Goal_Position 42, Torque_Limit 48, feedback block 56–70. The units of Present_Current (69) are unconfirmed.
