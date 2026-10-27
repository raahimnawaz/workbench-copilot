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
.venv/bin/python -m pytest -q                                   # 84 offline tests; CI runs them on every push
.venv/bin/python scripts/smoke_test.py                          # PAID, about $0.01: one call to each hosted model
echo "Zero response on SPI. Check my wiring against the firmware." | .venv/bin/python -m wbc.agent --sim   # PAID, about $0.04
.venv/bin/python scripts/gripper_probe.py --list                # free: find the servo bus port
.venv/bin/pio run -d firmware/fsr_sampler -t upload --upload-port /dev/cu.usbserial-0001   # flash the force sensor sampler
.venv/bin/python scripts/calibrate_fsr.py --port /dev/cu.usbserial-0001 watch   # free: also calibrate / creep --channel SD
```
- A `--sim` run applies Wonder's real patch to `firmware/dut_demo/include/pins.h` (`OLED_CS` 5 → 13). **Restore it afterwards** with `git checkout -- firmware/dut_demo/include/pins.h`. That bug is the Act 1 demo: the CS wire goes to GPIO 13 while `pins.h` says 5.
- Tracked spend lives in `.wbc/spend.json` ($0.049 on Oct 27). The hard stop is $140 and the warning $100 (`budget` in `config/models.yaml`).
- PlatformIO is installed in the venv (the `firmware` extra), and both firmware projects compile. `ask_coder` runs its compile check only when `pio` is on PATH, so activate the venv first.

## State on Oct 27 (`main` = 9741561, PRs #1–#3 merged)
- Router (`wbc/agent/loop.py`): Muse Spark 1.3 on the Responses API with parallel tool calls. `ask_coder` sends code changes to Muse Wonder and returns only diffs that apply (and compile, when `pio` is present). A Chat Completions router for Muse Glimmer on this Mac is coded (`--backup`) but has never run.
- Verified live on Oct 26: every model id is in the account's model list; Spark tool calls, Wonder diffs and Voice Transcribe all work; a `--sim` Act 1 run worked end to end.
- **Open problem:** that Act 1 turn took 81 s, 75 of them in three Spark calls, against a 55 s slot. The router now asks for `effort: low` and logs each call's time. It hasn't been re-timed.
- FK/IK (`wbc/motion/kinematics.py`) is tested on the real URDF: 200/200 bench poses within 0.2 mm and 0.5°.
- `scripts/gripper_probe.py` (with `wbc/motion/sts.py`) has never run on the arm.
- Force sensors (Oct 27): `firmware/fsr_sampler` is flashed on the ESP32 at `/dev/cu.usbserial-0001` and streams 500.0 Hz with no dropped or bad frames. `scripts/calibrate_fsr.py` (`watch`, `calibrate`, `creep`, with `wbc/reflex/fsr_link.py` and `fsr_cal.py`) is tested against a fake sensor but hasn't run with a real one.
- Not started: SAM 3.1 client and breadboard netlist reader, web HUD, slip detector, motion skills, compliant hold.

## Next, in order
0. Now: wire the sensors, calibrate each one, and run the creep test (100 g for 120 s). The result decides whether the compliant hold regulates force or locks the jaw position.
1. Re-time Act 1 with `effort: low` (paid, ask first). If it's still slow, cut router rounds and compare `muse-spark-1.1`.
2. Arm bring-up: LeRobot calibration, scripted moves, test the power cut-off (a switched power strip). Then the gripper probe: `watch` to find the closed position, then `squeeze` with labeled pulls.
3. Voice with the Razer mic (`--voice`). There's no foot pedal, so set `ears.pedal_key` to a key on the keyboard (for example `alt_r`).
4. Oct 28: mount the C920 about 30 cm over the breadboard, print the AprilTags, wire the demo board (pins below), flash it, and build the SAM 3.1 reader to ≥ 98% of wire endpoints on 20 photos. Touch test ≤ 3 mm.
5. Oct 29: web HUD; force-sensor pads and the ESP32 sampler; slip detector on recorded traces.
6. Oct 30: Act 1 end to end, 10 runs under 60 s, then the backup video.
7. Oct 31 – Nov 2: grip PI loop, slip catch and regrasp, compliant hold. ROS 2 nodes come only after Act 1 works (see `../ros2-gate`).
8. Nov 3 – 4: rehearse, record, submit by 9:00 PM ET, post the video on X.

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
