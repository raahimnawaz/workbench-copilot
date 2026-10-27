# Workbench Co-Pilot

[![tests](https://github.com/raahimnawaz/workbench-copilot/actions/workflows/tests.yml/badge.svg)](https://github.com/raahimnawaz/workbench-copilot/actions/workflows/tests.yml)

An AI lab partner that lives in your code editor and on your workbench. You talk to it through a foot pedal. **Muse Spark** routes each request: **Muse Wonder** writes code, **SAM 3.1** reads the breadboard, **Muse Voice Transcribe** hears you, and an **SO-101** arm points, grasps and holds boards while you probe.

Built for the Meta Global AI Developer Hackathon (Indie track, Push the Frontier), Oct 26 – Nov 4, 2026.

## Built during the hackathon, and what it builds on

Everything in this repository was written during the hackathon, starting Oct 26, 2026, and the commit history dates every change. The one exception is the vendored URDF listed below.

| Built here | What it does |
|---|---|
| `wbc/agent/` | Router loop, tool registry, `ask_coder`, voice front end, budget guard, console HUD |
| `wbc/motion/kinematics.py` | URDF chain loader, forward kinematics, damped least-squares IK, the (x, y, z, pitch, roll) grasp frame |
| `wbc/motion/sts.py`, `wbc/motion/probe.py` | A small STS3215 servo client and the gripper feedback probe |
| `wbc/reflex/fsr_link.py`, `wbc/reflex/fsr_cal.py` | The force sensor link (500 Hz frames with a CRC), per-sensor calibration fit and creep analysis |
| `config/`, `fixtures/`, `scripts/`, `tests/` | Model routing config, the recorded bench for `--sim`, the smoke test, gripper probe and force sensor tools, 84 tests |
| `firmware/dut_demo/` | The Act 1 demo board firmware, with the wiring bug planted on purpose |
| `firmware/fsr_sampler/` | ESP32 firmware that samples the four jaw sensors at 500 Hz and streams them over USB |

| Reused | License | Used for |
|---|---|---|
| `hardware/so101_new_calib.urdf`, from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) | Apache-2.0 | The arm's kinematic chain (unchanged copy) |
| [LeRobot](https://github.com/huggingface/lerobot) | Apache-2.0 | Servo calibration and the Feetech bus (from Oct 27) |
| Muse Spark, Muse Wonder, Muse Voice Transcribe, SAM 3.1 on the Meta Model API | Meta API terms | Routing, coding, speech, segmentation |
| Muse Glimmer 30B weights | Apache-2.0 | The local backup router |
| openai, numpy, PyYAML, httpx, sounddevice, pynput, pyserial, pytest | Apache-2.0, BSD-3, MIT, BSD-3, MIT, LGPL-3.0, BSD-3, MIT | Python dependencies |
| Adafruit SSD1306 and Adafruit GFX | BSD | Demo board display driver |
| [PlatformIO](https://platformio.org) and the [Arduino core for ESP32](https://github.com/espressif/arduino-esp32) | Apache-2.0, LGPL-2.1 | Building and flashing both firmware projects |

## Status, Oct 26 (Day 1)

| Piece | State |
|---|---|
| Router loop on Muse Spark (Responses API) | Built and tested against a fake API. Not yet run against the live API. |
| Backup router on Muse Glimmer (Chat Completions) | Built and tested against a fake API. |
| `ask_coder` on Muse Wonder | Diff extraction, `git apply --check` and sandbox compile are tested. The live model call isn't yet. |
| Voice front end | The request format was checked against the live API on Oct 26 and is tested against a fake server. Not yet tried with the pedal and a live mic. |
| Budget guard | Built and tested. Spend persists in `.wbc/spend.json`. |
| FK/IK | Tested on the real SO-101 URDF. 200 of 200 bench poses round-trip to within 0.2 mm and 0.5°, at about 33 ms per solve. |
| STS3215 client and gripper probe | Packet format, decoding, the squeeze ramp and torque release are tested against a fake servo. Not yet run on the arm. |
| Force sensor sampler and tools (Oct 27) | The firmware runs on the ESP32: 500.0 Hz, no dropped frames and no CRC errors. On a real sensor, creep measured +6.8% over 3 s and +28% over 60 s under 51 g. Calibration is rough so far (coins up to 51 g, errors up to 27%), and readings at 34 g and 51 g came out about the same. That's an open question to settle with heavier weights before the jaw pads. |
| Perception, motion layer, slip detector, web HUD | Not started (blueprint sheet 10). Their tools return a clear "not wired yet" error, or use the recorded bench with `--sim`. |

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

Then type `Zero response on SPI. Check my wiring against the firmware.` With `--sim`, `read_breadboard` returns the recorded Act 1 scene from `fixtures/` and `point_at` pretends to move. That exercises the real router and the real Wonder coder with no hardware attached.

- `--voice` also listens on the foot pedal (`ears.pedal_key`, default F13). The first time, macOS asks you to allow Input Monitoring for your terminal. The Mac Studio has no built-in microphone, so plug one in.
- `--backup` routes with Muse Glimmer on your own machine. Serve it with any OpenAI-compatible server that supports tool calls (llama.cpp's `llama-server`, LM Studio or Ollama), then set `backup.base_url` and `backup.model` in `config/models.yaml`. The 4-bit build fits easily in 64 GB.

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
- LeRobot reports joints in its own calibrated units. Check every joint's sign against a ruler before trusting IK on the real arm.

## Check on Day 1

- Model ids and prices: run `scripts/smoke_test.py` and compare with Meta's pricing page. The Spark 1.3, Wonder and Voice Transcribe prices in `models.yaml` come from third-party write-ups.
- The SAM 3.1 request format, which isn't in this scaffold yet. Also request access to the gated `facebook/sam3.1` weights.
