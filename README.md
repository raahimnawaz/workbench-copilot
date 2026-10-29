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
| `scripts/set_servo_ids.py` | Gives each servo of an already-assembled arm its id, growing the chain from the base |
| `scripts/first_moves.py` | The arm's first powered moves: torque on without a jump, then each joint a little, recording which way it turns |
| `scripts/power_cut.py` | The power cut-off test: switching the servo supply off makes the arm let go, and it stays limp when the power returns |
| `wbc/reflex/fsr_link.py`, `wbc/reflex/fsr_cal.py` | The force sensor link (500 Hz frames with a CRC), per-sensor calibration fit and creep analysis |
| `wbc/perception/` | The SAM 3.1 client (one noun phrase per request, sent in parallel; masks decoded and drawn), camera capture, and the board reader: the carrier's tags, the top-down board view, the hole grid fit, wire tracing, the netlist and its check against `pins.h` |
| `scripts/sam_probe.py`, `scripts/read_board.py`, `scripts/make_carrier.py` | What SAM 3.1 finds in a still; the board reader on a still; the printable carrier sheet |
| `config/`, `fixtures/`, `scripts/`, `tests/` | Model routing config, the recorded bench for `--sim`, the smoke test, gripper probe and force sensor tools, 207 tests |
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
| Adafruit SSD1306 and Adafruit GFX | BSD | Demo board display driver |
| [PlatformIO](https://platformio.org) and the [Arduino core for ESP32](https://github.com/espressif/arduino-esp32) | Apache-2.0, LGPL-2.1 | Building and flashing both firmware projects |

## Status, Oct 26 (Day 1)

| Piece | State |
|---|---|
| Router loop on Muse Spark (Responses API) | Runs live. On Oct 28 a spoken Act 1 on the recorded bench took about 63 s in four router rounds. Code now fetches the bench facts first and stages checked patches itself, and the same turn takes 25 s in two rounds. |
| Backup router on Muse Glimmer (Chat Completions) | Built and tested against a fake API. |
| `ask_coder` on Muse Wonder | Runs live: Wonder wrote the Act 1 fix in 3–7 s on Oct 28. The fix compiles in a sandbox copy in about 8 s once its build cache is warm (about 14 s cold). |
| Voice front end | The request format was checked against the live API on Oct 26. The wake phrase and the loudness gate that cuts the mic into utterances are tested on synthetic audio, and on Oct 28 they worked on the live mic (the Razer, run from Terminal.app). |
| Budget guard | Built and tested. Spend persists in `.wbc/spend.json`. |
| FK/IK | Tested on the real SO-101 URDF. 200 of 200 bench poses round-trip to within 0.2 mm and 0.5°, at about 33 ms per solve. |
| STS3215 client and gripper probe | Packet format, decoding, the squeeze ramp and torque release are tested against a fake servo. Not yet run on the arm. |
| Force sensor sampler and tools (Oct 27) | The firmware runs on the ESP32: 500.0 Hz, no dropped frames and no CRC errors. On a real sensor, creep measured +6.8% over 3 s and +28% over 60 s under 51 g. Calibration is rough so far (coins up to 51 g, errors up to 27%), and readings at 34 g and 51 g came out about the same. That's an open question to settle with heavier weights before the jaw pads. |
| SAM 3.1 client (Oct 28) | Built to Meta's documented request and reply format and tested offline on replies written with Meta's own mask encoder. Not yet called live. |
| Board reader (Oct 28) | Built and tested on a synthetic bench photo: the carrier's tags, the top-down view, the hole grid fit, wire tracing, the netlist and the check against `pins.h`. Not yet run on a real photo, and `bench.yaml` doesn't place the modules until the demo board is wired. |
| Motion layer, slip detector, web HUD | Not started (blueprint sheet 10). Their tools return a clear "not wired yet" error, or use the recorded bench with `--sim`. |

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
5. **Netlist and `pins.h`.** Each row's a–e and f–j strips and each rail is a node, wires join nodes, and the ESP32's and display's pins sit in holes given in `config/bench.yaml`. Plain code then compares each display pin's GPIO with `pins.h` and reports the mismatch, its wire and hole, and the line to fix.

It reads flat pre-cut jumpers. Loose jumper leads stand about 15 mm off the board, and from a camera 30 cm up their tips look up to 4 mm away from where they go in, more than a hole.

```bash
python scripts/make_carrier.py          # .wbc/vision/carrier.pdf: print at 100%, check the 100 mm bar
python scripts/read_board.py check      # free: the tags, and the fitted hole grid drawn over the board
python scripts/read_board.py read       # $0.0025: the whole read, drawn in .wbc/vision/board-<time>.png
```

- Both take a still from the camera in `bench.yaml` and keep it as `.wbc/vision/photo-<time>.png`, or read one with `--photo`. Run them from Terminal.app.
- `config/bench.yaml` holds the camera, the carrier's tags, each board's place on the sheet, where each module's headers start, and which display pin each `pins.h` signal drives. `config/parts.yaml` holds the breadboard's geometry and the module pinouts. Change those and print the carrier again; never edit the sheet.
- `reader.tip_inset_mm` moves each wire end back from the tip of its mask, for jumpers whose masks stop short of their holes or run past them. Set it from the golden photos' residuals.
- Each read saves the drawing, the top-down view and the result, with SAM's whole reply, as `.wbc/vision/board-<time>.*`.

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
