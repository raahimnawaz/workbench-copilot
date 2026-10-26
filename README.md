# Workbench Co-Pilot

An AI lab partner that lives in your code editor and on your workbench. You talk to it through a foot pedal. **Muse Spark** routes each request: **Muse Wonder** writes code, **SAM 3.1** reads the breadboard, **Muse Voice Transcribe** hears you, and an **SO-101** arm points, grasps and holds boards while you probe.

Built for the Meta Global AI Developer Hackathon (Indie track, Push the Frontier), Oct 26 – Nov 4, 2026.

## Built during the hackathon, and what it builds on

Everything in this repository was written during the hackathon, starting Oct 26, 2026, and the commit history dates every change. The one exception is the vendored URDF listed below.

| Built here | What it does |
|---|---|
| `wbc/agent/` | Router loop, tool registry, `ask_coder`, voice front end, budget guard, console HUD |
| `wbc/motion/kinematics.py` | URDF chain loader, forward kinematics, damped least-squares IK, the (x, y, z, pitch, roll) grasp frame |
| `config/`, `fixtures/`, `scripts/`, `tests/` | Model routing config, the recorded bench for `--sim`, the smoke test, 45 tests |
| `firmware/dut_demo/` | The Act 1 demo board firmware, with the wiring bug planted on purpose |

| Reused | License | Used for |
|---|---|---|
| `hardware/so101_new_calib.urdf`, from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) | Apache-2.0 | The arm's kinematic chain (unchanged copy) |
| [LeRobot](https://github.com/huggingface/lerobot) | Apache-2.0 | Servo calibration and the Feetech bus (from Oct 27) |
| Muse Spark, Muse Wonder, Muse Voice Transcribe, SAM 3.1 on the Meta Model API | Meta API terms | Routing, coding, speech, segmentation |
| Muse Glimmer 30B weights | Apache-2.0 | The local backup router |
| openai, numpy, PyYAML, httpx, sounddevice, pynput, pytest | Apache-2.0, BSD-3, MIT, BSD-3, MIT, LGPL-3.0, MIT | Python dependencies |
| Adafruit SSD1306 and Adafruit GFX | BSD | Demo board display driver |

## Status, Oct 26 (Day 1)

| Piece | State |
|---|---|
| Router loop on Muse Spark (Responses API) | Built and tested against a fake API. Not yet run against the live API. |
| Backup router on Muse Glimmer (Chat Completions) | Built and tested against a fake API. |
| `ask_coder` on Muse Wonder | Diff extraction, `git apply --check` and sandbox compile are tested. The live model call isn't yet. |
| Voice front end | WAV packing and upload are tested against a fake server. The endpoint and form fields are marked `VERIFY`. |
| Budget guard | Built and tested. Spend persists in `.wbc/spend.json`. |
| FK/IK | Tested on the real SO-101 URDF. 200 of 200 bench poses round-trip to within 0.2 mm and 0.5°, at about 33 ms per solve. |
| Perception, servo bus, reflex loop, web HUD | Not started (blueprint sheet 10). Their tools return a clear "not wired yet" error, or use the recorded bench with `--sim`. |

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

## How the router works

Muse Spark is the only model that talks to you and the only one that decides. Every other model sits behind a tool:

- `ask_coder` sends a task, the bench facts and up to 4 files to Muse Wonder. The diff it returns must pass `git apply --check`, and, if `coder.compile_cmd` is set, a PlatformIO build in a sandbox copy. Only then does the router see it, as a `patch_id`.
- `propose_patch(patch_id)` applies the diff to the working tree and opens VS Code at the first changed line. It never commits or flashes.
- Tool calls in one round run in parallel, so the arm can point while Wonder writes.
- Every model id and price lives in `config/models.yaml`. The budget guard stops the agent at `budget.limit_usd`, which defaults to $120 of the $150 credit.

## Kinematics notes

- In the URDF the gripper approaches along its z-axis. `config/robot.yaml` rotates the tool frame so the approach is x.
- The point between the jaws moves ±8 mm sideways as the wrist rolls, so a 5-joint arm can't always face exactly along the target's bearing. `solve_task` lets yaw move up to 20° and reports the result in `IKResult.yaw`. Near a top-down grasp, a yaw change also turns the jaws, so the grasp skill should correct roll by it when jaw angle matters.
- Poses behind the base need the arm folded back over itself, and random starts rarely find that solution. Pass `q0`, the current joint angles, when you need continuity.
- LeRobot reports joints in its own calibrated units. Check every joint's sign against a ruler before trusting IK on the real arm.

## Check on Day 1

- Model ids and prices: run `scripts/smoke_test.py` and compare with Meta's pricing page. The Spark 1.3, Wonder and Voice Transcribe prices in `models.yaml` come from third-party write-ups.
- The Voice Transcribe endpoint, form fields and response shape (marked `VERIFY` in `wbc/agent/ears.py`).
- The `reasoning` parameter name for Muse Wonder (`wbc/agent/coder.py`).
- The SAM 3.1 request format, which isn't in this scaffold yet. Also request access to the gated `facebook/sam3.1` weights.
