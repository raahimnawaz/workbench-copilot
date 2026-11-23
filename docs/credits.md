# What was built here, and what it reuses

Back to the [README](../README.md).

## Built during the hackathon, and what it builds on

Everything in this repository was written during the hackathon, starting Oct 26, 2026, and the commit history dates every change. The exceptions are the vendored URDF and the two pieces of MIT-licensed code the HUD's page adapts, all listed below.

| Built here | What it does |
|---|---|
| `wbc/agent/` | The router loop on Muse Spark, its tools, `ask_coder` on Muse Wonder, the demo board's serial link, the voice front end (wake phrase, spoken stop, short answers), spoken replies, the budget guard and the console HUD |
| `wbc/agent/memory.py` | Sessions that survive a restart and a full context, and the `remember` / `recall` / `forget` notebook |
| `wbc/agent/fast_arm.py` | The arm's fast path: plain rules and a local Llama 3.2 3B have to agree on an arm command, which then runs with no Spark call |
| `scripts/toolformer_data.py`, `toolformer_train.py`, `toolformer_mlx.py` | Fine-tuning the fast path's Llama after Meta's Toolformer: Spark writes and labels spoken requests in the fast path's own format, a label is kept only when its call runs on the recorded bench, and a LoRA is trained on Modal and converted for `mlx_lm.server` |
| `hud/`, `wbc/agent/web_hud.py` | The HUD's orb page (canvas and WebGL, no packages) and its live feed over Server-Sent Events: the mic listening, each request, every model call with its time, the reply and its speech, and each turn's time and spend. `hud/CREDITS.md` lists what the page borrows |
| `wbc/motion/kinematics.py` | URDF chain loader, forward kinematics, damped least-squares IK, the (x, y, z, pitch, roll) grasp frame |
| `wbc/motion/joints.py`, `wbc/motion/trajectory.py` | Raw servo ticks to the URDF's joint angles, with signs checked by hand on the arm, and quintic moves timed under joint and tip speed caps |
| `wbc/motion/arm.py`, `wbc/motion/sts.py` | The arm driver, the one owner of the servo bus: power-on without a jump, moves in URDF angles, refusals before anything moves, every motor cut when a joint strays, re-aim, grip and release. And a small STS3215 client for the read-only tools |
| `wbc/motion/hand.py`, `wbc/motion/table.py` | Picking a thing up and putting it down: every pose solved before the first one moves, the fixed jaw beside the thing, and floors from the desk as the arm measured it |
| `wbc/motion/carrier.py`, `scripts/register_arm.py` | Registering the arm to the carrier from points touched with the motors off, and the touch test |
| `wbc/perception/` | The SAM 3.1 client, camera capture, the board reader (the carrier's tags, the top-down view, the hole grid fit, wire tracing, the netlist and its check against `pins.h`) and the desk camera's `locate` |
| `scripts/` | The smoke test, the board reader and its carrier sheet, the desk camera, the joint map check and arm registration |
| `wbc/perception/schematic.py`, `scripts/make_schematic.py`, `hardware/*.kicad_sch` | The demo boards' KiCad schematics, drawn from KiCad's own symbols, and `read_schematic`: their nets and KiCad's electrical rules check, through `kicad-cli` |
| `wbc/agent/parts.py` | `add_part`: Muse Spark Open teaches itself a part from `parts.yaml` (its wiring to the DevKit's free GPIOs, its library and setup code, saved as a skill in `.wbc/skills/`), and the part is drawn into a copy of the schematic in `.wbc/schematics/`, kept only if KiCad's electrical rules check passes, and shown in KiCad with `--show` |
| `scripts/kokoro_server.py`, `scripts/kokoro-say` | The co-pilot's voice: Kokoro-82M kept loaded on this Mac, and a stand-in for `say` that plays its audio |
| `config/`, `fixtures/`, `tests/` | Model, bench and arm config, the recorded bench for `--sim`, 908 offline tests |
| `firmware/dut_demo/` | The demo board: lights an LED, keeps it on and says so over serial |

| Reused | License | Used for |
|---|---|---|
| `hardware/so101_new_calib.urdf`, from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) | Apache-2.0 | The arm's kinematic chain (unchanged copy) |
| [LeRobot](https://github.com/huggingface/lerobot) | Apache-2.0 | Servo calibration and the Feetech bus |
| The dot lattice from [thinking-orbs](https://github.com/Jakubantalik/thinking-orbs) | MIT | The orb's rings of dots in `hud/orb-light.html`, adapted to turn on the Siri line and ride our own waves (the notice is at the end of the page) |
| The bloom and sun rays from [WebGL-Fluid-Simulation](https://github.com/PavelDoGreat/WebGL-Fluid-Simulation) | MIT | The light streaming from the orb's dots in `hud/orb-light.html`, adapted to carry the dots' colors (the notice is at the end of the page) |
| Muse Spark, Muse Wonder, Muse Voice Transcribe, SAM 3.1 on the Meta Model API | Meta API terms | Routing, coding, speech, segmentation |
| [meta-sam-parser](https://pypi.org/project/meta-sam-parser/) 0.1, from Meta | SAM License | Reading SAM 3.1's replies: the token grammar and the mask codec (installed from PyPI, not vendored) |
| Llama 3.2 3B Instruct, 4-bit MLX (`mlx-community/Llama-3.2-3B-Instruct-4bit`), served by mlx-lm | Llama 3.2 Community License, and MIT for mlx-lm. **Built with Llama** | The arm's fast path: a second opinion on spoken arm commands, on this Mac |
| Llama 3.2 3B Instruct in full precision, from unsloth's ungated copy (`unsloth/Llama-3.2-3B-Instruct`) | Llama 3.2 Community License. **Built with Llama** | The base the Toolformer-style fine-tune trains its LoRA on (`scripts/toolformer_train.py`) |
| PyTorch, Hugging Face Transformers, PEFT, Accelerate and safetensors | BSD-3-Clause; Apache-2.0 for the other four | The fine-tune on Modal, and turning its adapter into one `mlx_lm.server` loads |
| [Modal](https://modal.com) and its Python client | Modal's terms; the client Apache-2.0 | The A10G the fine-tune runs on (`pip install -e '.[train]'`) |
| [AprilTag](https://april.eecs.umich.edu/software/apriltag)'s tag36h11 family, as built into OpenCV's ArUco module | BSD-2-Clause | The carrier's and the desk's tags |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) weights, run by [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) with onnxruntime and espeak-ng (through espeakng-loader and phonemizer) | Apache-2.0; MIT; MIT; GPL-3.0 (espeak-ng, phonemizer) | The co-pilot's voice (Meta's API has no text-to-speech model), in its own venv: `pip install kokoro-onnx==0.6.1` |
| [Visual Studio Code](https://code.visualstudio.com)'s `code` command | Microsoft's product license | `--show` opens `pins.h` as Wonder's fix lands in it, and the skill `add_part` saves |
| macOS `say` | Part of macOS | The voice only when `mouth.command` is unset: its voices can't be in a published video |
| [KiCad](https://www.kicad.org) 10 and its symbol libraries | GPL-3.0; CC-BY-SA-4.0 with KiCad's exception for designs | `kicad-cli` exports the schematics' netlists and runs their electrical rules checks; the symbols are drawn into `hardware/*.kicad_sch` |
| Muse Spark Open (`meta-models/muse-spark-open-1026`) on [Baseten](https://www.baseten.co) | Meta's open weights, served by Baseten | Questions no tool is needed for (`open` in `models.yaml`) |
| openai, numpy, PyYAML, httpx, sounddevice, pyserial, pytest, OpenCV, Pillow | Apache-2.0, BSD-3, MIT, BSD-3, MIT, BSD-3, MIT, Apache-2.0, MIT-CMU | Python dependencies |
| [PlatformIO](https://platformio.org) and the [Arduino core for ESP32](https://github.com/espressif/arduino-esp32) | Apache-2.0, LGPL-2.1 | Building and flashing the demo board's firmware |
