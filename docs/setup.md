# Setup and running

How to install the co-pilot, run it with or without the bench, and what hardware the full bench uses. Back to the [README](../README.md).

## Setup (Apple Silicon)

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e '.[voice,dev]'
cp .env.example .env        # then put your META_API_KEY in .env
pytest
python scripts/smoke_test.py
```

The smoke test makes one small call to each hosted model and reports which model ids your key can see. Run it before anything else. The arm needs the `arm` and `lerobot` extras, the cameras `eyes`, and flashing the demo board `firmware`.

## Run

```bash
python -m wbc.agent --sim
```

Then type `My LED isn't on. Check my wiring against the firmware.` With `--sim`, `read_breadboard` returns the recorded LED scene from `fixtures/`, and `build_and_flash` pretends. That runs the real router and the real Wonder coder with no hardware attached.

- `--voice` also listens on the bench mic (`ears.input_device`). Say "Copilot," and your request, or "Copilot." and then the request. Every utterance is transcribed, and only the ones addressed to the co-pilot go to the router. After a reply that asks something, a short answer such as "Yes." or "Go." needs no "Copilot," for 8 s. `--hear` is a mic check: it prints what it heard and routes nothing. The first time, macOS asks to let your terminal use the microphone.
- With `--voice`, replies are said out loud: the first sentence or two of each, while the HUD shows all of it. The mic drops what it hears while the co-pilot speaks, so it never answers itself. `--no-speak` keeps replies on the HUD, and `--speak` says them without `--voice`. The voice is Kokoro-82M on this Mac: install it in a venv of its own with a short path (`python3 -m venv ~/kokoro-venv && ~/kokoro-venv/bin/pip install kokoro-onnx==0.6.1`), and leave `~/kokoro-venv/bin/python -I scripts/kokoro_server.py` running (`--models` names the folder with kokoro-onnx's v1.0 model files). `mouth:` in `config/models.yaml` picks the voice (the server's `/voices` lists them).
- `--hud` shows each turn live on the HUD's orb page: open <http://127.0.0.1:8766/> in Chrome (`--hud 8800` picks another port). `hud/README.md` has the details.
- `--show`, for a take on macOS, opens the HUD (with `--hud`) as a Chrome app window of its own, with no tabs or address bar, and the board's schematic in KiCad's Schematic Editor. The window closes when the agent ends (`wbc/agent/show.py`).
- For a recorded take, a few switches cut the dead air, all off by default: `hand.idle_at_start` in `config/robot.yaml` (the arm waits in its idle pose from the start), and in `config/models.yaml` `ears.every_utterance` (no wake phrase, with `ears.hangover_s` 1.4 so a slow question isn't cut in two) and a `take:` section with `warm_at_start` (one tiny call to each model), `alongside` (an arm command runs while another turn does) and `lanes` (a "go" to an offered flash, adding the screen and arm commands run together, with no planning call to Spark).
- `--arm` lets the arm's tools move it. With `hand.go_ahead: request` in `config/robot.yaml`, your request is the go-ahead and the arm moves at once. "Copilot, stop", or "Stop!" on its own, goes straight from the ears to the arm, even mid-turn: it holds where it is with its motors on, and the requests waiting behind it are dropped. `--enter` makes each move wait for Enter at the keyboard, for bench tests. Ctrl-C cuts every motor and the arm drops, so keep a hand near it. `--no-prefetch` skips the board checks (the board read and `pins.h`) in a session that only moves the arm.
- Arm commands can skip Spark. Serve Llama 3.2 3B on this Mac in its own venv: `pip install mlx-lm==0.32.2`, then `mlx_lm.server --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8090 --prompt-cache-bytes 536870912` (the first run downloads 1.8 GB). While it runs, a request naming a thing and a place the rules know (`fast_arm:` in `config/models.yaml`) runs at once if Llama agrees on the kind of command, and the result is said when the move is done. Everything else goes to Spark with no extra wait. `--no-fast-arm` sends everything to Spark, to compare.
- Every turn is recorded in `.wbc/session.json`, however it ends. Each run starts a new session; `--resume` picks up the last one, and `--sim` keeps its own. A conversation that grows too long, or fails twice in a row, becomes a summary that the next one starts from, and code never runs a tool twice: after a tool has run, it asks what's still to do. The turns a summary folds away are kept, and a request that mentions one brings it back whole, with the exchange around it, in that turn's own message, so "which pin was the LED on earlier?" finds the turn that said so while the instructions stay the same from turn to turn. Matches are scored by BM25 (a rare word counts most, whole words only, evened out for length), by age at 1 / ln(e + age) for up to half, so an old match fades slowly instead of falling off, and by what the turn did: one that changed the bench counts 1.5 times talk. Past the newest 20 notes, an old note the request is about reaches its turn the same way. Spark Open sees the last six turns, so a question can refer back to them.
- The router keeps a notebook in `.wbc/notebook.jsonl` with its `remember`, `recall` and `forget` tools, and the newest 20 notes go into every call. A note can say where a thing goes on the desk, with the short name `locate` should look for in quotes (the screwdriver goes on the "sticky note"), so "put the screwdriver away" works in a later session. A note never moves the arm by itself.

## Hardware

| Part | Role |
|---|---|
| SO-101 arm, six Feetech STS3215 servos (a 1/345 elbow) on a 1 Mbps bus | picks things up and puts them away |
| ESP32 DevKit (30-pin DOIT) on a breadboard, an LED and 330 Ω | the demo board the co-pilot debugs and flashes |
| Two Logitech C920 webcams | one over the breadboard, one 71 cm over the desk |
| Printed AprilTag sheets (tag36h11) | map each photo to millimeters and the desk to the arm's frame |
| Razer Seiren Mini microphone | hands-free voice |
| A Mac with Apple Silicon | runs the agent, Llama 3.2 3B, the Kokoro voice and KiCad |
| SSD1306 0.96" OLED screen | the part the co-pilot teaches itself to add |
