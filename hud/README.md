# The HUD's orb page

`orb-light.html` is the HUD's face for the co-pilot: the Muse orb, the model at work, the request, the reply, and a bar of model calls with their times. It needs only `saved/siri-line.js` beside it: there's no build step and nothing to install. `CREDITS.md` lists what it borrows, and the README's built-vs-reused table has the code it adapts.

## Live, from the agent

```bash
.venv/bin/python -m wbc.agent --sim --speak --hud
```

Then open <http://127.0.0.1:8766/> in a current Chrome. `--hud 8800` picks another port, if 8766 is taken. The agent serves the page and its live feed itself, with only the standard library (`wbc/agent/web_hud.py`), and the page shows each turn as it happens:

- **Someone talking at the mic** (with `--voice`): listening, under Voice Transcribe, then the transcription's time in the call bar.
- **The request**, in quotes. The call bar starts again for each turn.
- **Every model call**, under its model's name and in its colors from the moment it starts, then its time in the call bar: Spark routing, Wonder writing code, SAM reading the board or finding things on the desk, Transcribe hearing. Calls to one model that overlap, like SAM's phrases, make one entry. They're timed where each call goes out, so only real calls show: with `--sim` the board read is the recording, so no SAM appears.
- **The reply**, a sentence at a time, and speaking while it's said out loud (`--voice` or `--speak`). A turn that fails shows that instead.
- **The turn's time and spend**, closing the call bar: `Turn 1 · 25 s · $0.02`. A spoken turn counts from the end of the request, its transcription included.

A page opened mid-run catches up with the turn so far, and it reconnects by itself after the agent restarts. Its own text box doesn't reach the agent yet, so live it does nothing: type in the terminal, or speak. With `--backup`, the router shows as Spark Open, the page's open-weights model.

## On its own

From this folder:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

Then open <http://localhost:8765/orb-light.html>. About 1.5 seconds after loading, it plays a mocked Act 1 so you can see every state.

- **Keys:** A plays Act 1 again, H hides the controls (do this before recording), C opens the camera panel.
- **Controls along the top:** they switch the state and the model by hand. Mic feeds your real voice to the line while it listens.

## Driving it

These functions are globals in the page's script. Live, each event the agent sends is a JSON array of a function's name and its arguments, such as `["call", "wonder", 4.1]`, and `["clear"]` empties the call bar for a new turn.

| Call | What it does |
|---|---|
| `setState(s)` | `'idle'`, `'listening'`, `'thinking'` or `'speaking'`. The sea, the line's waves and the voice glow follow it. |
| `setModel(m)` | `'transcribe'`, `'spark'`, `'sam'`, `'wonder'` or `'open'` (Spark Open on Baseten). Sets the orb's colors, the model's name and its role line. |
| `you(text)` | Shows what the user said, in quotes. `you('')` clears it. |
| `say(html)` | Shows Muse's reply, a sentence at a time. `say('')` clears it. |
| `begin(m)` | Adds model `m` to the call bar at the bottom when it starts. |
| `call(m, seconds)` | Fills in that call's time, or adds the entry if `begin` wasn't called. |
| `total(text)` | Adds a closing line to the call bar, like `Act 1 · 21 s · $0.03`. |

- **Models' names, roles and colors:** in `MODELS` near the top of the script. `page_models` in `wbc/agent/web_hud.py` says which model in `config/models.yaml` is which.
- **The orb's palette:** `NAVY` (the primary blue) and `MUSE` (Muse blue) are under `MODELS`.
