"""Console HUD: prints the agent timeline with the model behind every step.

The web HUD (web_hud.py, `--hud`) prints the same and shows each turn on the orb page too. What only the page needs
comes in through the methods after reply(), which do nothing here.
"""
from __future__ import annotations

import contextlib
import json


def _short(value, limit: int = 90) -> str:
    text = value if isinstance(value, str) else json.dumps(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


class ConsoleHud:
    def call(self, label: str, name: str, arguments: str) -> None:
        print(f"  [{label}] → {name} {_short(arguments)}", flush=True)

    def result(self, label: str, name: str, result: dict, seconds: float) -> None:
        status = "ok" if result.get("ok", True) else f"FAILED: {_short(result.get('error', ''), 120)}"
        split = "".join(f", {part} {s:.1f} s" for part, s in (result.get("_timing") or {}).items())
        print(f"  [{label}] ← {name} {status} ({seconds:.1f} s{split})", flush=True)

    def note(self, text: str) -> None:
        print(f"  · {text}", flush=True)

    def reply(self, label: str, text: str) -> None:
        print(f"\n[{label}] {text}\n", flush=True)

    def watch(self, client):
        """A model client whose calls the HUD times; here, the client itself."""
        return client

    def calling(self, model: str):
        """`with hud.calling(model_id):` around one model call."""
        return contextlib.nullcontext()

    def hearing(self, on: bool) -> None:
        """Someone has started (or stopped) talking at the mic."""

    def heard(self, request: str) -> None:
        """The ears heard a request, just transcribed; its turn comes next."""

    def turn(self, text: str) -> None:
        """A request, typed or heard, starts a turn."""

    def turn_over(self, failure: str = "") -> None:
        """The turn has ended; `failure` is what to show instead of a reply when it failed."""

    def speaking(self, on: bool) -> None:
        """The co-pilot has started (or stopped) saying a reply out loud."""
