"""Console HUD: prints the agent timeline with the model behind every step.

The web HUD (FastAPI + WebSocket, sheet 03) will implement the same four methods.
"""
from __future__ import annotations

import json


def _short(value, limit: int = 90) -> str:
    text = value if isinstance(value, str) else json.dumps(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


class ConsoleHud:
    def call(self, label: str, name: str, arguments: str) -> None:
        print(f"  [{label}] → {name} {_short(arguments)}", flush=True)

    def result(self, label: str, name: str, result: dict, seconds: float) -> None:
        status = "ok" if result.get("ok", True) else f"FAILED: {_short(result.get('error', ''), 120)}"
        print(f"  [{label}] ← {name} {status} ({seconds:.1f} s)", flush=True)

    def note(self, text: str) -> None:
        print(f"  · {text}", flush=True)

    def reply(self, label: str, text: str) -> None:
        print(f"\n[{label}] {text}\n", flush=True)
