"""Run the bench agent: python -m wbc.agent [--sim] [--voice] [--backup]

Typed lines and pedal transcripts go on one queue, and the router handles them one turn at a time.
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
from pathlib import Path

from wbc import config
from wbc.agent import ears
from wbc.agent.budget import Budget, BudgetExceeded
from wbc.agent.hud import ConsoleHud
from wbc.agent.loop import ChatRouter, Router
from wbc.agent.tools import Bench, build_toolbox

SYSTEM = (Path(__file__).parent / "prompts" / "system.md").read_text(encoding="utf-8")


def build(sim: bool, backup: bool):
    models = config.load_yaml("models.yaml")
    budget = Budget.from_config(models, config.ROOT)
    hud = ConsoleHud()
    meta = config.meta_client(models)  # the coder always runs on the Meta Model API
    toolbox = build_toolbox(Bench(config.ROOT, models, meta, budget, sim=sim))
    max_steps = models["router"].get("max_steps", 12)
    if backup:
        b = models["backup"]
        router = ChatRouter(config.backup_client(models), b["model"], toolbox, budget, hud, SYSTEM,
                            max_steps=max_steps, reasoning=b.get("reasoning"))
    else:
        r = models["router"]
        router = Router(meta, r["model"], toolbox, budget, hud, SYSTEM, max_steps=max_steps,
                        fallback=r.get("fallback"))
    return models, budget, hud, router


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wbc", description=__doc__.splitlines()[0])
    ap.add_argument("--sim", action="store_true", help="use the recorded bench instead of cameras and arm")
    ap.add_argument("--voice", action="store_true", help="listen on the foot pedal as well as typed input")
    ap.add_argument("--backup", action="store_true", help="route with the local open-weights model")
    args = ap.parse_args(argv)

    try:
        models, budget, hud, router = build(args.sim, args.backup)
    except RuntimeError as e:  # missing key or voice extras
        print(e, file=sys.stderr)
        return 2
    hud.note(f"router: {router.label} ({router.model}) · budget left ${budget.remaining:.2f}"
             + (" · SIM bench" if args.sim else ""))
    inbox = queue.Queue()

    def typed() -> None:
        for line in sys.stdin:
            if line.strip():
                inbox.put(line.strip())
        inbox.put(None)

    threading.Thread(target=typed, daemon=True).start()
    if args.voice:
        threading.Thread(target=ears.listen_forever, daemon=True,
                         args=(inbox.put, models["ears"], config.api_key(), budget, hud)).start()

    while True:
        text = inbox.get()
        if text is None:
            return 0
        try:
            router.run_turn(text)
        except BudgetExceeded as e:
            hud.note(str(e))
            return 1
        except Exception as e:
            hud.note(f"turn failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
