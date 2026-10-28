"""Run the bench agent: python -m wbc.agent [--sim] [--voice] [--hear] [--backup]

Typed lines and spoken requests go on one queue, and the router handles them one turn at a time.
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
                        fallback=r.get("fallback"), effort=r.get("effort"))
    return models, budget, hud, router


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wbc", description=__doc__.splitlines()[0])
    ap.add_argument("--sim", action="store_true", help="use the recorded bench instead of cameras and arm")
    ap.add_argument("--voice", action="store_true", help='listen for "Copilot, ..." as well as typed input')
    ap.add_argument("--hear", action="store_true",
                    help="mic check: print what the mic hears and route nothing (implies --voice)")
    ap.add_argument("--backup", action="store_true", help="route with the local open-weights model")
    args = ap.parse_args(argv)

    try:
        models, budget, hud, router = build(args.sim, args.backup)
    except RuntimeError as e:  # missing key or voice extras
        print(e, file=sys.stderr)
        return 2
    hud.note(f"router: {router.label} ({router.model}) · spent ${budget.spent:.2f}, "
             f"stops at ${budget.limit:.0f}" + (" · SIM bench" if args.sim else ""))
    try:
        budget.check()
    except BudgetExceeded as e:
        hud.note(str(e))
        return 1
    warning = budget.pop_warning()
    if warning:
        hud.note(warning)
    inbox = queue.Queue()

    def typed() -> None:
        for line in sys.stdin:
            if line.strip():
                inbox.put(line.strip())
        inbox.put(None)

    threading.Thread(target=typed, daemon=True).start()
    if args.voice or args.hear:
        on_text = (lambda text: None) if args.hear else inbox.put  # listen_forever prints what it heard
        threading.Thread(target=ears.listen_forever, daemon=True,
                         args=(on_text, models["ears"], config.api_key(), budget, hud)).start()
    if args.hear:
        hud.note("mic check: nothing goes to the router. Ctrl-D quits.")

    while True:
        text = inbox.get()
        if text is None:
            return 0
        if args.hear:
            continue
        try:
            router.run_turn(text)
        except BudgetExceeded as e:
            hud.note(str(e))
            return 1
        except Exception as e:
            hud.note(f"turn failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
