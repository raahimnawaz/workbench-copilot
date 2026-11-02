"""Run the bench agent: python -m wbc.agent [--sim] [--voice] [--hear] [--speak] [--backup] [--arm] [--no-prefetch]

Typed lines and spoken requests go on one queue, and the router handles them one turn at a time. Before each turn,
code fetches the bench facts in models.yaml's router.prefetch (Act 1's board read, pins.h and the sensor);
--no-prefetch skips them, for a session that only moves the arm. With --voice (or
--speak), the co-pilot says the first sentence or two of each reply out loud, and the mic is muted while it speaks.
With --arm, the arm's tools may move it, each move only after Enter at the keyboard: while a move waits, the next
typed line answers it instead of going to the router, and nothing heard by voice does. Ctrl-C cuts every motor (the
arm drops, so keep a hand near it) and quits at once, and no arm move starts after it. At the end of the input (Ctrl-D)
a question still waiting gets a no, no arm move starts, and once the turn ends the arm goes back to where it rested and
its motors go off; any other exit asks before moving it.

Every turn is recorded in .wbc/session.json (wbc/agent/memory.py), however it ends. Each run starts a new session, and
--resume picks up the last one instead; the notebook the router keeps with `remember` carries over either way.

With --hud, the HUD's orb page shows each turn live at http://127.0.0.1:8766/ (wbc/agent/web_hud.py): the mic
listening, the request, every model call with its time, the reply and its speech, and the turn's time and spend.
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
from pathlib import Path

from wbc import config
from wbc.agent import coder, ears
from wbc.agent.approve import Keyboard
from wbc.agent.budget import Budget, BudgetExceeded
from wbc.agent.hud import ConsoleHud
from wbc.agent.loop import ChatRouter, Router
from wbc.agent.memory import Memory
from wbc.agent.mouth import Mouth
from wbc.agent.tools import Bench, add_memory_tools, build_toolbox
from wbc.agent.web_hud import WebHud

SYSTEM = (Path(__file__).parent / "prompts" / "system.md").read_text(encoding="utf-8")
FAILED = "That turn failed. The error is on the screen."


def build(sim: bool, backup: bool, router_model: str | None = None, arm: bool = False,
          calibration_dir: str | None = None, prefetch: bool = True, resume: bool = False,
          hud_port: int | None = None):
    models = config.load_yaml("models.yaml")
    budget = Budget.from_config(models, config.ROOT)
    hud = ConsoleHud() if hud_port is None else WebHud(models, budget, hud_port, backup=backup)
    meta = hud.watch(config.meta_client(models))  # the coder always runs on the Meta Model API
    robot_cfg = config.load_yaml("robot.yaml")
    keyboard = Keyboard(say=hud.note)
    bench = Bench(config.ROOT, models, meta, budget, sim=sim, say=hud.note, layout=config.load_yaml("bench.yaml"),
                  parts=config.load_yaml("parts.yaml"), robot_cfg=robot_cfg, desk_cfg=config.load_yaml("desk.yaml"),
                  arm_port=robot_cfg.get("port") if arm else None,
                  calibration_dir=calibration_dir or robot_cfg.get("calibration_dir"), confirm=keyboard.ask)
    toolbox = build_toolbox(bench)
    max_steps = models["router"].get("max_steps", 12)
    prefetch = models["router"].get("prefetch") if prefetch else None
    memory = Memory.from_config(models, config.ROOT, resume=resume, sim=sim)
    add_memory_tools(toolbox, memory.notebook)
    if backup:
        b = models["backup"]
        router = ChatRouter(hud.watch(config.backup_client(models)), b["model"], toolbox, budget, hud, SYSTEM,
                            max_steps=max_steps, prefetch=prefetch, memory=memory, reasoning=b.get("reasoning"))
    else:
        r = models["router"]
        router = Router(meta, router_model or r["model"], toolbox, budget, hud, SYSTEM, max_steps=max_steps,
                        prefetch=prefetch, memory=memory, fallback=r.get("fallback"), effort=r.get("effort"))
    return models, budget, hud, router, bench, keyboard


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wbc", description=__doc__.splitlines()[0])
    ap.add_argument("--sim", action="store_true", help="use the recorded bench instead of cameras and arm")
    ap.add_argument("--voice", action="store_true", help='listen for "Copilot, ..." as well as typed input')
    ap.add_argument("--hear", action="store_true",
                    help="mic check: print what the mic hears and route nothing (implies --voice)")
    ap.add_argument("--speak", action=argparse.BooleanOptionalAction,
                    help="say each reply's first sentence or two with macOS say (on with --voice)")
    ap.add_argument("--backup", action="store_true", help="route with the local open-weights model")
    ap.add_argument("--router-model", metavar="ID",
                    help="route with this model instead of router.model, e.g. muse-spark-1.1 to compare speed")
    ap.add_argument("--arm", action="store_true",
                    help="let the arm's tools move it, each move after Enter (the port is robot.yaml's)")
    ap.add_argument("--calibration-dir", metavar="DIR",
                    help="lerobot-calibrate's folder for the arm, when it's another macOS account's")
    ap.add_argument("--no-prefetch", action="store_true",
                    help="skip the bench facts fetched before each turn (Act 1's board read, pins.h and the sensor)")
    ap.add_argument("--resume", action="store_true",
                    help="pick up the last session (.wbc/session.json) instead of starting a new one")
    ap.add_argument("--hud", nargs="?", const=8766, type=int, metavar="PORT",
                    help="show each turn live on the HUD's orb page, at http://127.0.0.1:8766/ (or PORT)")
    args = ap.parse_args(argv)

    try:
        models, budget, hud, router, bench, keyboard = build(args.sim, args.backup, args.router_model, args.arm,
                                                             args.calibration_dir, not args.no_prefetch, args.resume,
                                                             args.hud)
    except RuntimeError as e:  # missing key or voice extras
        print(e, file=sys.stderr)
        return 2
    hud.note(f"router: {router.label} ({router.model}) · spent ${budget.spent:.2f}, "
             f"stops at ${budget.limit:.0f}" + (" · SIM bench" if args.sim else ""))
    memory = getattr(router, "memory", None)
    if memory is not None:
        hud.note(memory.describe())
    try:
        budget.check()
    except BudgetExceeded as e:
        hud.note(str(e))
        return 1
    warning = budget.pop_warning()
    if warning:
        hud.note(warning)
    mouth = None
    if (args.voice if args.speak is None else args.speak) and not args.hear:
        mouth = Mouth(models.get("mouth") or {}, hud=hud)
        if mouth.command is None:
            hud.note("spoken replies off: macOS say isn't on this machine")
            mouth = None
    inbox = queue.Queue()

    def typed() -> None:
        for line in sys.stdin:
            if keyboard.offer(line):  # an arm move waiting for Enter takes it
                continue
            if line.strip():
                inbox.put(line.strip())
        bench.stopped.set()  # the end of the input: no arm move starts now, and a question waiting gets its no
        keyboard.close()
        inbox.put(None)

    threading.Thread(target=typed, daemon=True).start()
    if not args.hear:  # build the firmware once now, so the session's first patch skips the cold build
        threading.Thread(target=coder.compile_in_sandbox, args=("", config.ROOT, models["coder"]), daemon=True).start()
    if args.voice or args.hear:
        on_text = (lambda text: None) if args.hear else inbox.put  # listen_forever prints what it heard
        threading.Thread(target=ears.listen_forever, daemon=True,
                         args=(on_text, models["ears"], config.api_key(), budget, hud),
                         kwargs={"muted": mouth.busy if mouth else None}).start()
    if args.hear:
        hud.note("mic check: nothing goes to the router. Ctrl-D quits.")

    home = True  # at the end of the input (Ctrl-D) the arm goes back to rest before its motors go off, as --help says
    try:
        while True:
            text = inbox.get()
            if text is None:
                return 0
            if args.hear:
                continue
            hud.turn(text)
            try:
                reply = router.run_turn(text)
                if mouth:
                    mouth.speak(reply)
                hud.turn_over()
            except BudgetExceeded as e:
                hud.note(str(e))
                hud.turn_over(str(e))
                # An exit nobody typed: the arm moves home only on Enter, and holds where it is until there's an
                # answer. Once the input has ended (Ctrl-D), before the question or during it, it goes home as ever
                home = bench.can_go_home()
                if home and not keyboard.closed:
                    if mouth:
                        mouth.speak("The agent is stopping. Press Enter at the keyboard to take the arm back to rest.")
                    home = keyboard.ask("The agent is stopping. Take the arm back to rest before its motors go off? "
                                        "Anything else turns them off where it is, so hold the arm.",
                                        forever=True) is not False
                return 1
            except Exception as e:
                hud.note(f"turn failed: {type(e).__name__}: {e}")
                hud.turn_over(FAILED)
                if mouth:
                    mouth.speak(FAILED)
    except KeyboardInterrupt:
        home = False
        keyboard.close()  # a move waiting for Enter gets a no, and so does any that asks later
        bench.stop_arm()
        if mouth:
            mouth.stop()
        hud.note("Ctrl-C: every motor is cut." if bench.hand is not None else "Ctrl-C.")
        record = getattr(router, "record_unfinished", None)
        if record is not None:  # after the motors, never before: the turn it stopped, for a --resume to know
            record("stopped by Ctrl-C, which cut every motor" if bench.hand is not None else "stopped by Ctrl-C")
        return 130
    finally:
        keyboard.close()
        pool = getattr(router, "pool", None)
        if pool is not None:  # tool calls not started yet don't start
            pool.shutdown(wait=False, cancel_futures=True)
        bench.park(home=home)


def cli() -> None:
    """python -m wbc.agent and the wbc script: main(), then out at once. The arm is parked by then, so a tool call
    still running after Ctrl-C (a coder call, a locate) is dropped rather than waited for, and what it costs isn't in
    .wbc/spend.json: the spend is added when a reply comes back."""
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    cli()
