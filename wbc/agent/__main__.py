"""Run the bench agent: python -m wbc.agent [--sim] [--voice] [--hear] [--speak] [--arm] [--enter]
[--no-prefetch] [--hud] [--show]

Typed lines and spoken requests go on one queue, and the router handles them one turn at a time. Before each turn,
code fetches the bench facts in models.yaml's router.prefetch: pins.h and the sensor, and Act 1's board read, which is
made once at start-up and kept until the router reads the board itself; --no-prefetch skips them, for a session that
only moves the arm. With --voice (or
--speak), the co-pilot says the first sentence or two of each reply out loud, and the mic is muted while it speaks.
With --arm, the arm's tools may move it. With robot.yaml's hand.go_ahead "request", the request is the go-ahead and
the arm moves at once. With "enter", or --enter for bench tests, each move waits for Enter at the keyboard: while a
move waits, the next typed line answers it instead of going to the router, and nothing heard by voice does.
"Copilot, stop", or "Stop!" on its own, is heard even while the co-pilot speaks: a move stops at its next tick and
the arm holds where it is, its motors on, requests waiting behind it are dropped, and the next request may move it
again. Ctrl-C cuts every motor (the arm drops, so keep a hand near it) and quits at once, and no arm move starts after
it. At the end of the input (Ctrl-D) a question still waiting gets a no, no arm move starts, and once the turn ends
the arm goes back to where it rested, unless it was told to stop, and its motors go off; any other exit asks before
moving it.

Every turn is recorded in .wbc/session.json (wbc/agent/memory.py), however it ends. Each run starts a new session, and
--resume picks up the last one instead; the notebook the router keeps with `remember` carries over either way.

With --hud, the HUD's orb page shows each turn live at http://127.0.0.1:8766/ (wbc/agent/web_hud.py): the mic
listening, the request, every model call with its time, the reply and its speech, and the turn's time and spend.
With --show, for a take, it opens in a window of its own, and the board's schematic opens in KiCad (wbc/agent/show.py).
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path

from wbc import config
from wbc.agent import coder, ears
from wbc.agent.approve import Keyboard
from wbc.agent.budget import Budget, BudgetExceeded
from wbc.agent.fast_arm import FastArm
from wbc.agent.hud import ConsoleHud
from wbc.agent.loop import Router
from wbc.agent.memory import Memory
from wbc.agent.mouth import Mouth
from wbc.agent.tools import Bench, add_memory_tools, build_toolbox
from wbc.agent.web_hud import WebHud

SYSTEM = (Path(__file__).parent / "prompts" / "system.md").read_text(encoding="utf-8")
FAILED = "That turn failed. The error is on the screen."
GO_AHEADS = ("request", "enter")  # robot.yaml's hand.go_ahead: what lets an arm move go


def build(sim: bool, router_model: str | None = None, arm: bool = False,
          calibration_dir: str | None = None, prefetch: bool = True, resume: bool = False,
          hud_port: int | None = None, enter: bool = False, fast: bool = True):
    models = config.load_yaml("models.yaml")
    budget = Budget.from_config(models, config.ROOT)
    hud = ConsoleHud() if hud_port is None else WebHud(models, budget, hud_port)
    meta = hud.watch(config.meta_client(models))  # the coder always runs on the Meta Model API
    robot_cfg = config.load_yaml("robot.yaml")
    go_ahead = "enter" if enter else (robot_cfg.get("hand") or {}).get("go_ahead", "enter")
    if go_ahead not in GO_AHEADS:
        raise RuntimeError(f"robot.yaml's hand.go_ahead is {' or '.join(GO_AHEADS)}, not {go_ahead!r}")
    keyboard = Keyboard(say=hud.note)
    bench = Bench(config.ROOT, models, meta, budget, sim=sim, say=hud.note, layout=config.load_yaml("bench.yaml"),
                  parts=config.load_yaml("parts.yaml"), robot_cfg=robot_cfg, desk_cfg=config.load_yaml("desk.yaml"),
                  arm_port=robot_cfg.get("port") if arm else None,
                  move_logs=config.ROOT / ".wbc" / "logs" if arm and not sim else None,
                  calibration_dir=calibration_dir or robot_cfg.get("calibration_dir"), confirm=keyboard.ask,
                  go_ahead=go_ahead)
    if arm:
        hud.note("the arm moves when you ask; \"Copilot, stop\" holds it, Ctrl-C cuts it" if go_ahead == "request" else
                 "each arm move waits for Enter at the keyboard; Ctrl-C cuts it")
    if arm and not sim:  # the desk camera kept open, so a locate before a move needn't open it and wait 3 s
        from wbc.perception import camera  # the eyes extra

        cam = bench.desk_cfg["camera"]
        try:
            bench.desk_stream = camera.Stream(cam["index"], cam["width"], cam["height"])
            hud.note(f"desk camera {cam['index']} kept open for locate")
        except RuntimeError as e:
            hud.note(f"desk camera: {e} Each locate opens it instead.")
    toolbox = build_toolbox(bench)
    max_steps = models["router"].get("max_steps", 12)
    prefetch = models["router"].get("prefetch") if prefetch else None
    memory = Memory.from_config(models, config.ROOT, resume=resume, sim=sim)
    add_memory_tools(toolbox, memory.notebook)
    if fast and models.get("fast_arm"):
        fast = FastArm(models["fast_arm"], toolbox, bench, memory.notebook, hud)
    else:
        fast = None
    o = models.get("open")
    reaches = config.open_client(models) if o and not sim else None
    open_route = (hud.watch(reaches), o["model"], o.get("label", "Spark Open")) if reaches else None
    bench.open = open_route  # add_part asks Spark Open too
    if o and not sim:
        hud.note(f"{o.get('label', 'Spark Open')}: " + ("questions no tool is needed for" if reaches else
                                                      f"off, no {o.get('key_env', 'BASETEN_API_KEY')}"))
    r = models["router"]
    router = Router(meta, router_model or r["model"], toolbox, budget, hud, SYSTEM, max_steps=max_steps,
                    prefetch=prefetch, memory=memory, fast=fast, fallback=r.get("fallback"), effort=r.get("effort"),
                    open=open_route)
    return models, budget, hud, router, bench, keyboard


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wbc", description=__doc__.splitlines()[0])
    ap.add_argument("--sim", action="store_true", help="use the recorded bench instead of cameras and arm")
    ap.add_argument("--voice", action="store_true", help='listen for "Copilot, ..." as well as typed input')
    ap.add_argument("--hear", action="store_true",
                    help="mic check: print what the mic hears and route nothing (implies --voice)")
    ap.add_argument("--speak", action=argparse.BooleanOptionalAction,
                    help="say each reply's first sentence or two with macOS say (on with --voice)")
    ap.add_argument("--router-model", metavar="ID",
                    help="route with this model instead of router.model, e.g. muse-spark-1.1 to compare speed")
    ap.add_argument("--arm", action="store_true",
                    help="let the arm's tools move it, as robot.yaml's hand.go_ahead says (the port is robot.yaml's)")
    ap.add_argument("--enter", action="store_true",
                    help="each arm move waits for Enter at the keyboard, whatever hand.go_ahead says (bench tests)")
    ap.add_argument("--no-fast-arm", action="store_true",
                    help="send arm commands to the router's model too, not the local fast path (to compare times)")
    ap.add_argument("--calibration-dir", metavar="DIR",
                    help="lerobot-calibrate's folder for the arm, when it's another macOS account's")
    ap.add_argument("--no-prefetch", action="store_true",
                    help="skip the bench facts fetched before each turn (Act 1's board read, pins.h and the sensor)")
    ap.add_argument("--resume", action="store_true",
                    help="pick up the last session (.wbc/session.json) instead of starting a new one")
    ap.add_argument("--hud", nargs="?", const=8766, type=int, metavar="PORT",
                    help="show each turn live on the HUD's orb page, at http://127.0.0.1:8766/ (or PORT)")
    ap.add_argument("--show", action="store_true",
                    help="for a take: the HUD (with --hud) in a window of its own, and the schematic in KiCad")
    args = ap.parse_args(argv)

    try:
        models, budget, hud, router, bench, keyboard = build(args.sim, args.router_model, args.arm,
                                                             args.calibration_dir, not args.no_prefetch, args.resume,
                                                             args.hud, args.enter, not args.no_fast_arm)
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
    question = ears.Question()  # while the last reply waits for an answer, a short one needs no "Copilot,"

    def stop() -> None:
        """The spoken stop, on the ears' thread, whatever the turn is doing: the arm holds where it is, requests waiting
        behind it are dropped, and the co-pilot stops talking."""
        bench.halt()
        dropped = []
        while True:
            try:
                item = inbox.get_nowait()
            except queue.Empty:
                break
            if item is None:  # the end of the input stays the last thing on the queue
                inbox.put(None)
                break
            dropped.append(item)
        if mouth:
            mouth.speak("Stopped.")  # cuts off whatever it was saying
        hud.note("stopped: no arm move starts until the next request"
                 + (f"; dropped what was waiting: {' / '.join(dropped)}" if dropped else ""))

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
    read_ahead = getattr(router, "read_ahead", None)
    if read_ahead is not None and not args.hear:  # the board read, made now and kept, so no turn waits for a photo
        threading.Thread(target=read_ahead, daemon=True).start()
    fast = getattr(router, "fast", None)
    if fast is not None and not args.hear:  # it loads its model on the first request, so that goes now
        threading.Thread(target=fast.start, daemon=True).start()
    if not args.hear:  # build the firmware once now, so the session's first patch skips the cold build
        threading.Thread(target=coder.compile_in_sandbox, args=("", config.ROOT, models["coder"]), daemon=True).start()
    if args.voice or args.hear:
        on_text = (lambda text: None) if args.hear else inbox.put  # listen_forever prints what it heard
        threading.Thread(target=ears.listen_forever, daemon=True,
                         args=(on_text, models["ears"], config.api_key(), budget, hud),
                         kwargs={"muted": mouth.busy if mouth else None, "on_stop": None if args.hear else stop,
                                 "question": None if args.hear else question}).start()
    if args.hear:
        hud.note("mic check: nothing goes to the router. Ctrl-D quits.")
    window = None
    if args.show:
        from wbc.agent import show
        if args.hud:
            window = show.hud_window(f"http://127.0.0.1:{args.hud}/", config.ROOT / ".wbc" / "hud-window")
            hud.note("HUD: in a window of its own" if window else "HUD: no Chrome for its window, so open the page")
        drawn = (bench.layout or {}).get("schematic")
        if drawn and not args.sim:
            hud.note(show.schematic(config.ROOT / drawn))
            bench.show = show.schematic  # add_part's copy goes in the same window

    home = True  # at the end of the input (Ctrl-D) the arm goes back to rest before its motors go off, as --help says
    try:
        while True:
            text = inbox.get()
            if text is None:
                return 0
            if args.hear:
                continue
            bench.resume()  # a new request: after a stop, the arm may move again
            question.close()
            hud.turn(text)
            try:
                reply = router.run_turn(text)
                if mouth:
                    mouth.speak(reply)
                question.asked(reply, time.monotonic())
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
        stream = getattr(bench, "desk_stream", None)
        if stream is not None:
            stream.close()
        if window is not None:
            window.terminate()


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
