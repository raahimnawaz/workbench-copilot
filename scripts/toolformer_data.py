"""Training data for the arm's fast path, after Meta's Toolformer: tool-call labels a model writes, kept only when the
call works.

Toolformer teaches a model to use tools from labels it writes itself, keeping only the calls that help. Here the
student is the fast path's Llama 3.2 3B (wbc/agent/fast_arm.py), and the steps are:

1. Spark writes varied spoken requests for each intent below: arm commands on the recorded desk's screwdriver, tray
   and cup, and the things that only sound like them (questions, "I put...", "don't...", idioms, wiring requests).
2. Spark labels each one exactly as the fast path asks Llama: fast_arm.SYSTEM, and the bench state (fast_arm.asked).
3. A label is kept only when its kind matches the intent the request was written for, and, for an arm tool, the call
   runs on the recorded bench (--sim): a put_down labeled while nothing is held fails there and is dropped.
4. Anything the benchmark tests (--exclude, its phrasings one per line) is left out, so the evaluation stays unseen.

Out: .wbc/toolformer/train.jsonl and valid.jsonl, chat messages as the fast path sends them, with the label as the
answer; and labels.jsonl, every request with its label and why it was kept or dropped.

    .venv/bin/python scripts/toolformer_data.py --per 30          # PAID: about $0.65 for 13 intents
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wbc import config  # noqa: E402
from wbc.agent import fast_arm  # noqa: E402
from wbc.agent.budget import Budget  # noqa: E402
from wbc.agent.tools import Bench, build_toolbox  # noqa: E402
from wbc.config import ROOT, load_yaml  # noqa: E402

OUT = ROOT / ".wbc" / "toolformer"
# Each intent: what the requests should mean, what the arm holds while they're said, and the tool a right label
# names (the fast path's kinds decide what counts as the same).
INTENTS = [
    ("move the screwdriver off the desk into or onto the cup or the tray, naming both", None, "move_thing"),
    ("pick up the screwdriver and keep holding it, naming it", None, "pick_up"),
    ("put down what the arm holds (the screwdriver) on the tray or in the cup, naming the place", "screwdriver",
     "put_down"),
    ("send the arm back to its rest or home position", None, "arm_home"),
    ("ask the arm to hold something the user is handing it, such as a probe or a wire", None, "hold"),
    ("tell the arm to let go of what the user handed it", None, "let_go"),
    ("ask a question about the arm or the desk without asking it to move (where is it, is it holding, can it reach)",
     None, "not_mine"),
    ("say what you yourself did or will do with the screwdriver, cup or tray (I put..., I'm going to...)", None,
     "not_mine"),
    ("tell the arm not to move yet, to wait, or not to touch the screwdriver", None, "not_mine"),
    ("use pick, put, take, hold, grab or move in another sense, not about the arm (pick the faster option, put that in "
     "the notes, hold on what was that)", None, "not_mine"),
    ("ask about wiring, firmware, flashing the board, the schematic, a reading or a voltage", None, "not_mine"),
    ("answer the co-pilot's question or thank it (yes do it, go ahead, thanks, sounds good)", None, "not_mine"),
    ("ask the co-pilot to add a part, check the board or explain something", None, "not_mine"),
]
WRITE = ("Write {n} different things a person at an electronics bench might say out loud to their voice co-pilot, "
         "which runs a robot arm beside them, that {intent}. Vary the wording, length, politeness and word order; "
         "about a third start with \"Copilot,\"; include a few with fillers (um, okay) and a few with the slips speech "
         "recognition makes (screw driver, the cup's). One per line, no numbering, no quotes, nothing else.")


def spark_text(client, models: dict, budget, instructions: str, text: str) -> str:
    budget.check()
    r = client.responses.create(model=models["router"]["model"], instructions=instructions, input=text,
                                reasoning={"effort": models["router"].get("effort", "low")})
    budget.add_tokens(models["router"]["model"], getattr(r, "usage", None))
    return getattr(r, "output_text", "") or ""


def lines(text: str) -> list:
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"').strip()
        if 3 <= len(line) <= 160:
            out.append(line)
    return out


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def kind(tool: str | None) -> str | None:
    return fast_arm.KINDS.get(tool) if tool != "not_mine" else "not_mine"


def runs(tb, bench, label: dict, holding: str | None) -> tuple:
    """Whether the label's call works on the recorded bench, and what it said."""
    tool = label.get("tool")
    if tool not in tb.names():  # hold and let_go aren't tools yet, and not_mine is no call: the kind check stands
        return True, "no call to run"
    bench.sim_held, bench.seen = holding, {}
    call = {"tool": tool, "thing": label.get("thing"), "place": label.get("place")}
    result = tb.dispatch(tool, fast_arm.arguments(call))
    return bool(result.get("ok")), result.get("error", "ran")


def example(text: str, holding: str | None, label: dict) -> dict:
    answer = json.dumps({"tool": label["tool"], "thing": label.get("thing"), "place": label.get("place")})
    return {"messages": [{"role": "system", "content": fast_arm.SYSTEM},
                         {"role": "user", "content": fast_arm.asked(text, holding)},
                         {"role": "assistant", "content": answer}]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per", type=int, default=30, help="requests Spark writes per intent")
    ap.add_argument("--exclude", type=Path, help="phrasings to leave out, one per line (the benchmark's)")
    ap.add_argument("--valid", type=float, default=0.15, help="share kept back to check training")
    args = ap.parse_args()

    models = load_yaml("models.yaml")
    client, budget = config.meta_client(models), Budget.from_config(models, ROOT)
    bench = Bench(ROOT, models, None, budget, sim=True, say=lambda s: None, robot_cfg=load_yaml("robot.yaml"))
    tb = build_toolbox(bench)
    excluded = {norm(t) for t in args.exclude.read_text().splitlines()} if args.exclude else set()
    OUT.mkdir(parents=True, exist_ok=True)
    kept, record, seen = [], [], set()
    t0, spent0 = time.monotonic(), budget.spent
    for intent, holding, tool in INTENTS:
        written = lines(spark_text(client, models, budget, "Write exactly what's asked.",
                                   WRITE.format(n=args.per, intent=intent)))
        fresh = []
        for text in written:
            if norm(text) in seen or norm(text) in excluded:
                record.append({"text": text, "intent": tool, "kept": False, "why": "repeat or in the benchmark"})
                continue
            seen.add(norm(text))
            fresh.append(text)
        with ThreadPoolExecutor(max_workers=4) as pool:  # the labels in parallel; the bench runs them one at a time
            labels = list(pool.map(lambda text: fast_arm.parse(spark_text(
                client, models, budget, fast_arm.SYSTEM, fast_arm.asked(text, holding))), fresh))
        for text, label in zip(fresh, labels):
            if kind(label.get("tool")) != kind(tool):
                why, ok = f"labeled {label.get('tool')}, written as {tool}", False
            else:
                ok, why = runs(tb, bench, label, holding)
            record.append({"text": text, "holding": holding, "intent": tool, "label": label, "kept": ok, "why": why})
            if ok:
                kept.append(example(text, holding, label))
        print(f"{tool:10} {sum(r['kept'] for r in record if r['intent'] == tool):3} kept of {len(written)}  "
              f"${budget.spent - spent0:.3f} so far", flush=True)
    random.Random(0).shuffle(kept)
    n_valid = max(1, round(len(kept) * args.valid))
    for name, rows in (("valid", kept[:n_valid]), ("train", kept[n_valid:])):
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (OUT / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in record))
    print(f"{len(kept)} kept of {len(record)}: {len(kept) - n_valid} to train on, {n_valid} to check; "
          f"${budget.spent - spent0:.2f}, {time.monotonic() - t0:.0f} s. In {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
