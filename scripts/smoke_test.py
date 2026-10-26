"""Day 1 smoke test: one small call to each hosted model, so you know the ids, key and formats work.

Costs a few cents at most. Run from the repo root: python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wbc import config  # noqa: E402
from wbc.agent.budget import Budget  # noqa: E402
from wbc.agent.ears import Transcriber, pcm_to_wav  # noqa: E402


def step(name: str, fn) -> bool:
    t0 = time.monotonic()
    try:
        detail = fn()
    except Exception as e:
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        return False
    print(f"PASS  {name} ({time.monotonic() - t0:.1f} s) {detail or ''}")
    return True


def main() -> int:
    models = config.load_yaml("models.yaml")
    client = config.meta_client(models)
    budget = Budget.from_config(models, config.ROOT)
    router, coder = models["router"]["model"], models["coder"]["model"]

    def model_list():
        ids = {m.id for m in client.models.list()}
        wanted = [models[k]["model"] for k in ("router", "coder", "eyes", "ears")] + [models["router"]["fallback"]]
        missing = [w for w in wanted if w not in ids]
        return f"{len(ids)} models listed; missing: {', '.join(missing) or 'none'}"

    def router_tool_call():
        tools = [{"type": "function", "name": "get_bench_time", "description": "Return the bench clock.",
                  "parameters": {"type": "object", "properties": {}}}]
        r = client.responses.create(model=router, tools=tools,
                                    input="What time is it on the bench? Use the tool.")
        budget.add_tokens(router, r.usage)
        calls = [i for i in r.output if getattr(i, "type", "") == "function_call"]
        if not calls:
            raise RuntimeError(f"no function_call in the output: {r.output_text[:200]!r}")
        return f"called {calls[0].name}"

    def coder_diff():
        r = client.responses.create(model=coder, reasoning={"effort": "low"},
                                    input="Write a unified diff for x.py that changes the line `a = 1` to `a = 2`, "
                                          "then one line starting with WHY:.")
        budget.add_tokens(coder, r.usage)
        if "+a = 2" not in r.output_text:
            raise RuntimeError(f"unexpected reply: {r.output_text[:200]!r}")
        return "diff looks right"

    def transcribe():
        wav = pcm_to_wav(b"\x00\x00" * 16000, 16000)
        text = Transcriber(models["ears"], config.api_key(), budget).transcribe(wav, 1.0)
        return f"1 s of silence gave {text!r}"

    results = [
        step("model list", model_list),
        step("router tool call", router_tool_call),
        step("coder diff", coder_diff),
        step("voice transcribe", transcribe),
    ]
    print("SKIP  SAM 3.1: add once the request format is confirmed from Meta's SAM docs")
    print(f"spent so far: ${budget.spent:.4f}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
