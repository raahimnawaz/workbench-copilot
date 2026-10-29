import json
import time

import pytest
from fakes import (FakeChatClient, FakeClient, SilentHud, chat_response, function_call, response)

from wbc.agent.budget import Budget
from wbc.agent.loop import ChatRouter, Router
from wbc.agent.tools import Bench, Toolbox, build_toolbox
from wbc.config import ROOT, load_yaml

PRICES = {"spark": {"in": 1.25, "out": 4.25}, "spark-old": {"in": 1.25, "out": 4.25},
          "glimmer": {"in": 0, "out": 0}}
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def slow_toolbox(log):
    tb = Toolbox()

    @tb.register("ask_coder", "", {"type": "object", "properties": {}}, label="Wonder")
    def ask_coder():
        time.sleep(0.3)
        log.append("coder")
        return {"ok": True, "patch_id": "p1"}

    @tb.register("point_at", "", {"type": "object", "properties": {"target": {"type": "string"}}}, label="arm")
    def point_at(target):
        time.sleep(0.3)
        log.append(target)
        return {"ok": True, "_image": PNG}

    return tb


def make_router(scripted, toolbox, **kw):
    client = FakeClient(scripted)
    hud = SilentHud()
    router = Router(client, "spark", toolbox, Budget(10, PRICES), hud, "system", **kw)
    return router, client.responses, hud


def test_parallel_tool_round_then_reply():
    log = []
    router, api, hud = make_router([
        response("r1", [function_call("c1", "ask_coder", "{}"),
                        function_call("c2", "point_at", '{"target": "row:5"}')]),
        response("r2", text="The LED is on GPIO 4. Patch staged."),
    ], slow_toolbox(log))

    t0 = time.monotonic()
    assert router.run_turn("check my wiring") == "The LED is on GPIO 4. Patch staged."
    assert time.monotonic() - t0 < 0.55, "tool calls should run in parallel"
    assert sorted(log) == ["coder", "row:5"]

    second = api.requests[1]
    assert second["previous_response_id"] == "r1"
    outputs = [i for i in second["input"] if i.get("type") == "function_call_output"]
    assert [o["call_id"] for o in outputs] == ["c1", "c2"]
    assert "_image" not in outputs[1]["output"]
    # the image goes after every function output, as a user message
    assert second["input"][-1]["content"][1]["image_url"].startswith("data:image/png;base64,")
    assert router.prev_id == "r2"
    assert ("call", "Wonder", "ask_coder") in hud.events
    assert router.budget.spent > 0


def test_reasoning_effort_is_sent_only_when_set():
    router, api, hud = make_router([response("r1", text="a"), response("r2", text="b")], Toolbox())
    router.run_turn("one")
    assert "reasoning" not in api.requests[0]
    router.effort = "low"
    router.run_turn("two")
    assert api.requests[1]["reasoning"] == {"effort": "low"}
    assert sum(1 for e in hud.events if e[0] == "note" and "answered in" in e[1]) == 2


def test_next_turn_chains_on_the_last_response():
    router, api, _ = make_router([response("r1", text="hi"), response("r2", text="again")], Toolbox())
    router.run_turn("one")
    router.run_turn("two")
    assert api.requests[1]["previous_response_id"] == "r1"


def test_router_stops_before_a_call_once_the_budget_is_spent():
    from wbc.agent.budget import BudgetExceeded

    router, api, _ = make_router([response("r1", text="hi")], Toolbox())
    router.budget.spent = router.budget.limit
    with pytest.raises(BudgetExceeded):
        router.run_turn("hello")
    assert api.requests == [], "no paid call may go out past the stop"


def test_router_shows_the_spend_warning_on_the_hud():
    router, _, hud = make_router([response("r1", text="hi")], Toolbox())
    router.budget.warn = 0.0001
    router.run_turn("hello")
    assert any(e[0] == "note" and "Spend has passed" in e[1] for e in hud.events)


def test_bad_json_arguments_come_back_as_a_tool_error():
    router, api, _ = make_router([
        response("r1", [function_call("c1", "point_at", "{not json")]),
        response("r2", text="sorry"),
    ], slow_toolbox([]))
    router.run_turn("point")
    out = json.loads(api.requests[1]["input"][0]["output"])
    assert out["ok"] is False and "not valid JSON" in out["error"]


def test_falls_back_when_the_model_is_not_found():
    class NotFoundError(Exception):
        pass

    router, api, hud = make_router([NotFoundError("no such model"), response("r1", text="ok")],
                                   Toolbox(), fallback="spark-old")
    router.run_turn("hello")
    assert api.requests[1]["model"] == "spark-old"
    assert router.model == "spark-old"


def test_stops_after_max_steps_and_does_not_chain_onto_pending_calls():
    looping = [response(f"r{i}", [function_call(f"c{i}", "ask_coder", "{}")]) for i in range(4)]
    router, _, hud = make_router(looping, slow_toolbox([]), max_steps=3)
    router.run_turn("loop forever")
    assert router.prev_id is None
    assert any(e[0] == "note" and "stopped" in e[1] for e in hud.events)


PREFETCH = [{"tool": "read_breadboard"}, {"tool": "read_file", "args": {"path": "pins.h"}}]


def facts_toolbox(log, board=None):
    tb = Toolbox()

    @tb.register("read_breadboard", "", {"type": "object", "properties": {}}, label="SAM")
    def read_breadboard():
        log.append("board")
        return dict(board or {"ok": True, "mismatches": [{"signal": "LED_PIN"}]})

    @tb.register("read_file", "", {"type": "object", "properties": {"path": {"type": "string"}}})
    def read_file(path):
        log.append(path)
        return {"ok": True, "path": path, "text": "#define LED_PIN 5"}

    return tb


def test_bench_facts_go_out_with_the_first_call():
    """Act 1 on Oct 28 spent its first router round (4.4 s) only asking for the board read and pins.h."""
    log = []
    router, api, hud = make_router([response("r1", text="The LED is on 4.")], facts_toolbox(log), prefetch=PREFETCH)
    router.run_turn("check my wiring")
    assert sorted(log) == ["board", "pins.h"] and len(api.requests) == 1
    first = api.requests[0]["input"][0]["content"]
    assert first.startswith("check my wiring") and "LED_PIN" in first and "#define LED_PIN 5" in first
    assert ("call", "SAM", "read_breadboard") in hud.events


def test_facts_the_conversation_has_are_not_sent_again():
    board = {"ok": True, "mismatches": []}
    router, api, _ = make_router([response(f"r{i}", text="ok") for i in range(3)], facts_toolbox([], board),
                                 prefetch=PREFETCH)
    router.run_turn("one")
    router.run_turn("two")
    second = api.requests[1]["input"][0]["content"]
    assert second.count("unchanged since you last saw it") == 2 and "mismatches" not in second
    board["mismatches"] = [{"signal": "LED_PIN"}]
    router.run_turn("three")
    third = api.requests[2]["input"][0]["content"]
    assert "LED_PIN" in third and third.count("unchanged") == 1  # only pins.h stayed the same


def test_a_broken_chain_gets_every_fact_again():
    looping = [response(f"r{i}", [function_call(f"c{i}", "read_breadboard", "{}")]) for i in range(2)]
    router, api, _ = make_router(looping + [response("r9", text="ok")], facts_toolbox([]), prefetch=PREFETCH,
                                 max_steps=1)
    router.run_turn("loop")  # stops after one round, so the next turn starts a new conversation
    router.run_turn("again")
    assert "unchanged" not in api.requests[-1]["input"][0]["content"]


def test_a_prefetched_image_rides_with_the_first_call():
    router, api, _ = make_router([response("r1", text="ok")], facts_toolbox([], {"ok": True, "_image": PNG}),
                                 prefetch=PREFETCH[:1])
    router.run_turn("look")
    content = api.requests[0]["input"][0]["content"]
    assert content[0]["type"] == "input_text" and "_image" not in content[0]["text"]
    assert content[-1]["image_url"].startswith("data:image/png;base64,")


def test_private_keys_stay_out_of_what_the_model_sees():
    tb = Toolbox()

    @tb.register("ask_coder", "", {"type": "object", "properties": {}}, label="Wonder")
    def ask_coder():
        return {"ok": True, "_timing": {"wonder": 7.0, "compile": 14.0}}

    router, api, _ = make_router([response("r1", [function_call("c1", "ask_coder", "{}")]),
                                  response("r2", text="done")], tb)
    router.run_turn("fix it")
    assert "_timing" not in api.requests[1]["input"][0]["output"]


def test_chat_router_gets_the_same_bench_facts():
    router = ChatRouter(FakeChatClient([chat_response(content="ok")]), "glimmer", facts_toolbox([]),
                        Budget(10, PRICES), SilentHud(), "system", prefetch=PREFETCH)
    router.run_turn("check my wiring")
    assert "#define LED_PIN 5" in router.messages[1]["content"]


def test_chat_router_message_order():
    client = FakeChatClient([
        chat_response(tool_calls=[("t1", "point_at", '{"target": "c41"}')]),
        chat_response(content="Pointing at c41."),
    ])
    router = ChatRouter(client, "glimmer", slow_toolbox([]), Budget(10, PRICES), SilentHud(), "system",
                        reasoning="medium")
    assert router.run_turn("point at c41") == "Pointing at c41."
    roles = [m["role"] for m in router.messages]
    assert roles == ["system", "user", "assistant", "tool", "user", "assistant"]
    assert router.messages[0]["content"].endswith("Reasoning strength: medium")
    assert router.messages[2]["tool_calls"][0]["function"]["name"] == "point_at"


def test_toolbox_turns_exceptions_into_results():
    tb = Toolbox()

    @tb.register("boom", "", {"type": "object", "properties": {}})
    def boom():
        raise RuntimeError("servo bus timeout")

    assert tb.dispatch("missing", {})["ok"] is False
    assert "bad arguments" in tb.dispatch("boom", {"x": 1})["error"]
    assert tb.dispatch("boom", {}) == {"ok": False, "error": "RuntimeError: servo bus timeout"}


def test_sim_bench_tools():
    models = load_yaml("models.yaml")
    tb = build_toolbox(Bench(ROOT, models, client=None, budget=Budget(10, {}), sim=True))
    scene = tb.dispatch("read_breadboard", {})
    assert scene["ok"] and scene["mismatches"][0]["signal"] == "LED_PIN"
    assert tb.dispatch("point_at", {"target": "row:5"})["sim"] is True
    assert "call ask_coder first" in tb.dispatch("propose_patch", {"patch_id": "p9"})["error"]
    assert set(tb.names()) >= {"read_file", "ask_coder", "propose_patch", "read_breadboard", "look", "point_at"}


def test_read_file_refuses_paths_outside_the_repo():
    tb = build_toolbox(Bench(ROOT, load_yaml("models.yaml"), None, Budget(10, {})))
    assert tb.dispatch("read_file", {"path": "../../etc/passwd"})["ok"] is False
    assert "LED_PIN" in tb.dispatch("read_file", {"path": "firmware/dut_demo/include/pins.h"})["text"]


def test_schemas_in_both_formats():
    tb = build_toolbox(Bench(ROOT, load_yaml("models.yaml"), None, Budget(10, {})))
    assert all(s["type"] == "function" and "name" in s for s in tb.schemas())
    assert all("function" in s and "name" in s["function"] for s in tb.chat_schemas())


@pytest.mark.parametrize("name", ["router", "coder", "ears", "eyes"])
def test_models_yaml_prices_every_hosted_model(name):
    models = load_yaml("models.yaml")
    assert models[name]["model"] in models["budget"]["prices"]
