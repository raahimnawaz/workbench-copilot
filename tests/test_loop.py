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
                        function_call("c2", "point_at", '{"target": "row:41"}')]),
        response("r2", text="CS is on GPIO 13. Patch staged."),
    ], slow_toolbox(log))

    t0 = time.monotonic()
    assert router.run_turn("check my wiring") == "CS is on GPIO 13. Patch staged."
    assert time.monotonic() - t0 < 0.55, "tool calls should run in parallel"
    assert sorted(log) == ["coder", "row:41"]

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


def test_next_turn_chains_on_the_last_response():
    router, api, _ = make_router([response("r1", text="hi"), response("r2", text="again")], Toolbox())
    router.run_turn("one")
    router.run_turn("two")
    assert api.requests[1]["previous_response_id"] == "r1"


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
    assert scene["ok"] and scene["mismatches"][0]["signal"] == "OLED_CS"
    assert tb.dispatch("point_at", {"target": "row:41"})["sim"] is True
    assert "call ask_coder first" in tb.dispatch("propose_patch", {"patch_id": "p9"})["error"]
    assert set(tb.names()) >= {"read_file", "ask_coder", "propose_patch", "read_breadboard", "look", "point_at"}


def test_read_file_refuses_paths_outside_the_repo():
    tb = build_toolbox(Bench(ROOT, load_yaml("models.yaml"), None, Budget(10, {})))
    assert tb.dispatch("read_file", {"path": "../../etc/passwd"})["ok"] is False
    assert "OLED_CS" in tb.dispatch("read_file", {"path": "firmware/dut_demo/include/pins.h"})["text"]


def test_schemas_in_both_formats():
    tb = build_toolbox(Bench(ROOT, load_yaml("models.yaml"), None, Budget(10, {})))
    assert all(s["type"] == "function" and "name" in s for s in tb.schemas())
    assert all("function" in s and "name" in s["function"] for s in tb.chat_schemas())


@pytest.mark.parametrize("name", ["router", "coder", "ears", "eyes"])
def test_models_yaml_prices_every_hosted_model(name):
    models = load_yaml("models.yaml")
    assert models[name]["model"] in models["budget"]["prices"]
