import json
import time

import pytest
from fakes import FakeChatClient, FakeClient, SilentHud, asked, chat_response, function_call, response

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
        response("r2", text="The sensor is on GPIO 4. Patch staged."),
    ], slow_toolbox(log))

    t0 = time.monotonic()
    assert router.run_turn("check my wiring") == "The sensor is on GPIO 4. Patch staged."
    assert time.monotonic() - t0 < 0.55, "tool calls should run in parallel"
    assert sorted(log) == ["coder", "row:5"]

    second = api.requests[1]
    assert "previous_response_id" not in second
    assert [i.get("type") for i in second["input"][1:3]] == ["function_call", "function_call"]  # r1's, sent back
    outputs = [i for i in second["input"] if i.get("type") == "function_call_output"]
    assert [o["call_id"] for o in outputs] == ["c1", "c2"]
    assert "_image" not in outputs[1]["output"]
    # the image goes after every function output, as a user message
    assert second["input"][-1]["content"][1]["image_url"].startswith("data:image/png;base64,")
    assert router.history[-1] == {"role": "assistant", "content": "The sensor is on GPIO 4. Patch staged."}
    assert ("call", "Wonder", "ask_coder") in hud.events
    assert router.budget.spent > 0


def test_only_the_first_arm_call_of_a_round_runs_in_sim_or_on_the_arm():
    log = []
    tb = slow_toolbox(log)

    @tb.register("pick_up", "", {"type": "object", "properties": {}}, label="arm")
    def pick_up():
        log.append("pick")
        return {"ok": True}

    router, api, hud = make_router([
        response("r1", [function_call("c1", "pick_up", "{}"), function_call("c2", "point_at", '{"target": "row:5"}'),
                        function_call("c3", "ask_coder", "{}")]),
        response("r2", text="Picked it up."),
    ], tb)
    assert router.run_turn("pick it up and fix the pin") == "Picked it up."
    assert sorted(log) == ["coder", "pick"]  # point_at, a second arm tool, never ran
    outputs = {o["call_id"]: json.loads(o["output"]) for o in api.requests[1]["input"]
               if o.get("type") == "function_call_output"}
    assert outputs["c2"] == {"ok": False, "error": "not run yet: pick_up came first this round, and arm tools run one "
                                                   "at a time. Call point_at again once pick_up has returned, if it's "
                                                   "still wanted"}


def test_an_arm_call_after_one_whose_arguments_dont_parse_still_waits():
    log = []
    tb = slow_toolbox(log)

    @tb.register("pick_up", "", {"type": "object", "properties": {"thing": {"type": "string"}}}, label="arm")
    def pick_up(thing):
        log.append("pick")
        return {"ok": True}

    @tb.register("put_down", "", {"type": "object", "properties": {"thing": {"type": "string"}}}, label="arm")
    def put_down(thing):
        log.append("put")
        return {"ok": True}

    router, api, hud = make_router([
        response("r1", [function_call("c1", "pick_up", '{"thing": "scr'),
                        function_call("c2", "put_down", '{"thing": "tray"}')]),
        response("r2", text="Try again?"),
    ], tb)
    assert router.run_turn("put the screwdriver in the tray") == "Try again?"
    assert log == []  # the put needs the pick to have happened, so it doesn't ask to go ahead without it
    outputs = {o["call_id"]: json.loads(o["output"]) for o in api.requests[1]["input"]
               if o.get("type") == "function_call_output"}
    assert outputs["c1"]["error"].startswith("arguments were not valid JSON")
    assert outputs["c2"]["error"].startswith("not run yet: pick_up came first this round")


def test_reasoning_effort_is_sent_only_when_set():
    router, api, hud = make_router([response("r1", text="a"), response("r2", text="b")], Toolbox())
    router.run_turn("one")
    assert "reasoning" not in api.requests[0]
    router.effort = "low"
    router.run_turn("two")
    assert api.requests[1]["reasoning"] == {"effort": "low"}
    assert sum(1 for e in hud.events if e[0] == "note" and "answered in" in e[1]) == 2


def test_next_turn_sends_the_conversation_so_far():
    router, api, _ = make_router([response("r1", text="hi"), response("r2", text="again")], Toolbox())
    router.run_turn("one")
    router.run_turn("two")
    assert api.requests[1]["input"] == [{"role": "user", "content": "one"}, {"role": "assistant", "content": "hi"},
                                        {"role": "user", "content": "two"}]
    assert "previous_response_id" not in api.requests[1]


def test_a_reply_and_its_calls_go_back_into_the_conversation_without_its_reasoning():
    from types import SimpleNamespace as NS
    first = response("r1", [NS(type="reasoning", summary=[]), function_call("c1", "ask_coder", "{}")])
    first.output_text = "Asking the coder."
    router, api, _ = make_router([first, response("r2", text="Staged.")], slow_toolbox([]))
    router.run_turn("fix it")
    sent = api.requests[1]["input"]
    assert sent[:3] == [{"role": "user", "content": "fix it"}, {"role": "assistant", "content": "Asking the coder."},
                        {"type": "function_call", "call_id": "c1", "name": "ask_coder", "arguments": "{}"}]
    assert sent[3]["type"] == "function_call_output" and len(sent) == 4


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
    out = json.loads(next(i["output"] for i in api.requests[1]["input"] if i.get("type") == "function_call_output"))
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
    assert router.history == []
    assert any(e[0] == "note" and "stopped" in e[1] for e in hud.events)


PREFETCH = [{"tool": "read_breadboard"}, {"tool": "read_file", "args": {"path": "pins.h"}}]


def facts_toolbox(log, board=None):
    tb = Toolbox()

    @tb.register("read_breadboard", "", {"type": "object", "properties": {}}, label="SAM")
    def read_breadboard():
        log.append("board")
        return dict(board or {"ok": True, "mismatches": [{"signal": "SENSOR_PIN"}]})

    @tb.register("read_file", "", {"type": "object", "properties": {"path": {"type": "string"}}})
    def read_file(path):
        log.append(path)
        return {"ok": True, "path": path, "text": "#define SENSOR_PIN 15"}

    return tb


def test_bench_facts_go_out_with_the_first_call():
    """Act 1 on Oct 28 spent its first router round (4.4 s) only asking for the board read and pins.h."""
    log = []
    router, api, hud = make_router([response("r1", text="The sensor is on 4.")], facts_toolbox(log), prefetch=PREFETCH)
    router.run_turn("check my wiring")
    assert sorted(log) == ["board", "pins.h"] and len(api.requests) == 1
    first = api.requests[0]["input"][0]["content"]
    assert first.startswith("check my wiring") and "SENSOR_PIN" in first and "#define SENSOR_PIN 15" in first
    assert ("call", "SAM", "read_breadboard") in hud.events


def test_facts_the_conversation_has_are_not_sent_again():
    board = {"ok": True, "mismatches": []}
    router, api, _ = make_router([response(f"r{i}", text="ok") for i in range(3)], facts_toolbox([], board),
                                 prefetch=PREFETCH)
    router.run_turn("one")
    router.run_turn("two")
    second = asked(api.requests[1])
    assert second.count("unchanged since you last saw it") == 2 and "mismatches" not in second
    board["mismatches"] = [{"signal": "SENSOR_PIN"}]
    router.run_turn("three")
    third = asked(api.requests[2])
    assert "SENSOR_PIN" in third and third.count("unchanged") == 1  # only pins.h stayed the same


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
    assert all("_timing" not in i["output"] for i in api.requests[1]["input"] if i.get("type") == "function_call_output")


def test_chat_router_gets_the_same_bench_facts():
    router = ChatRouter(FakeChatClient([chat_response(content="ok")]), "glimmer", facts_toolbox([]),
                        Budget(10, PRICES), SilentHud(), "system", prefetch=PREFETCH)
    router.run_turn("check my wiring")
    assert "#define SENSOR_PIN 15" in router.messages[1]["content"]


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
    assert scene["ok"] and scene["mismatches"][0]["signal"] == "SENSOR_PIN"
    assert "point_at" not in tb.names(), "Act 1 has no arm step: live, the stub only said it wasn't wired"
    assert "call ask_coder first" in tb.dispatch("propose_patch", {"patch_id": "p9"})["error"]
    assert set(tb.names()) >= {"read_file", "ask_coder", "propose_patch", "read_breadboard", "look"}


def test_nothing_the_agent_reads_by_default_says_where_the_bug_is():
    """Nov 1: pins.h said "the sensor's jumper goes to GPIO 4, but this file says 15", and Spark quoted it, so a
    diagnosis proved nothing. The board read and the live readings have to find it."""
    tb = build_toolbox(Bench(ROOT, load_yaml("models.yaml"), client=None, budget=Budget(10, {}), sim=True))
    seen = {name: (ROOT / "wbc/agent/prompts" / name).read_text(encoding="utf-8") for name in ("system.md", "coder.md")}
    seen["the tools"] = json.dumps(tb.schemas())
    seen["the recording's note"] = tb.dispatch("read_breadboard", {})["note"]
    for path in ("firmware/dut_demo/include/pins.h", "firmware/dut_demo/src/main.cpp"):
        seen[path] = tb.dispatch("read_file", {"path": path})["text"]
    for name, text in seen.items():
        for hint in ("planted", "bug", "gpio 4", "gpio4", "says 15", "wrong pin", "wrong gpio"):
            assert hint not in text.lower(), f"{name} says {hint!r}"

def test_read_file_refuses_paths_outside_the_repo():
    tb = build_toolbox(Bench(ROOT, load_yaml("models.yaml"), None, Budget(10, {})))
    assert tb.dispatch("read_file", {"path": "../../etc/passwd"})["ok"] is False
    assert "SENSOR_PIN" in tb.dispatch("read_file", {"path": "firmware/dut_demo/include/pins.h"})["text"]


def test_schemas_in_both_formats():
    tb = build_toolbox(Bench(ROOT, load_yaml("models.yaml"), None, Budget(10, {})))
    assert all(s["type"] == "function" and "name" in s for s in tb.schemas())
    assert all("function" in s and "name" in s["function"] for s in tb.chat_schemas())


@pytest.mark.parametrize("name", ["router", "coder", "ears", "eyes"])
def test_models_yaml_prices_every_hosted_model(name):
    models = load_yaml("models.yaml")
    assert models[name]["model"] in models["budget"]["prices"]
