"""The arm's fast path (wbc/agent/fast_arm.py): its rules, the local model's veto, and the router running it first."""
import json
from types import SimpleNamespace as NS

import pytest
from fakes import FakeChatClient, FakeClient, SilentHud, chat_response, response

from wbc.agent import fast_arm
from wbc.agent.budget import Budget
from wbc.agent.fast_arm import FastArm, call, goes, parse, rules, said
from wbc.agent.loop import Router
from wbc.agent.memory import Memory, Notebook, Session
from wbc.agent.tools import Bench, build_toolbox
from wbc.config import ROOT, load_yaml

CFG = load_yaml("models.yaml")["fast_arm"]
THINGS, PLACES = CFG["things"], CFG["places"]
NOTES = ['The screwdriver goes on the yellow "sticky note" in the long recess of the bit case.',
         'The tweezers go in the "blue tray".', "The user wants short spoken answers."]


@pytest.mark.parametrize("text, holding, expect", [
    # the take's own style: the thing and the place named
    ("pick my screwdriver up and put it on the sticky note", None, call("move_thing", "screwdriver", "sticky note")),
    ("Copilot, put the screwdriver in the cup.", None, call("move_thing", "screwdriver", "cup")),
    ("um could you move the plyers over to the tray", None, call("move_thing", "pliers", "tray")),
    ("put the screwdriver away", None, call("move_thing", "screwdriver", "sticky note")),  # the note's quoted place
    ("grab the pliers", None, call("pick_up", "pliers")),
    ("put it on the sticky note", "screwdriver", call("put_down", place="sticky note")),
    ("put it back", "screwdriver", call("put_down", place="sticky note")),
    ("go back to your rest position", None, call("arm_home")),
    ("send the arm home", "screwdriver", call("arm_home")),
    # Spark's: questions, negations, talk about yourself, a bare "that", things it doesn't know, other senses
    ("what's on the sticky note", None, call("not_mine")),
    ("why didn't you put my screwdriver in the cup", None, call("not_mine")),
    ("don't move the screwdriver", None, call("not_mine")),
    ("wait, put the screwdriver in the cup", None, call("not_mine")),
    ("I put my screwdriver in the cup.", None, call("not_mine")),  # the rules' wrong moves on Nov 1
    ("I'm going to pick my screwdriver up myself", None, call("not_mine")),
    ("grab that and put it away", None, call("not_mine")),
    ("put the multimeter in the cup", None, call("not_mine")),
    ("pick the faster option", None, call("not_mine")),
    ("move the trigger to one volt", None, call("not_mine")),
    ("hold on, let me recheck the ground wire", None, call("not_mine")),
    ("remember that the screwdriver goes in the cup", None, call("not_mine")),
    ("go ahead", None, call("not_mine")),
    ("yes", None, call("not_mine")),
    ("put the tweezers in the cup", "screwdriver", call("not_mine")),  # it holds something else
])
def test_rules(text, holding, expect):
    assert rules(text, holding, THINGS, PLACES, NOTES) == expect


def test_the_notebook_says_where_things_go_by_their_quoted_names():
    assert goes(NOTES, THINGS, PLACES) == {"screwdriver": "sticky note", "tweezers": "tray"}
    assert goes(['The pliers go in the "red bin".'], THINGS, PLACES) == {"pliers": "red bin"}  # as written


@pytest.mark.parametrize("answer, obj", [
    ('{"tool": "pick_up", "thing": "pliers", "place": null}', {"tool": "pick_up", "thing": "pliers", "place": None}),
    ('Sure: {"tool": "arm_home"} done', {"tool": "arm_home"}),
    ("I think you want the arm", {}), ("[1, 2]", {}), ("", {}),
])
def test_parse_takes_the_answers_json_object(answer, obj):
    assert parse(answer) == obj


@pytest.mark.parametrize("tool, result, line", [
    ("move_thing", {"ok": True, "moved": "screwdriver", "onto": "sticky note"},
     "The screwdriver is on the sticky note."),
    ("move_thing", {"ok": True, "moved": "pliers", "onto": "cup"}, "The pliers are in the cup."),
    ("put_down", {"ok": True, "put": "screwdriver", "on": "blue tray"}, "The screwdriver is in the blue tray."),
    ("pick_up", {"ok": True, "holding": "pliers"}, "I've got the pliers."),
    ("arm_home", {"ok": True, "home": True}, "Back at rest."),
])
def test_what_it_says_once_the_move_is_done(tool, result, line):
    assert said(tool, result) == line


class Llama:
    """mlx_lm.server as the OpenAI client sees it: each answer in turn, or an exception to raise."""

    def __init__(self, *answers):
        self.answers, self.asked = list(answers), []
        self.chat = NS(completions=self)

    def create(self, **kwargs):
        self.asked.append(kwargs)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return NS(choices=[NS(message=NS(content=answer))])


def answer(tool, thing=None, place=None):
    return json.dumps({"tool": tool, "thing": thing, "place": place})


def bench(tmp_path, notes=NOTES):
    b = Bench(ROOT, {}, None, Budget(10, {}), sim=True)
    notebook = Notebook(tmp_path / "notebook.jsonl")
    for note in notes:
        notebook.add(note)
    return b, build_toolbox(b), notebook


def ready(tmp_path, *answers, notes=NOTES):
    b, tb, notebook = bench(tmp_path, notes)
    llama, hud = Llama(*answers), SilentHud()
    fast = FastArm({**CFG, "keep_warm_s": 0}, tb, b, notebook, hud, client=llama)
    fast.ready = True
    return fast, llama, hud, b


def test_an_arm_command_runs_only_when_the_model_names_the_same_kind(tmp_path):
    fast, llama, hud, _ = ready(tmp_path, answer("move_thing", "screw driver", "post-it"),
                                answer("pick_up", "screwdriver"), answer("arm_home"), answer("not_mine"), "no idea")
    take = "put the screwdriver in the cup"
    run = {"tool": "move_thing", "args": {"thing": "screwdriver", "place": "cup"}}
    assert fast.take(take) == run  # with the rules' names, not the model's
    assert fast.take(take) == run  # pick_up is the same kind: carrying a thing
    for _ in range(3):  # another kind, not the arm's, no answer: all Spark's
        assert fast.take(take) is None
    assert ("calling", CFG["model"]) in hud.shown and len(llama.asked) == 5


def test_what_the_rules_dont_take_never_reaches_the_model(tmp_path):
    fast, llama, _, _ = ready(tmp_path)
    for text in ("check my wiring against the firmware", "grab that and put it away", "what should this read?"):
        assert fast.take(text) is None
    assert llama.asked == []
    fast.ready = False  # before the server has answered
    assert fast.take("put the screwdriver in the cup") is None and llama.asked == []


def test_a_slow_or_missing_server_leaves_the_request_to_spark(tmp_path):
    fast, llama, hud, _ = ready(tmp_path, TimeoutError("timed out"))
    assert fast.take("put the screwdriver in the cup") is None
    assert any("didn't answer (TimeoutError), so Spark takes it" in e[1] for e in hud.events if e[0] == "note")


def test_the_model_gets_the_prompt_measured_on_nov_1_with_the_bench_state(tmp_path):
    fast, llama, _, b = ready(tmp_path, answer("put_down", place="sticky note"))
    b.sim_held = "screwdriver"
    assert fast.take("put it back") == {"tool": "put_down", "args": {"thing": "sticky note"}}
    request = llama.asked[0]
    assert (request["model"], request["temperature"], request["max_tokens"]) == (CFG["model"], 0, 64)
    system, user = (m["content"] for m in request["messages"])
    assert system == fast_arm.SYSTEM and '(2) The tweezers go in the "blue tray".' in user
    assert "The arm is holding: the screwdriver, which it picked up." in user and user.endswith('"put it back"')


def test_start_turns_it_on_only_once_the_server_answers(tmp_path):
    b, tb, notebook = bench(tmp_path)
    hud = SilentHud()
    on = FastArm({**CFG, "keep_warm_s": 0}, tb, b, notebook, hud, client=Llama(answer("not_mine")))
    slept = []
    off = FastArm({**CFG, "keep_warm_s": 0}, tb, b, notebook, hud, client=Llama(*[ConnectionError("refused")] * 3),
                  sleep=slept.append)
    assert on.start() and on.ready and not off.start() and not off.ready
    assert slept == [fast_arm.START_GAP_S] * (fast_arm.START_TRIES - 1)
    notes = [e[1] for e in hud.events if e[0] == "note"]
    assert notes[0].startswith("fast path on: Llama 3.2 3B answered in")
    assert notes[1] == ("fast path off: Llama 3.2 3B at http://127.0.0.1:8090/v1 didn't answer "
                        "(ConnectionError: refused)")


def test_a_start_that_fails_once_tries_again_and_keeping_it_warm_turns_it_on_later(tmp_path):
    b, tb, notebook = bench(tmp_path)
    hud = SilentHud()
    retried = FastArm({**CFG, "keep_warm_s": 0}, tb, b, notebook, hud,
                      client=Llama(AttributeError("a module half loaded"), answer("not_mine")), sleep=lambda s: None)
    assert retried.start() and retried.ready
    class Enough(Exception):
        pass
    waits = []

    def sleep(seconds):  # start()'s gaps between tries, one keep-warm wait, and then the test has seen enough
        waits.append(seconds)
        if len(waits) > fast_arm.START_TRIES:
            raise Enough
    later = FastArm(CFG, tb, b, notebook, hud, client=Llama(*[ConnectionError("refused")] * 3, answer("not_mine")),
                    sleep=sleep)
    started = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fast_arm.threading, "Thread", lambda target, daemon: NS(start=lambda: started.append(target)))
        assert not later.start() and not later.ready
    (keep_warm,) = started
    with pytest.raises(Enough):
        keep_warm()
    assert later.ready and waits == [fast_arm.START_GAP_S] * (fast_arm.START_TRIES - 1) + [CFG["keep_warm_s"]] * 2
    notes = [e[1] for e in hud.events if e[0] == "note"]
    assert notes[-2].endswith("didn't answer (ConnectionError: refused); it asks again every 20 s")
    assert notes[-1] == "fast path on: Llama 3.2 3B answered"


# The router runs it first


def router_with(tmp_path, fast, scripted=()):
    memory = Memory(Session(tmp_path / "session.json"), fast.notebook)
    client = FakeClient(list(scripted))
    router = Router(client, "spark", fast.toolbox, Budget(10, {"spark": {"in": 1, "out": 1}}), fast.hud, "system",
                    memory=memory, fast=fast)
    return router, client.responses, memory


def test_a_command_it_takes_runs_and_is_said_with_no_call_to_spark(tmp_path):
    fast, _, hud, b = ready(tmp_path, answer("move_thing", "screwdriver", "cup"))
    router, api, memory = router_with(tmp_path, fast)
    assert router.run_turn("put the screwdriver in the cup") == "The screwdriver is in the cup."
    assert api.requests == [] and ("reply", "fast path", "The screwdriver is in the cup.") in hud.events
    assert router.history == [{"role": "user", "content": "put the screwdriver in the cup"},
                              {"role": "assistant", "content": "The screwdriver is in the cup."}]
    [turn] = memory.session.turns
    assert turn["calls"][0]["tool"] == "move_thing" and turn["calls"][0]["ok"] and turn["live"]


def test_spark_hears_about_it_in_the_next_turn(tmp_path):
    fast, _, _, _ = ready(tmp_path, answer("move_thing", "screwdriver", "cup"))
    router, api, _ = router_with(tmp_path, fast, [response("r1", text="It's in the cup, as you asked.")])
    router.run_turn("put the screwdriver in the cup")
    router.run_turn("where did you put it?")
    assert api.requests[0]["input"][:2] == router.history[:2]


def test_a_move_that_fails_goes_to_spark_to_explain_without_running_again(tmp_path):
    fast, _, _, _ = ready(tmp_path, answer("move_thing", "screwdriver", "sticky note"))  # none on the recorded desk
    router, api, memory = router_with(tmp_path, fast, [response("r1", text="I can't see a sticky note.")])
    assert router.run_turn("put the screwdriver on the sticky note") == "I can't see a sticky note."
    sent = api.requests[0]["input"][-1]["content"]
    assert sent.startswith("put the screwdriver on the sticky note\n\nThe arm's fast path already ran move_thing")
    assert "locate found no sticky note on the desk" in sent
    [turn] = memory.session.turns
    assert turn["user"] == "put the screwdriver on the sticky note" and [c["tool"] for c in turn["calls"]] == [
        "move_thing"] and not turn["calls"][0]["ok"]


@pytest.mark.parametrize("text", ["Copilot, put my screwdriver in my cup and add my screen.",
                                  "Put my screwdriver in my cup, and while it does that, add my screen.",
                                  "Add my screen, and put my screwdriver in my cup.",
                                  "While you put my screwdriver in my cup, add my screen to the board.",
                                  "Put my screwdriver in my cup and take a reading.",
                                  "Move the screwdriver to the tray, then go home.",
                                  "Put the screwdriver in the cup and put the tweezers on the sticky note."])
def test_a_request_with_a_clause_for_something_else_goes_to_spark_whole(text):
    """So Spark runs the arm's move and the other request side by side, instead of the fast path dropping half."""
    assert fast_arm.rules(text, None, THINGS, PLACES)["tool"] == "not_mine"


@pytest.mark.parametrize("text", ["Copilot, pick my screwdriver up and put it on the sticky note.",
                                  "Put my screwdriver in my cup, please.", "Put my screwdriver in my cup, thanks.",
                                  "Um, put my screwdriver in my cup.", "Co-pilot, put my screwdriver in my cup.",
                                  "Put my screwdriver, in my cup."])
def test_clauses_that_all_move_the_arm_or_are_courtesy_stay_on_the_fast_path(text):
    assert fast_arm.rules(text, None, THINGS, PLACES)["tool"] == "move_thing"
