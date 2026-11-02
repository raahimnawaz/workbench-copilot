"""The web HUD: the orb page served by the agent, and each turn sent to it as calls to the page's own functions."""
import contextlib
import http.client
import io
import itertools
import json
import re
import shutil
import threading
from types import SimpleNamespace as NS

import pytest
from fakes import FakeChatClient, FakeClient, SilentHud, function_call, response

from wbc import config
from wbc.agent import __main__ as agent
from wbc.agent.budget import Budget
from wbc.agent.loop import Router
from wbc.agent.tools import Toolbox
from wbc.agent.web_hud import Watched, WebHud, page_models, reply_html
from wbc.config import ROOT, load_yaml

MODELS = load_yaml("models.yaml")
SPARK, WONDER, SAM, EARS = (MODELS[role]["model"] for role in ("router", "coder", "eyes", "ears"))
PRICES = MODELS["budget"]["prices"]
PAGE = (ROOT / "hud" / "orb-light.html").read_text(encoding="utf-8")
LIVE = {"setState", "setModel", "you", "say", "begin", "call", "total", "clear"}  # what the page's feed runs


@pytest.fixture
def hud():
    """A web HUD on a free port, its clock a second further on at every look."""
    hud = WebHud(MODELS, Budget(10, PRICES), port=0, clock=itertools.count().__next__)
    yield hud
    hud.close()


def drain(page) -> list:
    events = []
    while not page.empty():
        events.append(json.loads(page.get()))
    return events


class Feed:
    """The page's end of /events: every event the HUD sends, as it arrives."""

    def __init__(self, port: int):
        self.events, self.arrived = [], threading.Condition()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        conn.request("GET", "/events")
        self.response = conn.getresponse()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.response:
            if line.startswith(b"data: "):
                with self.arrived:
                    self.events.append(json.loads(line[6:]))
                    self.arrived.notify_all()

    def until(self, done) -> list:
        with self.arrived:
            assert self.arrived.wait_for(lambda: done(self.events), timeout=5), self.events
            return list(self.events)


def test_every_model_with_a_role_has_its_place_on_the_page():
    pages = page_models(MODELS)
    assert pages == {WONDER: "wonder", SAM: "sam", EARS: "transcribe", MODELS["backup"]["model"]: "open",
                     MODELS["fast_arm"]["model"]: "llama"}
    block = re.search(r"const MODELS = \{(.*?)\n\};", PAGE, re.S).group(1)
    assert set(pages.values()) | {"spark"} <= set(re.findall(r"^\s+(\w+): \{ name:", block, re.M))
    line = next(line for line in PAGE.splitlines() if "const live = {" in line)
    assert all(re.search(rf"\b{name}\b", line) for name in LIVE)


def test_the_page_opens_live_from_the_root_and_a_port_in_use_is_named(hud):
    conn = http.client.HTTPConnection("127.0.0.1", hud.port, timeout=5)
    conn.request("GET", "/")
    r = conn.getresponse()
    assert (r.status, r.getheader("Location")) == (302, "/orb-light.html?live")
    r.read()
    for path, has in (("/orb-light.html?live", b"new EventSource('events')"), ("/saved/siri-line.js", b"SiriLine")):
        conn.request("GET", path)
        r = conn.getresponse()
        assert r.status == 200 and has in r.read()
    with pytest.raises(RuntimeError, match=f"port {hud.port}"):
        WebHud(MODELS, Budget(10, PRICES), port=hud.port)


def test_a_typed_turn_reaches_the_page_call_by_call(hud):
    """Through the real router, its client watched by the HUD, and the page's feed over HTTP."""
    feed = Feed(hud.port)
    assert feed.until(lambda events: len(events) == 3) == [["clear"], ["setModel", "spark"], ["setState", "idle"]]
    client = hud.watch(FakeClient([response("r1", [function_call("c1", "ask_coder", "{}")]),
                                   response("w1", text="WHY: the wire is on GPIO 4"),
                                   response("r2", text="It's on GPIO 4, so I staged the fix.\n**Your turn:** flash it.")]))
    tb = Toolbox()

    @tb.register("ask_coder", "", {"type": "object", "properties": {}}, label="Wonder")
    def ask_coder():
        hud.budget.add_tokens(WONDER, client.responses.create(model=WONDER, input="fix it").usage)
        return {"ok": True}

    router = Router(client, SPARK, tb, hud.budget, hud, "system")
    hud.turn("Check my wiring <now>")
    router.run_turn("Check my wiring <now>")
    hud.turn_over()
    events = feed.until(lambda events: events[-1] == ["setState", "idle"] and len(events) > 3)[3:]
    assert events == [
        ["clear"], ["you", "Check my wiring <now>"], ["say", "", False], ["setState", "thinking"],
        ["begin", "spark"], ["call", "spark", 1],
        ["begin", "wonder"], ["setModel", "wonder"], ["call", "wonder", 1],
        ["begin", "spark"], ["setModel", "spark"], ["call", "spark", 1],
        ["say", "It's on GPIO 4, so I staged the fix. <b>Your turn:</b> flash it."],
        ["total", "Turn 1 · 7 s · $0.009"], ["setState", "idle"]]  # two Spark calls and one Wonder, 1,000 tokens in
    assert {event[0] for event in events} <= LIVE


def test_a_spoken_request_opens_its_turn_with_its_transcription(hud):
    page = hud._join()
    drain(page)
    hud.hearing(True)
    with hud.calling(EARS):
        hud.budget.add_audio(EARS, 3600)  # $0.18 an hour
    hud.heard("check my wiring")
    hud.hearing(False)
    hud.turn("check my wiring")
    hud.turn_over()
    assert drain(page) == [
        ["setModel", "transcribe"], ["setState", "listening"], ["begin", "transcribe"], ["call", "transcribe", 1],
        ["setModel", "spark"], ["setState", "idle"],
        ["clear"], ["you", "check my wiring"], ["say", "", False], ["call", "transcribe", 1], ["setState", "thinking"],
        ["total", "Turn 1 · 2 s · $0.18"], ["setState", "idle"]]  # from the start of its transcription


def test_overlapping_calls_to_a_model_are_one_entry_and_the_orb_shows_the_newest_still_working(hud):
    page = hud._join()
    drain(page)
    with contextlib.ExitStack() as wonder:
        wonder.enter_context(hud.calling(WONDER))
        with contextlib.ExitStack() as sam:
            first = sam.enter_context(contextlib.ExitStack())
            first.enter_context(hud.calling(SAM))
            sam.enter_context(hud.calling(SAM))  # SAM's phrases go out together
            first.close()
        assert drain(page) == [["begin", "wonder"], ["setModel", "wonder"], ["begin", "sam"], ["setModel", "sam"],
                               ["call", "sam", 2], ["setModel", "wonder"]]
    assert drain(page) == [["call", "wonder", 4]]  # and the orb stays on Wonder until another model starts


def test_speaking_shows_over_everything_and_a_failed_turn_says_so(hud):
    page = hud._join()
    drain(page)
    hud.turn("flash it")
    hud.speaking(True)
    hud.turn_over("That turn failed. The error is on the screen.")
    hud.speaking(False)
    assert drain(page)[4:] == [["setState", "speaking"], ["say", "That turn failed. The error is on the screen."],
                               ["total", "Turn 1 · 1 s · $0.000"], ["setState", "idle"]]


def test_a_page_that_connects_mid_turn_catches_up(hud):
    hud.turn("check my wiring")
    with hud.calling(SPARK):
        hud.reply("Spark", "Reading the board.")
        assert drain(hud._join()) == [["clear"], ["setModel", "spark"], ["setState", "thinking"],
                                      ["you", "check my wiring"], ["say", "Reading the board.", False],
                                      ["begin", "spark"]]


def test_a_reply_is_safe_html_on_one_line():
    assert reply_html("Row 26 <b>goes</b> to `GPIO 4`.\nSay **go** & I'll flash it.") == \
        "Row 26 &lt;b&gt;goes&lt;/b&gt; to GPIO 4. Say <b>go</b> &amp; I'll flash it."


def test_a_watched_client_times_every_call_even_one_that_fails(hud):
    page = hud._join()
    drain(page)
    chat = hud.watch(FakeChatClient([RuntimeError("no server")]))
    with pytest.raises(RuntimeError):
        chat.chat.completions.create(model=MODELS["backup"]["model"], messages=[])
    assert drain(page) == [["begin", "open"], ["setModel", "open"], ["call", "open", 1]]
    assert isinstance(chat, Watched) and chat.chat.completions.requests  # the rest is the client's own


def test_hud_puts_the_web_hud_in_front_of_every_model_client(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "config", tmp_path / "config")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "meta_client", lambda models: FakeClient([]))
    monkeypatch.setattr(config, "backup_client", lambda models: FakeChatClient([]))
    for backup in (False, True):
        _, _, hud, router, bench, _ = agent.build(True, backup, hud_port=0)
        try:
            assert isinstance(hud, WebHud) and isinstance(router.client, Watched) and bench.client._hud is hud
            assert hud._shown == ("idle", "open" if backup else "spark")
        finally:
            hud.close()


@pytest.mark.parametrize("flags, port", [([], None), (["--hud"], 8766), (["--hud", "8800"], 8800)])
def test_each_turn_and_how_it_ended_go_to_the_hud(monkeypatch, flags, port):
    built, hud = [], SilentHud()
    replies = iter(["Fixed.", RuntimeError("boom")])

    def run_turn(text):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    router = NS(label="Spark", model="spark", run_turn=run_turn)
    bench = NS(park=lambda home=True: None, stop_arm=lambda: None, hand=None, stopped=threading.Event(),
               can_go_home=lambda: False, resume=lambda: None)
    budget = NS(spent=0.0, limit=5.0, check=lambda: None, pop_warning=lambda: None)
    parts = ({"ears": {}, "coder": {}, "mouth": {}}, budget, hud, router, bench,
             NS(offer=lambda line: False, close=lambda: None))
    monkeypatch.setattr(agent, "build", lambda *a: built.append(a) or parts)
    monkeypatch.setattr(agent.coder, "compile_in_sandbox", lambda *a: None)
    monkeypatch.setattr("sys.stdin", io.StringIO("check my wiring\nflash it\n"))
    assert agent.main(flags) == 0 and built[0][7] == port
    assert hud.shown == [("turn", "check my wiring"), ("turn_over", ""), ("turn", "flash it"),
                         ("turn_over", agent.FAILED)]
