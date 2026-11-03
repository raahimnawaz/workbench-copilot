"""Web HUD: the Muse orb page (hud/orb-light.html) showing the agent live, over Server-Sent Events.

`python -m wbc.agent --hud` serves the page at http://127.0.0.1:8766/ from the agent's own process, with nothing but
the standard library, and sends it each thing as it happens: the mic hearing someone, every model call from its start
to its time (Spark, Wonder, SAM and Llama, timed where the call goes out, and Voice Transcribe for the request's own
words), the request, the reply's first sentences (SUMMARY characters at most), the co-pilot speaking, and each turn's
total time and spend. Every event is one of the page's own functions with its
arguments (hud/README.md lists them), so all the page does is call it. The console prints its timeline as before.
"""
from __future__ import annotations

import contextlib
import functools
import html
import http.server
import json
import queue
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from wbc import config

from .hud import ConsoleHud
from .mouth import plain, sentences_of

PAGE = "orb-light.html"
PING_S = 15.0  # an SSE comment this often keeps a quiet feed open, and finds pages that have gone
BAR = 200  # the most call-bar events kept for a page that connects mid-turn
HEARD = 8  # the most requests heard and waiting for their turns
SUMMARY = 144  # the most characters of a reply the page shows


def page_models(models: dict) -> dict:
    """The page's name (MODELS in the page) for each model id models.yaml gives a role. Any other id is the
    router's: router.model, its fallback, or --router-model."""
    roles = (("coder", "wonder"), ("eyes", "sam"), ("ears", "transcribe"), ("backup", "open"), ("fast_arm", "llama"),
             ("open", "open"))
    return {models[role]["model"]: page for role, page in roles if role in models}


def reply_html(text: str) -> str:
    """A reply as the page's say() takes it: the words escaped, **bold** kept, on one line (say() finds sentences
    within a line)."""
    words = html.escape(" ".join(text.replace("`", "").split()), quote=False)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", words)


def summary(text: str, limit: int = SUMMARY) -> str:
    """What the page shows of a reply: its first sentences as plain words, whole, as many as fit in `limit`
    characters, or the first one cut at a word with an ellipsis. The rest is in the console."""
    words = plain(text)
    if not words:
        return ""
    shown = ""
    for sentence in sentences_of(words):
        if len(shown) + 1 + len(sentence) > limit:
            break
        shown = f"{shown} {sentence}".strip()
    return shown or words[:limit - 1].rsplit(" ", 1)[0].rstrip(",;:—-") + "…"


def dollars(usd: float) -> str:
    return f"${usd:.2f}" if usd >= 0.01 else f"${usd:.3f}"


class WebHud(ConsoleHud):
    """The console's timeline, and the same turns on the orb page.

    The orb's state and model follow from what's going on: speaking while the co-pilot speaks; listening while
    someone talks or their words are being transcribed; thinking while a turn runs; idle otherwise. It shows the
    newest model working, and between calls the last one that worked (the router before the first turn).
    """

    def __init__(self, models: dict, budget, port: int = 8766, backup: bool = False, folder: Path | None = None,
                 clock=time.monotonic):
        self.models = page_models(models)
        self.budget = budget
        self.clock = clock
        self._lock = threading.Lock()
        self._pages = []  # a queue for each page connected to /events
        self._flying = {}  # page model: [calls in flight, when the first of them began]
        self._order = []  # the page models of the calls in flight, newest last
        self._last = "open" if backup else "spark"
        self._hearing = self._speaking = False
        self._turn = None  # (number, began, spend before) while a turn runs
        self._heard = {}  # a request from the ears: (began, spend before, transcription seconds) for its turn
        self._transcribed = None  # the same for the last transcription
        self._shown = (None, None)  # the page's (state, model)
        self._you = self._say = None  # the page's request and reply; None leaves the page's own
        self._bar = []  # the call bar since the turn began, as events
        self.turns = 0
        self._show()
        try:
            self.server = http.server.ThreadingHTTPServer(
                ("127.0.0.1", port), functools.partial(_Page, directory=str(folder or config.ROOT / "hud")))
        except OSError as e:
            raise RuntimeError(f"The HUD can't serve on port {port} ({e.strerror}). Is another agent or a "
                               f"python -m http.server using it? Pick another with --hud PORT.") from None
        self.server.hud = self
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.note(f"HUD: http://127.0.0.1:{self.port}/ (open it in Chrome)")

    def watch(self, client):
        return Watched(client, self)

    @contextlib.contextmanager
    def calling(self, model: str):
        """One model call: the call bar gets the model as it starts and its time when it's done. Calls to one model
        that overlap (SAM's phrases go out together) are one entry, timed from the first start to the last end."""
        m = self.models.get(model, "spark")
        with self._lock:
            flying = self._flying.setdefault(m, [0, 0.0])
            if not flying[0]:
                flying[1] = self.clock()
                if m != "transcribe":  # every utterance is transcribed: only a request's opens its turn (turn())
                    self._call_bar("begin", m)
            flying[0] += 1
            self._order.append(m)
            spent = self.budget.spent
            self._show()
        try:
            yield
        finally:
            with self._lock:
                flying[0] -= 1
                self._order.remove(m)
                seconds = round(self.clock() - flying[1], 2)
                if not flying[0] and m != "transcribe":
                    self._call_bar("call", m, seconds)
                if m == "transcribe":
                    self._transcribed = (flying[1], spent, seconds)
                else:
                    self._last = m
                self._show()

    def hearing(self, on: bool) -> None:
        with self._lock:
            self._hearing = on
            self._show()

    def heard(self, request: str) -> None:
        with self._lock:
            if self._transcribed is not None:
                self._heard[request] = self._transcribed
            while len(self._heard) > HEARD:  # requests no turn took up, as in a mic check (--hear)
                del self._heard[next(iter(self._heard))]

    def turn(self, text: str) -> None:
        with self._lock:
            began, spent, transcribed = self._heard.pop(text, None) or (self.clock(), self.budget.spent, None)
            self.turns += 1
            self._turn = (self.turns, began, spent)
            self._bar = []
            self._send("clear")
            self._you, self._say = text, ""
            self._send("you", text)
            self._send("say", "", False)
            if transcribed is not None:  # the request's own transcription opens its call bar
                self._call_bar("call", "transcribe", transcribed)
            self._show()

    def reply(self, label: str, text: str) -> None:
        super().reply(label, text)
        with self._lock:
            self._say = reply_html(summary(text))
            self._send("say", self._say)

    def turn_over(self, failure: str = "") -> None:
        with self._lock:
            if self._turn is None:
                return
            number, began, spent = self._turn
            self._turn = None
            if failure:
                self._say = reply_html(summary(failure))
                self._send("say", self._say)
            self._call_bar("total", f"Turn {number} · {self.clock() - began:.0f} s · "
                                    f"{dollars(self.budget.spent - spent)}")
            self._show()

    def speaking(self, on: bool) -> None:
        with self._lock:
            self._speaking = on
            self._show()

    def close(self) -> None:
        """Stop serving: every page's feed ends, then the server."""
        with self._lock:
            for page in self._pages:
                page.put(None)
        self.server.shutdown()
        self.server.server_close()

    # The rest runs holding self._lock.

    def _show(self) -> None:
        if self._speaking:
            state = "speaking"
        elif self._hearing or "transcribe" in self._order:
            state = "listening"
        else:
            state = "thinking" if self._turn is not None else "idle"
        model = "transcribe" if state == "listening" else self._order[-1] if self._order else self._last
        if model != self._shown[1]:
            self._send("setModel", model)
        if state != self._shown[0]:
            self._send("setState", state)
        self._shown = (state, model)

    def _call_bar(self, *event) -> None:
        self._bar = self._bar[1 - BAR:] + [event]
        self._send(*event)

    def _send(self, *event) -> None:
        data = json.dumps(event)
        for page in self._pages:
            page.put(data)

    def _join(self) -> queue.Queue:
        """A feed for a page that has just connected, starting with what the page should show now."""
        page = queue.Queue()
        with self._lock:
            state, model = self._shown
            now = [("clear",), ("setModel", model), ("setState", state)]
            now += [("you", self._you)] if self._you is not None else []
            now += [("say", self._say, False)] if self._say is not None else []
            for event in now + self._bar:
                page.put(json.dumps(event))
            self._pages.append(page)
        return page

    def _leave(self, page: queue.Queue) -> None:
        with self._lock:
            if page in self._pages:
                self._pages.remove(page)


class _Page(http.server.SimpleHTTPRequestHandler):
    """The page and its files from hud/, and /events, the page's live feed. "/" opens the page in live mode."""

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(302)
            self.send_header("Location", f"/{PAGE}?live")
            self.end_headers()
        elif self.path == "/events":
            self._events()
        else:
            super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")  # a page edited mid-run loads fresh
        super().end_headers()

    def _events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        hud = self.server.hud
        page = hud._join()
        try:
            self.wfile.write(b"retry: 1000\n\n")  # after a break, the page asks again within a second
            while True:
                try:
                    data = page.get(timeout=PING_S)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    continue
                if data is None:
                    return
                self.wfile.write(f"data: {data}\n\n".encode())
        except OSError:  # the page went away
            pass
        finally:
            hud._leave(page)

    def log_message(self, format, *args) -> None:
        pass  # the console is the agent's timeline


class Watched:
    """A model client whose calls the web HUD times: responses.create and chat.completions.create each run inside
    hud.calling(model). Everything else is the client's own."""

    def __init__(self, client, hud):
        self._client, self._hud = client, hud

    @property
    def responses(self):
        return _Timed(self._client.responses, self._hud)

    @property
    def chat(self):
        return SimpleNamespace(completions=_Timed(self._client.chat.completions, self._hud))

    def __getattr__(self, name):
        return getattr(self._client, name)


class _Timed:
    def __init__(self, endpoint, hud):
        self._endpoint, self._hud = endpoint, hud

    def create(self, **kwargs):
        with self._hud.calling(kwargs.get("model")):
            return self._endpoint.create(**kwargs)

    def __getattr__(self, name):
        return getattr(self._endpoint, name)
