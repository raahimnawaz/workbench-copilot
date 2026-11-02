"""Spoken replies: the co-pilot says the first sentence or two of each reply out loud with macOS `say`.

Meta's API has no text-to-speech model, so speech stays on the Mac. The HUD keeps the whole reply and every tool
call; speech carries only the short answer. Sound the mic recorded while it spoke, or within `tail_s` after, is
`busy()`, and the ears drop it, so the co-pilot never hears its own voice as a request. That's judged by when the
sound was recorded, not when the ears get to it, since they fall behind while a transcription is out.
"""
from __future__ import annotations

import collections
import re
import shutil
import subprocess
import threading
import time

SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"')\]])\s+")  # also before "pins.h ...", and after a quote
ABBREVIATIONS = {"e.g.", "i.e.", "vs.", "approx.", "ca.", "no.", "min.", "max.", "fig."}  # these don't end one


def sentences_of(text: str) -> list:
    parts = []
    for piece in SENTENCE_END.split(text):
        if parts and parts[-1].rsplit(" ", 1)[-1].lstrip("(\"'[").lower() in ABBREVIATIONS:
            parts[-1] += " " + piece
        else:
            parts.append(piece)
    return parts


def spoken(text: str, sentences: int = 2, limit: int = 240) -> str:
    """The part of a reply worth saying: its first sentences as plain words, at most `limit` characters.

    Code blocks are dropped, since `say` would read them symbol by symbol, and markdown marks go. Sentences are said
    whole or not at all, since a cut one can lose its "unless". A first sentence over the limit is cut at a word and
    says so.
    """
    text = re.sub(r"```.*?(```|$)", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # a link says its words
    text = re.sub(r"^\s*(#+|[-*•]|\d+\.)\s+", "", text, flags=re.MULTILINE)  # headings, bullets, numbers
    text = re.sub(r"[`*]+|(?<!\w)_+|_+(?!\w)", "", text)  # code marks, bold and italics, not snake_case
    text = " ".join(text.split())
    said = ""
    for sentence in sentences_of(text)[:sentences]:
        if said and len(said) + 1 + len(sentence) > limit:
            break
        said = f"{said} {sentence}".strip()
    if len(said) > limit:
        said = said[:limit].rsplit(" ", 1)[0].rstrip(",;:—-.!?") + ". The rest is on the screen."
    return said


class Mouth:
    def __init__(self, cfg: dict, command: str | None = None, popen=subprocess.Popen, clock=time.monotonic,
                 hud=None):
        self.command = command or shutil.which("say")  # None off macOS: replies stay on the HUD
        self.voice = cfg.get("voice")
        self.rate = cfg.get("rate")
        self.sentences = cfg.get("sentences", 2)
        self.tail_s = cfg.get("tail_s", 0.4)
        self.popen = popen
        self.clock = clock
        self.hud = hud  # told when speech starts and when the last of it ends, for the web HUD's orb
        self._lock = threading.Lock()
        self._proc = None
        self._spans = collections.deque(maxlen=8)  # [start, end] of recent speech; end is None while it lasts
        self.waiter = None  # stamps when the latest speech ended

    def speak(self, text: str) -> None:
        """Start saying the short part of `text` and return at once. A new reply cuts off the one before."""
        words = spoken(text, self.sentences)
        if not words or not self.command:
            return
        args = [self.command]
        if self.voice:
            args += ["-v", str(self.voice)]
        if self.rate:
            args += ["-r", str(self.rate)]
        with self._lock:
            self._stop()
            try:
                proc = self.popen(args + ["--", words], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
            except OSError:  # `say` gone or not runnable: the replies stay on the HUD, and the turn goes on
                self.command = None
                return
            span = [self.clock(), None]
            self._proc = proc
            self._spans.append(span)
        if self.hud is not None:
            self.hud.speaking(True)
        self.waiter = threading.Thread(target=self._ended, args=(proc, span), daemon=True)
        self.waiter.start()

    def _ended(self, proc, span: list) -> None:
        proc.wait()
        with self._lock:
            span[1] = self.clock()
            over = self._proc is proc or self._proc is None  # not cut off by a newer reply, which goes on
        if over and self.hud is not None:
            self.hud.speaking(False)

    def busy(self, at: float | None = None) -> bool:
        """Whether sound recorded at `at` (now if None) may hold its voice: while it spoke, or `tail_s` after."""
        at = self.clock() if at is None else at
        with self._lock:
            return any(start <= at and (end is None or at < end + self.tail_s) for start, end in self._spans)

    def stop(self) -> None:
        with self._lock:
            self._stop()

    def _stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()  # its waiter stamps the end
        self._proc = None
