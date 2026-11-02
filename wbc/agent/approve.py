"""Enter at the keyboard: what every arm move waits for.

Voice can ask the agent to move something, but only a key press lets the arm go (the approval flow in
self-written-skills.md: never voice alone). While a tool waits here, the next typed line is its answer instead of a new
request: a bare Enter is yes, anything else no. What the mic hears never reaches it.
"""
from __future__ import annotations

import queue
import threading

WAIT_S = 120.0  # seconds a question waits for a key before it counts as no
CLOSING = object()  # what close() answers the question waiting with


class Keyboard:
    def __init__(self, say=print, wait_s: float = WAIT_S):
        self.say, self.wait_s = say, wait_s
        self._asking = threading.Lock()  # one question at a time
        self._lock = threading.Lock()
        self._waiting = None             # the answer queue of the question being asked
        self._closed = False             # after close(), every question is a no

    def ask(self, question: str, forever: bool = False) -> bool | None:
        """Ask, and wait for the next typed line: True only for a bare Enter, False for anything else or no answer in
        wait_s seconds, and None, a no too, once close() has run. `forever` waits for a line, close() or Ctrl-C
        however long they take, for a question whose no does something."""
        with self._asking:
            answer = queue.Queue(maxsize=1)
            with self._lock:
                if self._closed:
                    return None
                self._waiting = answer
            self.say(f"{question} Enter goes ahead; anything else stops.")
            try:
                line = answer.get() if forever else answer.get(timeout=self.wait_s)
            except queue.Empty:
                self.say(f"No key in {self.wait_s:.0f} s, so no.")
                return False
            finally:
                with self._lock:
                    self._waiting = None
            return None if line is CLOSING else line.strip() == ""

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Answer the question waiting, if any, with None, and every question after it: the session is ending (Ctrl-C,
        or the end of the input). A line typed before it stays that question's answer."""
        with self._lock:
            self._closed = True
            if self._waiting is not None:
                try:
                    self._waiting.put_nowait(CLOSING)
                except queue.Full:
                    pass

    def offer(self, line: str) -> bool:
        """A typed line: True when a question took it as its answer, so it isn't a new request."""
        with self._lock:
            if self._waiting is None:
                return False
            try:
                self._waiting.put_nowait(line)
            except queue.Full:  # answered already: a second key press isn't a request either
                pass
            return True
