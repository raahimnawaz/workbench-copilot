"""Enter at the keyboard: what every arm move waits for.

Voice can ask the agent to move something, but only a key press lets the arm go (the approval flow in
self-written-skills.md: never voice alone). While a tool waits here, the next typed line is its answer instead of a new
request: a bare Enter is yes, anything else no. What the mic hears never reaches it.
"""
from __future__ import annotations

import queue
import threading

WAIT_S = 120.0  # seconds a question waits for a key before it counts as no


class Keyboard:
    def __init__(self, say=print, wait_s: float = WAIT_S):
        self.say, self.wait_s = say, wait_s
        self._asking = threading.Lock()  # one question at a time
        self._lock = threading.Lock()
        self._waiting = None             # the answer queue of the question being asked

    def ask(self, question: str) -> bool:
        """Ask, and wait for the next typed line: True only for a bare Enter, False for anything else or no answer in
        wait_s seconds."""
        with self._asking:
            answer = queue.Queue(maxsize=1)
            with self._lock:
                self._waiting = answer
            self.say(f"{question} Enter goes ahead; anything else stops.")
            try:
                line = answer.get(timeout=self.wait_s)
            except queue.Empty:
                self.say(f"No key in {self.wait_s:.0f} s, so no.")
                return False
            finally:
                with self._lock:
                    self._waiting = None
            return line.strip() == ""

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
