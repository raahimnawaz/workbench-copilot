"""Voice front end, hands free: say "Copilot," and your request. The request becomes the router's next turn.

The mic stays open while voice is on. A loudness gate cuts the stream into utterances, Muse Voice Transcribe
transcribes each one, and only those that start with the wake phrase ("Copilot, check my wiring") go to the
router. The wake phrase alone ("Copilot.") makes the next utterance the request, and after a reply that asks
something, so does a short answer ("Yes.") for ANSWER_S seconds. While the co-pilot speaks a reply (`mouth.py`), what
the mic picks up is dropped, except the stop. "Copilot, stop", or "Stop!" on its own, never becomes a request: it goes
straight to the stop, which holds the arm where it is, even while a turn runs. Streaming partial transcripts over the
realtime socket is the v2 upgrade (sheet 08).

The request format was checked against the live API on 2026-10-26: a multipart upload with a JSON
`request` part (model, audioEncoding, keywords) and an `audio` part, answered with {"transcript": ...}.
"""
from __future__ import annotations

import collections
import io
import json
import queue
import re
import time
import wave

# httpx is imported here, not when the ears start: openai reads sys.modules["httpx"] without importing it, so a request
# on another thread while the ears' thread imports it fails ("partially initialized module 'httpx'").
import httpx
import numpy as np

FRAME_S = 0.03  # the loudness gate works on 30 ms frames
FILLERS = ("hey", "ok", "okay", "um", "uh", "so", "and")  # allowed before the wake phrase
AWAKE_S = 8.0  # after the wake phrase alone, how long the next utterance counts as the request
ANSWER_S = 8.0  # after a reply that asks something, how long a short answer counts without the wake phrase
ANSWER_WORDS = 5  # the most words such an answer has, as in "yes, go ahead and flash"
STOPS = ("stop", "halt", "freeze")  # the stop's words: after the wake phrase, or a whole utterance on their own
STOP_ALSO = ("it", "now", "please", "the", "arm", "copilot")  # what "Stop!" on its own may come with ("stop it now")
STOP_PIECE_S = 2.0  # while the co-pilot speaks, the mic is cut into pieces this long, each heard only for the stop
# How a question opens even when it ends in ".": a verb such as "does", then its subject ("Does the LED blink now").
# A verb followed by anything else is a statement: "Should be fixed now.", "Will do.", "Can't reach it."
ASKING = ("do", "does", "did", "is", "are", "was", "were", "can", "could", "should", "shall", "will", "would", "may",
          "might", "have", "has")
SUBJECTS = ("i", "you", "we", "it", "they", "he", "she", "this", "that", "these", "those", "there", "the", "a", "an",
            "my", "your", "its", "our", "their", "any", "both", "all", "each", "anything", "everything", "something")
_QUESTION = re.compile(
    r"(?:(?:so|and|but|now|then|ok|okay|also)\W+)?"  # "So, is it blinking"
    r"(?:(?:what|which|where|how|who|why)\s+(?:\w+\s+)?)?"  # "Which thing should I move"
    rf"(?:(?:{'|'.join(ASKING)})\s+(?:{'|'.join(SUBJECTS)})|want\s+(?:me|to))\b", flags=re.IGNORECASE)
# A reply that asks to be told something back: "say go and I'll flash it", "let me know if it blinks"
_INVITES = re.compile(r"\b(?:say\s+[\"'“‘]?(?:go|yes)|tell me|let me know)\b", flags=re.IGNORECASE)


def pcm_to_wav(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit, the only format the file endpoint accepts
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def parse_transcript(payload: dict) -> str:
    """The API answers with "transcript"; the other shapes are kept in case a later version changes it."""
    for key in ("transcript", "text"):
        if isinstance(payload.get(key), str):
            return payload[key].strip()
    for key in ("turns", "segments", "results"):
        items = payload.get(key)
        if isinstance(items, list):
            return " ".join(str(i.get("text", "")).strip() for i in items if isinstance(i, dict)).strip()
    raise ValueError(f"unrecognized transcript response: {list(payload)}")


def dbfs(rms: float) -> float:
    """A 16-bit signal level in dB below full scale, -inf for digital silence."""
    return 20 * float(np.log10(rms / 32768)) if rms > 0 else float("-inf")


def _letters(phrase: str) -> str:
    """A pattern for the wake phrase, which may come split or hyphenated ("co-pilot", "Co pilot")."""
    return r"[\W_]*".join(re.escape(c) for c in phrase.lower() if c.isalnum())


def wake_request(text: str, phrase: str) -> str | None:
    """What follows the wake phrase at the start of `text` ("" for the phrase alone), or None if it isn't there.

    The phrase may come split or hyphenated ("co-pilot", "Co pilot") and after a filler ("Hey, copilot").
    """
    m = re.match(rf"\W*(?:(?:{'|'.join(FILLERS)})\W+)?{_letters(phrase)}\b\W*", text, flags=re.IGNORECASE)
    return text[m.end():].strip() if m else None


def is_stop(request: str) -> bool:
    """Whether a request is the stop: "stop", "stop it", "please stop the arm", "halt"."""
    return re.match(rf"\W*(?:please\W+)?(?:{'|'.join(STOPS)})\b", request, flags=re.IGNORECASE) is not None


def bare_stop(text: str) -> bool:
    """Whether a whole utterance is the stop without the wake phrase: "Stop!", "Stop it, stop."."""
    words = re.findall(r"[a-z]+", text.lower())
    return any(w in STOPS for w in words) and all(w in STOPS or w in STOP_ALSO for w in words)


def said_stop(text: str, phrase: str) -> bool:
    """Whether "Copilot, stop" is anywhere in `text`, which may be the co-pilot's own words heard by the mic: they never
    name it."""
    return re.search(rf"\b{_letters(phrase)}\W+(?:please\W+)?(?:{'|'.join(STOPS)})\b", text,
                     flags=re.IGNORECASE) is not None


def asks(reply: str) -> bool:
    """Whether a reply waits for an answer: it ends with "?", its last sentence opens the way a question does though it
    ends with "." ("It's flashed. Does the LED blink now, though the latest photo shows no wire to pin 4."), or
    it asks to be told something back ("say go and I'll flash it"). A question it answers itself ("Is it? No.") doesn't
    count."""
    if re.search(r"\?[\"')\]]*\s*$", reply) or _INVITES.search(reply):
        return True
    last = re.split(r"(?<=[.!?])\s+", reply.strip())[-1]
    return any(_QUESTION.match(clause.lstrip("\"'“‘([ ")) for clause in re.split(r"[,;:–—]", last))


class Question:
    """Whether the co-pilot's last reply asked something: for ANSWER_S seconds after it was given, or after its speech
    ended, a short answer without the wake phrase ("Yes.") is the request. main() sets it at the end of each turn, and
    the ears close it once they've passed on a request."""

    def __init__(self):
        self.at = None  # when the last reply was given, if it asked something

    def asked(self, reply: str, at: float) -> None:
        self.at = at if asks(reply) else None

    def close(self) -> None:
        self.at = None

    def answer(self, text: str, now: float, spoke: float) -> bool:
        """Whether `text`, heard `now`, answers it: short, and within ANSWER_S of the reply or of its speech's end
        (`spoke`, when the mic last heard it)."""
        at = self.at
        return (at is not None and now < max(at, spoke) + ANSWER_S
                and 0 < len(re.findall(r"[\w']+", text)) <= ANSWER_WORDS)


class Segmenter:
    """Cuts a microphone stream into utterances by loudness.

    Speech starts once 3 frames in a row are `start_db` above the room's noise floor (and above `min_rms`), and
    it ends after `hangover_s` of quiet. Each utterance keeps up to `preroll_s` from before it started, so the
    first syllable survives. Bursts with less than `min_s` of loud audio (clicks, a bumped bench) are dropped,
    and speech running past `max_s` is cut there.
    """

    def __init__(self, rate: int = 16000, start_db: float = 12.0, min_rms: float = 150.0, hangover_s: float = 0.8,
                 preroll_s: float = 0.3, min_s: float = 0.25, max_s: float = 15.0):
        self.frame_bytes = 2 * int(rate * FRAME_S)
        self.ratio = 10 ** (start_db / 20)
        self.min_rms = min_rms
        self.hangover = round(hangover_s / FRAME_S)
        self.min_frames = round(min_s / FRAME_S)
        self.max_frames = round(max_s / FRAME_S)
        self.floor = None  # the room's RMS outside speech: drops at once, rises slowly
        self._pending = b""
        self._before = collections.deque(maxlen=round(preroll_s / FRAME_S))
        self._loud_run = 0
        self._speech = None  # frames of the utterance in progress
        self._voiced = 0
        self._quiet = 0

    def threshold(self) -> float:
        return max(self.min_rms, (self.floor or 0.0) * self.ratio)

    @property
    def hearing(self) -> bool:
        """Whether an utterance is in progress: it has started and not yet ended."""
        return self._speech is not None

    def reset(self) -> bytes | None:
        """Stop listening mid-stream, keeping the room's noise floor.

        An utterance that had already gone quiet is over, so it comes back, to be transcribed like any other. One
        still going, and the audio before it, are dropped.
        """
        done = None
        if self._speech is not None and self._quiet > 0 and self._voiced >= self.min_frames:
            done = b"".join(self._speech)
        self._pending = b""
        self._before.clear()
        self._loud_run = 0
        self._speech = None
        return done

    def feed(self, pcm: bytes) -> list:
        """Add audio and return the utterances that ended in it, as 16-bit PCM."""
        self._pending += pcm
        done = []
        while len(self._pending) >= self.frame_bytes:
            frame, self._pending = self._pending[: self.frame_bytes], self._pending[self.frame_bytes:]
            rms = float(np.sqrt(np.mean(np.frombuffer(frame, np.int16).astype(np.float64) ** 2)))
            if self.floor is None:
                self.floor = rms
            loud = rms > self.threshold()
            if self._speech is None:
                if not loud:
                    self.floor = rms if rms < self.floor else self.floor + 0.02 * (rms - self.floor)
                self._loud_run = self._loud_run + 1 if loud else 0
                self._before.append(frame)
                if self._loud_run >= 3:
                    self._speech, self._voiced, self._quiet = list(self._before), self._loud_run, 0
                    self._before.clear()
                continue
            self._speech.append(frame)
            self._voiced += loud
            self._quiet = 0 if loud else self._quiet + 1
            if self._quiet >= self.hangover or len(self._speech) >= self.max_frames:
                if self._voiced >= self.min_frames:
                    done.append(b"".join(self._speech))
                self._speech, self._loud_run = None, 0
        return done


class Transcriber:
    def __init__(self, cfg: dict, api_key: str, budget, http=None):
        self.cfg = cfg
        self.api_key = api_key
        self.budget = budget
        self.http = http or httpx.Client(timeout=30)

    def transcribe(self, wav: bytes, seconds: float) -> str:
        self.budget.check()
        request = {"model": self.cfg["model"], "audioEncoding": "WAV"}
        if self.cfg.get("keywords"):
            request["keywords"] = list(self.cfg["keywords"])
        r = self.http.post(
            self.cfg["url"],
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={"request": (None, json.dumps(request), "application/json"),
                   "audio": ("turn.wav", wav, "audio/wav")},
        )
        r.raise_for_status()
        self.budget.add_audio(self.cfg["model"], seconds)
        return parse_transcript(r.json())


class Microphone:
    """The bench mic as a stream of (time.monotonic() when it arrived, 16-bit mono chunk). The Mac Studio has no
    built-in mic, so plug one in."""

    def __init__(self, rate: int = 16000, device: str | None = None):
        try:
            import sounddevice
        except ImportError:
            raise RuntimeError("Voice needs the extras: pip install -e '.[voice]'") from None
        self.rate = rate
        self._chunks = queue.Queue()
        # sounddevice matches `device` against part of the name, so "Razer Seiren Mini" finds the Razer.
        self._stream = sounddevice.RawInputStream(samplerate=rate, channels=1, dtype="int16", device=device,
                                                  callback=lambda data, frames, t, status: self._chunks.put(
                                                      (time.monotonic(), bytes(data))))
        self.name = sounddevice.query_devices(self._stream.device)["name"]

    def chunks(self):
        self._stream.start()
        while True:
            yield self._chunks.get()


def listen_forever(on_text, cfg: dict, api_key: str, budget, hud, mic=None, transcriber=None,
                   clock=time.monotonic, muted=None, on_stop=None, question=None) -> None:
    """Loop: cut the mic stream into utterances, transcribe each, and hand on the ones addressed to the co-pilot.

    Audio for which `muted(t)` is true, t being when it was recorded (the co-pilot speaking), is dropped, along with
    any utterance it cut into. One that had already gone quiet when the co-pilot started is still heard.

    With `on_stop`, the stop ("Copilot, stop", or "Stop!" on its own) calls on_stop() at once instead of becoming a
    request, and it's heard over the co-pilot's voice too: while it speaks, the mic is cut into STOP_PIECE_S pieces,
    each transcribed only to look for "Copilot, stop". With `question` (a Question), a short answer without the wake
    phrase is the request while the co-pilot's last reply waits for one.

    The HUD hears about each utterance from its start until it's transcribed, and about each request heard.
    """
    try:
        mic = mic or Microphone(cfg.get("sample_rate", 16000), cfg.get("input_device"))
    except Exception as e:  # no voice extras, no such mic, or PortAudio refusing the stream
        hud.note(f"voice off: {e}")
        return
    transcriber = transcriber or Transcriber(cfg, api_key, budget)
    phrase = cfg.get("wake_phrase", "copilot")
    gate = Segmenter(mic.rate)
    under = Segmenter(mic.rate, max_s=STOP_PIECE_S) if on_stop is not None else None  # over its voice: the stop only
    hud.note(f'voice on {mic.name}: say "{phrase.capitalize()}," and your request')
    listened, checked, awake_until, hearing, spoke, before = 0.0, False, 0.0, False, 0.0, ""

    def transcribe(pcm: bytes, seconds: float, then: str = "; type the request instead") -> str | None:
        try:
            with hud.calling(cfg.get("model")):
                return transcriber.transcribe(pcm_to_wav(pcm, mic.rate), seconds)
        except Exception as e:
            hud.note(f"transcription failed ({type(e).__name__}: {e}){then}")
            return None

    for t, chunk in mic.chunks():
        if muted is not None and muted(t):
            spoke = t
            heard = [pcm for pcm in [gate.reset()] if pcm]  # only a request that ended before the co-pilot spoke
            over = []
            if under is not None:
                if not under.hearing:  # judged against the room, not its voice, so that's cut into pieces too
                    under.floor = gate.floor
                over = under.feed(chunk)
        else:  # a piece started over its voice runs on to its end, so a stop said as it finished is heard whole
            over = under.feed(chunk) if under is not None and under.hearing else []
            heard = gate.feed(chunk)
            if not checked:
                listened += len(chunk) / 2 / mic.rate
        for pcm in over:
            text = transcribe(pcm, len(pcm) / 2 / mic.rate, then=" over its voice; Ctrl-C stops the arm")
            if text is None:
                continue
            if said_stop(f"{before} {text}", phrase):  # with the piece before, for a stop cut in two
                before = ""
                hud.note(f"heard over its voice: {text}")
                on_stop()
            else:
                before = text
        for pcm in heard:
            seconds = len(pcm) / 2 / mic.rate
            text = transcribe(pcm, seconds)
            if text is None:
                continue
            request = wake_request(text, phrase)
            if request is None and clock() < awake_until:
                request = text  # the request after the wake phrase alone
            if request is None and question is not None and question.answer(text, clock(), spoke):
                request = text  # the answer to what the co-pilot just asked
            if on_stop is not None and (is_stop(request) if request is not None else bare_stop(text)):
                awake_until = 0.0
                hud.note(f"heard ({seconds:.1f} s): {text}")
                on_stop()
            elif request is None:
                hud.note(f"heard, not for me ({seconds:.1f} s): {text or '(no words)'}")
            elif not request:
                awake_until = clock() + AWAKE_S
                hud.note("yes? listening for your request")
            else:
                awake_until = 0.0
                if question is not None:
                    question.close()
                hud.note(f"heard ({seconds:.1f} s): {request}")
                hud.heard(request)
                on_text(request)
        if gate.hearing != hearing:  # after any transcription, so an utterance counts as heard until it's read
            hearing = gate.hearing
            hud.hearing(hearing)
        if not checked and listened >= 2.0:
            checked = True
            if gate.floor == 0:  # a mic macOS hasn't allowed records exact zeros
                hud.note("the mic is sending pure silence: macOS hasn't let the app this terminal runs in (Claude, "
                         "Terminal, ...) use the microphone. Switch it on in System Settings → Privacy & Security → "
                         "Microphone, then quit and reopen that app. If the app isn't listed there, run this from "
                         "Terminal.app, which asks for the mic on its first run.")
            else:
                hud.note(f"room noise {dbfs(gate.floor):.0f} dBFS; speech counts above {dbfs(gate.threshold()):.0f} dBFS")
