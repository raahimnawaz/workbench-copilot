"""Voice front end, hands free: say "Copilot," and your request. The request becomes the router's next turn.

The mic stays open while voice is on. A loudness gate cuts the stream into utterances, Muse Voice Transcribe
transcribes each one, and only those that start with the wake phrase ("Copilot, check my wiring") go to the
router. The wake phrase alone ("Copilot.") makes the next utterance the request. Streaming partial transcripts
over the realtime socket is the v2 upgrade (sheet 08).

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

import numpy as np

FRAME_S = 0.03  # the loudness gate works on 30 ms frames
FILLERS = ("hey", "ok", "okay", "um", "uh", "so", "and")  # allowed before the wake phrase
AWAKE_S = 8.0  # after the wake phrase alone, how long the next utterance counts as the request


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


def wake_request(text: str, phrase: str) -> str | None:
    """What follows the wake phrase at the start of `text` ("" for the phrase alone), or None if it isn't there.

    The phrase may come split or hyphenated ("co-pilot", "Co pilot") and after a filler ("Hey, copilot").
    """
    letters = r"[\W_]*".join(re.escape(c) for c in phrase.lower() if c.isalnum())
    m = re.match(rf"\W*(?:(?:{'|'.join(FILLERS)})\W+)?{letters}\b\W*", text, flags=re.IGNORECASE)
    return text[m.end():].strip() if m else None


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
        import httpx

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
    """The bench mic as a stream of 16-bit mono chunks. The Mac Studio has no built-in mic, so plug one in."""

    def __init__(self, rate: int = 16000, device: str | None = None):
        try:
            import sounddevice
        except ImportError:
            raise RuntimeError("Voice needs the extras: pip install -e '.[voice]'") from None
        self.rate = rate
        self._chunks = queue.Queue()
        # sounddevice matches `device` against part of the name, so "Razer Seiren Mini" finds the Razer.
        self._stream = sounddevice.RawInputStream(samplerate=rate, channels=1, dtype="int16", device=device,
                                                  callback=lambda data, frames, t, status: self._chunks.put(bytes(data)))
        self.name = sounddevice.query_devices(self._stream.device)["name"]

    def chunks(self):
        self._stream.start()
        while True:
            yield self._chunks.get()


def listen_forever(on_text, cfg: dict, api_key: str, budget, hud, mic=None, transcriber=None,
                   clock=time.monotonic) -> None:
    """Loop: cut the mic stream into utterances, transcribe each, and hand on the ones addressed to the co-pilot."""
    try:
        mic = mic or Microphone(cfg.get("sample_rate", 16000), cfg.get("input_device"))
    except Exception as e:  # no voice extras, no such mic, or PortAudio refusing the stream
        hud.note(f"voice off: {e}")
        return
    transcriber = transcriber or Transcriber(cfg, api_key, budget)
    phrase = cfg.get("wake_phrase", "copilot")
    gate = Segmenter(mic.rate)
    hud.note(f'voice on {mic.name}: say "{phrase.capitalize()}," and your request')
    listened, checked, awake_until = 0.0, False, 0.0
    for chunk in mic.chunks():
        if not checked:
            listened += len(chunk) / 2 / mic.rate
        for pcm in gate.feed(chunk):
            seconds = len(pcm) / 2 / mic.rate
            try:
                text = transcriber.transcribe(pcm_to_wav(pcm, mic.rate), seconds)
            except Exception as e:
                hud.note(f"transcription failed ({type(e).__name__}: {e}); type the request instead")
                continue
            request = wake_request(text, phrase)
            if request is None and clock() < awake_until:
                request = text  # the request after the wake phrase alone
            if request is None:
                hud.note(f"heard, not for me ({seconds:.1f} s): {text or '(no words)'}")
            elif not request:
                awake_until = clock() + AWAKE_S
                hud.note("yes? listening for your request")
            else:
                awake_until = 0.0
                hud.note(f"heard ({seconds:.1f} s): {request}")
                on_text(request)
        if not checked and listened >= 2.0:
            checked = True
            if gate.floor == 0:  # a mic macOS hasn't allowed records exact zeros
                hud.note("the mic is sending pure silence. Let this terminal use the microphone in System Settings → "
                         "Privacy & Security → Microphone, then restart the terminal.")
            else:
                hud.note(f"room noise {dbfs(gate.floor):.0f} dBFS; speech counts above {dbfs(gate.threshold()):.0f} dBFS")
