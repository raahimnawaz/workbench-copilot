"""Voice front end: hold the pedal, talk, let go. The transcript becomes the router's next turn.

v1 records while the pedal is down and uploads one WAV to Muse Voice Transcribe when you let go.
Streaming partial transcripts over the realtime socket is the v2 upgrade (sheet 08).

The upload's endpoint, form fields and response shape come from a third-party guide. They're
marked VERIFY: check them against Meta's speech docs and run scripts/smoke_test.py.
"""
from __future__ import annotations

import io
import threading
import wave


def pcm_to_wav(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit, the only format the file endpoint accepts
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def parse_transcript(payload: dict) -> str:
    """Accept the likely response shapes until the real one is confirmed."""  # VERIFY
    for key in ("text", "transcript"):
        if isinstance(payload.get(key), str):
            return payload[key].strip()
    for key in ("turns", "segments", "results"):
        items = payload.get(key)
        if isinstance(items, list):
            return " ".join(str(i.get("text", "")).strip() for i in items if isinstance(i, dict)).strip()
    raise ValueError(f"unrecognized transcript response: {list(payload)}")


class Transcriber:
    def __init__(self, cfg: dict, api_key: str, budget, http=None):
        import httpx

        self.cfg = cfg
        self.api_key = api_key
        self.budget = budget
        self.http = http or httpx.Client(timeout=30)

    def transcribe(self, wav: bytes, seconds: float) -> str:
        self.budget.check()
        r = self.http.post(
            self.cfg["url"],
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={"file": ("turn.wav", wav, "audio/wav")},                                    # VERIFY
            data={"model": self.cfg["model"], "keywords": ",".join(self.cfg.get("keywords", []))},  # VERIFY
        )
        r.raise_for_status()
        self.budget.add_audio(self.cfg["model"], seconds)
        return parse_transcript(r.json())


class Recorder:
    """Mono 16-bit microphone capture. The Mac Studio has no built-in mic, so plug one in."""

    def __init__(self, rate: int = 16000):
        try:
            import sounddevice
        except ImportError:
            raise RuntimeError("Voice needs the extras: pip install -e '.[voice]'") from None
        self.rate = rate
        self._chunks = []
        self._stream = sounddevice.RawInputStream(samplerate=rate, channels=1, dtype="int16",
                                                  callback=lambda data, frames, t, status: self._chunks.append(bytes(data)))

    def start(self) -> None:
        self._chunks = []
        self._stream.start()

    def stop(self) -> tuple:
        self._stream.stop()
        pcm = b"".join(self._chunks)
        return pcm, len(pcm) / 2 / self.rate


class Pedal:
    """A USB foot pedal is just a keyboard key. macOS asks for Input Monitoring permission the first time."""

    def __init__(self, key_name: str):
        try:
            from pynput import keyboard
        except ImportError:
            raise RuntimeError("Voice needs the extras: pip install -e '.[voice]'") from None
        self._key = getattr(keyboard.Key, key_name, None) or keyboard.KeyCode.from_char(key_name)
        self.down = threading.Event()
        self.up = threading.Event()
        self.up.set()
        listener = keyboard.Listener(on_press=self._press, on_release=self._release)
        listener.daemon = True
        listener.start()

    def _press(self, key) -> None:
        if key == self._key and not self.down.is_set():
            self.up.clear()
            self.down.set()

    def _release(self, key) -> None:
        if key == self._key:
            self.down.clear()
            self.up.set()


def listen_forever(on_text, cfg: dict, api_key: str, budget, hud) -> None:
    """Loop: wait for the pedal, record until it's released, transcribe, hand the text on."""
    pedal = Pedal(cfg.get("pedal_key", "f13"))
    recorder = Recorder(cfg.get("sample_rate", 16000))
    transcriber = Transcriber(cfg, api_key, budget)
    hud.note(f"voice ready: hold {cfg.get('pedal_key', 'f13')} to talk")
    while True:
        pedal.down.wait()
        recorder.start()
        hud.note("listening…")
        pedal.up.wait()
        pcm, seconds = recorder.stop()
        if seconds < 0.3:
            continue  # a tap, not a sentence
        try:
            text = transcriber.transcribe(pcm_to_wav(pcm, recorder.rate), seconds)
        except Exception as e:
            hud.note(f"transcription failed ({type(e).__name__}: {e}); type the request instead")
            continue
        if text:
            hud.note(f"heard: {text}")
            on_text(text)
