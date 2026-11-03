"""Kokoro-82M on this Mac for the co-pilot's voice: a local server that keeps the model loaded and turns words into a WAV.

macOS `say`'s voices can't be in a published video (the macOS license allows them only for personal, non-commercial
use), and Meta's API has no text to speech. Kokoro-82M's weights are Apache-2.0. scripts/kokoro-say, the mouth's
stand-in for `say` (config/models.yaml's mouth.command), posts each reply's spoken part here and plays what comes back.
It runs in its own venv, since kokoro-onnx brings onnxruntime and espeak-ng:

    python -m venv <venv> && <venv>/bin/pip install kokoro-onnx==0.6.1
    <venv>/bin/python -I scripts/kokoro_server.py [--port 8767] [--models /Users/Shared/models/kokoro-v1.0]

The weights (kokoro-v1.0.onnx and voices-v1.0.bin) are the kokoro-onnx project's release model-files-v1.0. espeak-ng
keeps its data path in a 160-character buffer, so a venv deeper than that falls back to the path it was built with.
POST /say?voice=af_heart with the words as the body answers a 16-bit mono WAV; GET /voices lists the voices.
"""
from __future__ import annotations

import argparse
import http.server
import io
import threading
import time
import urllib.parse
import wave
from pathlib import Path

import numpy as np
from kokoro_onnx import Kokoro

VOICE = "af_heart"
MAX_CHARS = 600  # the mouth says two sentences at most, 240 characters


def wav_bytes(samples, rate: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())
    return out.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--models", default="/Users/Shared/models/kokoro-v1.0")
    args = ap.parse_args()
    models = Path(args.models)
    t0 = time.monotonic()
    kokoro = Kokoro(str(models / "kokoro-v1.0.onnx"), str(models / "voices-v1.0.bin"))
    kokoro.create("Ready.", voice=VOICE, speed=1.0, lang="en-us")  # the first call is the slow one
    voices = set(kokoro.get_voices())
    lock = threading.Lock()  # one at a time: a reply cut off by the next one is dropped by the mouth anyway
    print(f"Kokoro-82M loaded in {time.monotonic() - t0:.1f} s, {len(voices)} voices, on 127.0.0.1:{args.port}",
          flush=True)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/voices":
                self.send_error(404)
                return
            self._send(200, "text/plain", "\n".join(sorted(voices)).encode())

        def do_POST(self) -> None:
            url = urllib.parse.urlparse(self.path)
            if url.path != "/say":
                self.send_error(404)
                return
            voice = urllib.parse.parse_qs(url.query).get("voice", [VOICE])[0]
            words = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8", "replace")
            words = " ".join(words.split())[:MAX_CHARS]
            if voice not in voices or not words:
                self.send_error(400, f"no such voice {voice!r}" if voice not in voices else "no words")
                return
            t = time.monotonic()
            with lock:
                samples, rate = kokoro.create(words, voice=voice, speed=1.0, lang="en-us")
            body = wav_bytes(samples, rate)
            print(f"{time.monotonic() - t:.2f} s for {len(samples) / rate:.1f} s of speech: {words[:60]}", flush=True)
            self._send(200, "audio/wav", body)

        def _send(self, code: int, kind: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args) -> None:
            pass

    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
