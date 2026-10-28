import io
import json
import wave
from types import SimpleNamespace as NS

import numpy as np
import pytest
from fakes import SilentHud

from wbc.agent.budget import Budget
from wbc.agent.ears import Segmenter, Transcriber, listen_forever, parse_transcript, pcm_to_wav, wake_request

CFG = {"model": "ears", "url": "https://example.invalid/asr", "keywords": ["GPIO", "SPI"]}
RATE = 16000


def quiet(seconds, level=20):
    """Room noise: white noise with this RMS."""
    return np.random.default_rng(0).normal(0, level, int(RATE * seconds)).astype(np.int16).tobytes()


def voice(seconds, amplitude=3000):
    """A stand-in for speech: a 220 Hz tone, RMS about 2100 (-24 dBFS)."""
    t = np.arange(int(RATE * seconds)) / RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.int16).tobytes()


def cut(*pieces, **gate):
    g = Segmenter(RATE, **gate)
    return [len(u) / 2 / RATE for u in g.feed(b"".join(pieces))]


def test_wav_is_16_bit_mono():
    wav = pcm_to_wav(b"\x01\x00" * 16000, 16000)
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 16000, 16000)


@pytest.mark.parametrize("payload, text", [
    ({"text": " check my wiring "}, "check my wiring"),
    ({"transcript": "flash it"}, "flash it"),
    ({"turns": [{"text": "check"}, {"text": "the bus"}]}, "check the bus"),
])
def test_parse_transcript_shapes(payload, text):
    assert parse_transcript(payload) == text


def test_parse_transcript_unknown_shape():
    with pytest.raises(ValueError):
        parse_transcript({"words": []})


class FakeHttp:
    def __init__(self, payload):
        self.payload = payload
        self.sent = None

    def post(self, url, **kwargs):
        self.sent = (url, kwargs)
        return NS(raise_for_status=lambda: None, json=lambda: self.payload)


def test_transcriber_posts_wav_and_bills_audio():
    http = FakeHttp({"transcript": "zero response on SPI", "audioDurationMs": 3600000, "turns": []})
    b = Budget(10, {"ears": {"per_hour": 0.18}})
    t = Transcriber(CFG, "key", b, http=http)
    assert t.transcribe(b"RIFF...", seconds=3600) == "zero response on SPI"
    url, kwargs = http.sent
    assert kwargs["headers"]["Authorization"] == "Bearer key"
    name, body, mime = kwargs["files"]["request"]
    assert (name, mime) == (None, "application/json")
    assert json.loads(body) == {"model": "ears", "audioEncoding": "WAV", "keywords": ["GPIO", "SPI"]}
    assert kwargs["files"]["audio"] == ("turn.wav", b"RIFF...", "audio/wav")
    assert b.spent == pytest.approx(0.18)


@pytest.mark.parametrize("text, request_", [
    ("Copilot, check my wiring.", "check my wiring."),
    ("Hey co-pilot, flash it", "flash it"),
    ("Okay, Co pilot. What's the grip force?", "What's the grip force?"),
    ("Copilot.", ""),
    ("pass me the solder", None),
    ("Copilots are everywhere now", None),
    ("I asked the copilot already", None),
])
def test_wake_phrase(text, request_):
    assert wake_request(text, "copilot") == request_


def test_gate_cuts_one_utterance_with_its_edges():
    [seconds] = cut(quiet(1), voice(1), quiet(1.5))
    assert 1.0 + 0.8 <= seconds <= 1.0 + 0.8 + 0.3 + 0.05  # the speech, the hangover, at most the preroll


def test_gate_drops_clicks_and_thumps():
    assert cut(quiet(1), voice(0.06), quiet(1), voice(0.15), quiet(1.5)) == []


def test_gate_keeps_a_pause_inside_a_sentence():
    assert len(cut(quiet(1), voice(0.6), quiet(0.5), voice(0.6), quiet(1.5))) == 1
    assert len(cut(quiet(1), voice(0.6), quiet(1.2), voice(0.6), quiet(1.5))) == 2


def test_gate_cuts_long_speech():
    seconds = cut(quiet(1), voice(5), quiet(1.5), max_s=2.0)
    assert len(seconds) >= 2 and seconds[0] == pytest.approx(2.0, abs=0.05)


def test_gate_follows_the_room_noise():
    assert cut(quiet(3, level=400)) == []  # a servo's whine, about -38 dBFS, isn't speech
    assert len(cut(quiet(1, level=400), voice(1), quiet(1.5, level=400))) == 1


class FakeMic:
    name, rate = "Fake Mic", RATE

    def __init__(self, *pieces):
        self.audio = b"".join(pieces)

    def chunks(self):
        for i in range(0, len(self.audio), 1600):  # 50 ms, about one stream callback
            yield self.audio[i:i + 1600]


class FakeTranscriber:
    def __init__(self, *texts):
        self.texts = list(texts)
        self.seconds = []

    def transcribe(self, wav, seconds):
        self.seconds.append(seconds)
        return self.texts.pop(0)


def listen(mic, transcriber, clock=lambda: 0.0):
    hud, sent = SilentHud(), []
    listen_forever(sent.append, {"wake_phrase": "copilot"}, "key", None, hud, mic=mic, transcriber=transcriber,
                   clock=clock)
    return sent, [e[1] for e in hud.events]


def test_routes_only_what_is_said_to_the_copilot():
    mic = FakeMic(quiet(2.5), voice(1), quiet(1.5), voice(1), quiet(1.5))
    sent, notes = listen(mic, FakeTranscriber("Copilot, check my wiring.", "pass me the solder"))
    assert sent == ["check my wiring."]
    assert any(n.startswith("heard, not for me") and "solder" in n for n in notes)
    assert any(n.startswith("room noise") for n in notes)


@pytest.mark.parametrize("later, routed", [(5.0, ["flash it"]), (9.0, [])])
def test_wake_phrase_alone_waits_for_the_request(later, routed):
    mic = FakeMic(quiet(2.5), voice(0.6), quiet(1.5), voice(1), quiet(1.5))
    sent, notes = listen(mic, FakeTranscriber("Copilot.", "flash it"), clock=iter([0.0, later]).__next__)
    assert sent == routed


def test_silent_mic_is_reported_and_never_sent():
    """A mic macOS hasn't allowed records exact zeros; nothing should be paid for, and the cause should show."""
    t = FakeTranscriber()
    sent, notes = listen(FakeMic(bytes(2 * RATE * 3)), t)
    assert sent == [] and t.seconds == []
    assert any("pure silence" in n for n in notes)
