import io
import wave
from types import SimpleNamespace as NS

import pytest

from wbc.agent.budget import Budget
from wbc.agent.ears import Transcriber, parse_transcript, pcm_to_wav

CFG = {"model": "ears", "url": "https://example.invalid/asr", "keywords": ["GPIO", "SPI"]}


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
    http = FakeHttp({"text": "zero response on SPI"})
    b = Budget(10, {"ears": {"per_hour": 0.18}})
    t = Transcriber(CFG, "key", b, http=http)
    assert t.transcribe(b"RIFF...", seconds=3600) == "zero response on SPI"
    url, kwargs = http.sent
    assert kwargs["headers"]["Authorization"] == "Bearer key"
    assert kwargs["data"] == {"model": "ears", "keywords": "GPIO,SPI"}
    assert b.spent == pytest.approx(0.18)
