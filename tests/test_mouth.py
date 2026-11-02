import io
import subprocess
import threading
from types import SimpleNamespace as NS

import pytest
from fakes import SilentHud

from wbc.agent import __main__ as agent
from wbc.agent.mouth import Mouth, spoken


@pytest.mark.parametrize("reply, said", [
    ("**Fixed.** The sensor is wired to `GPIO 4`, not 15. I staged the one-line patch.",
     "Fixed. The sensor is wired to GPIO 4, not 15."),
    ("The patch is staged:\n```cpp\n#define SENSOR_PIN 4\n```\nSay go to flash it.",
     "The patch is staged: Say go to flash it."),
    ("## Wiring\n- Row 26 goes to GPIO 4.\n- pins.h says 15.", "Wiring Row 26 goes to GPIO 4. pins.h says 15."),
    ("SENSOR_PIN in [pins.h](firmware/dut_demo/pins.h) is 15.", "SENSOR_PIN in pins.h is 15."),
    ("Readings rose from 0.14 V to 2.9 V when you pressed. It's fixed!",
     "Readings rose from 0.14 V to 2.9 V when you pressed. It's fixed!"),
    ("Grip at approx. 2 N, e.g. like an egg. Then lift it. Done.", "Grip at approx. 2 N, e.g. like an egg. Then lift it."),
    ("The fix is staged. pins.h now says 4. Flash it when you're ready.", "The fix is staged. pins.h now says 4."),
    ('You said "flash it." So I flashed it. It reads 2.9 V.', 'You said "flash it." So I flashed it.'),
    ("", ""),
])
def test_says_the_first_two_sentences_as_plain_words(reply, said):
    assert spoken(reply) == said


def test_a_long_sentence_is_cut_at_a_word_and_says_so():
    said = spoken("The arm " + "moves slowly and " * 30 + "stops.", limit=60)
    assert said.startswith("The arm moves slowly") and said.endswith(" and. The rest is on the screen.")


def test_a_second_sentence_that_doesnt_fit_is_left_whole_for_the_hud():
    reply = ("The sensor stayed at 0.1 V when you pressed, and the wire in row 26 runs to GPIO 4 while pins.h names "
             "GPIO 15, so that's the wiring mismatch. The patch that sets SENSOR_PIN to 4 is staged, and it is safe to "
             "flash it now unless the jumper moved off row 26.")
    assert spoken(reply).endswith("so that's the wiring mismatch.")


class Speech:
    def __init__(self, args, kwargs):
        self.args, self.kwargs, self.terminated = args, kwargs, False
        self.over = threading.Event()

    def poll(self):
        return 0 if self.over.is_set() else None

    def wait(self):
        self.over.wait()

    def finish(self):
        self.over.set()

    def terminate(self):
        self.terminated = True
        self.over.set()


class FakeSay:
    """Stands in for subprocess.Popen, so no test makes a sound."""

    def __init__(self):
        self.runs = []

    def __call__(self, args, **kwargs):
        self.runs.append(Speech(args, kwargs))
        return self.runs[-1]


def test_speaks_with_the_configured_voice_and_never_reads_text_as_an_option():
    say = FakeSay()
    Mouth({"voice": "Samantha", "rate": 190}, command="/usr/bin/say", popen=say).speak("-v is a flag. Second. Third.")
    assert say.runs[0].args == ["/usr/bin/say", "-v", "Samantha", "-r", "190", "--", "-v is a flag. Second."]
    assert say.runs[0].kwargs["stdin"] is subprocess.DEVNULL


def test_says_nothing_for_an_empty_reply_or_without_say(monkeypatch):
    say = FakeSay()
    Mouth({}, command="/usr/bin/say", popen=say).speak("")
    monkeypatch.setattr("wbc.agent.mouth.shutil.which", lambda name: None)  # not a Mac
    mouth = Mouth({}, popen=say)
    mouth.speak("Fixed.")
    assert mouth.command is None and say.runs == []


def test_busy_while_speaking_and_for_the_tail_after():
    now = [0.0]
    say = FakeSay()
    mouth = Mouth({"tail_s": 0.4}, command="say", popen=say, clock=lambda: now[0])
    assert not mouth.busy()
    mouth.speak("Fixed.")
    now[0] = 3.0
    assert mouth.busy()
    say.runs[0].finish()
    mouth.waiter.join(1)
    now[0] = 3.3
    assert mouth.busy()
    now[0] = 3.5
    assert not mouth.busy()


def test_sound_is_judged_by_when_it_was_recorded():
    """The ears fall behind while a transcription is out; a request said after the reply must still count."""
    now = [6.3]
    say = FakeSay()
    mouth = Mouth({"tail_s": 0.4}, command="say", popen=say, clock=lambda: now[0])
    mouth.speak("Fixed.")
    now[0] = 7.0
    say.runs[0].finish()
    mouth.waiter.join(1)
    now[0] = 9.0  # the ears get to the backlog
    assert [mouth.busy(t) for t in (6.0, 6.5, 7.2, 7.5)] == [False, True, True, False]


def test_a_new_reply_cuts_off_the_last_one():
    say = FakeSay()
    mouth = Mouth({}, command="say", popen=say, clock=lambda: 0.0)
    mouth.speak("First reply.")
    mouth.speak("Second reply.")
    first, second = say.runs
    assert first.terminated and second.args[-1] == "Second reply." and mouth.busy()
    mouth.stop()
    assert second.terminated


def test_a_say_that_wont_run_leaves_replies_on_the_hud():
    def broken(args, **kwargs):
        raise FileNotFoundError(args[0])

    mouth = Mouth({}, command="/usr/bin/say", popen=broken)
    mouth.speak("Fixed.")
    mouth.speak("Still fixed.")
    assert mouth.command is None and not mouth.busy()


@pytest.mark.parametrize("flags, said, ears", [
    (["--voice"], ["Fixed. The patch is staged."], True),
    (["--speak"], ["Fixed. The patch is staged."], False),
    (["--voice", "--no-speak"], [], True),
    (["--hear"], [], True),
    ([], [], False),
])
def test_flags_choose_when_it_speaks_and_the_mic_waits_for_it(monkeypatch, flags, said, ears):
    say, heard, started = FakeSay(), {}, threading.Event()
    budget = NS(spent=0.0, limit=5.0, check=lambda: None, pop_warning=lambda: None)
    router = NS(label="Spark", model="spark", run_turn=lambda text: "Fixed. The patch is staged. Say go to flash it.")
    bench = NS(park=lambda: None, stop_arm=lambda: None, hand=None)
    monkeypatch.setattr(agent, "build", lambda *a: ({"ears": {}, "coder": {}, "mouth": {}}, budget, SilentHud(), router,
                                                    bench, NS(offer=lambda line: False)))
    monkeypatch.setattr(agent, "Mouth", lambda cfg: Mouth(cfg, command="say", popen=say))
    monkeypatch.setattr(agent.coder, "compile_in_sandbox", lambda *a: None)
    monkeypatch.setattr(agent.config, "api_key", lambda: "key")
    monkeypatch.setattr(agent.ears, "listen_forever", lambda *a, **kw: (heard.update(kw), started.set()))
    monkeypatch.setattr("sys.stdin", io.StringIO("check my wiring\n"))
    assert agent.main(flags) == 0
    assert [r.args[-1] for r in say.runs] == said
    assert started.wait(2 if ears else 0.1) == ears  # the ears start before the first turn, or never
    if said and ears:
        assert heard["muted"]()  # the reply is still being said, so the mic drops what it hears
