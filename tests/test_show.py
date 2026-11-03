"""--show's windows, with Chrome, KiCad and the shell faked: nothing opens on the screen."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from wbc.agent import show


def test_the_hud_opens_as_a_chrome_app_window_in_a_profile_of_its_own(tmp_path):
    chrome, profile, started = tmp_path / "Google Chrome", tmp_path / "hud-window", []
    chrome.touch()
    window = show.hud_window("http://127.0.0.1:8766/", profile, popen=lambda args, **kw: started.append(args) or "proc",
                             chrome=chrome)
    assert window == "proc" and profile.is_dir()
    args = started[0]
    assert args[0] == str(chrome) and "--app=http://127.0.0.1:8766/" in args  # no tabs or address bar
    assert f"--user-data-dir={profile}" in args  # its own Chrome, whatever else is open
    assert "--window-size=1000,760" in args  # sized the first time only
    show.hud_window("http://127.0.0.1:8766/", profile, popen=lambda args, **kw: started.append(args), chrome=chrome)
    assert not any(a.startswith("--window-size") for a in started[1])  # then Chrome keeps where it was put


def test_no_chrome_no_window(tmp_path):
    assert show.hud_window("http://127.0.0.1:8766/", tmp_path / "p", popen=None, chrome=tmp_path / "none") is None
    assert not (tmp_path / "p").exists()


def fake_run(running: bool, calls: list):
    def run(args, **kw):
        calls.append(args)
        return SimpleNamespace(returncode=0 if (args[0] == "pgrep" and running) else (1 if args[0] == "pgrep" else 0))
    return run


def test_the_schematic_opens_in_kicads_schematic_editor(tmp_path):
    editor, drawn, calls = tmp_path / "Schematic Editor.app", tmp_path / "demo_led.kicad_sch", []
    editor.mkdir()
    drawn.write_text("(kicad_sch)")
    said = show.schematic(drawn, run=fake_run(False, calls), editor=editor)
    assert calls[-1] == ["open", "-a", str(editor), str(drawn)] and said == "schematic: demo_led.kicad_sch opened in KiCad"


def test_the_schematic_isnt_opened_twice(tmp_path):
    editor, drawn, calls = tmp_path / "Schematic Editor.app", tmp_path / "demo_led.kicad_sch", []
    editor.mkdir()
    drawn.write_text("(kicad_sch)")
    said = show.schematic(drawn, run=fake_run(True, calls), editor=editor)
    assert [c[0] for c in calls] == ["pgrep"] and "open already" in said


def test_without_kicad_or_the_file_it_says_so_and_opens_nothing(tmp_path):
    calls = []
    assert "isn't in" in show.schematic(tmp_path / "x.kicad_sch", run=fake_run(False, calls), editor=tmp_path / "no")
    editor = tmp_path / "Schematic Editor.app"
    editor.mkdir()
    assert "there's no" in show.schematic(tmp_path / "x.kicad_sch", run=fake_run(False, calls), editor=editor)
    assert calls == []


def test_the_agent_takes_show(capsys):
    from wbc.agent.__main__ import main
    with pytest.raises(SystemExit) as e:
        main(["--show", "--help"])
    assert e.value.code == 0 and "--show" in capsys.readouterr().out
