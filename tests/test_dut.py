import shutil

import pytest

from wbc.agent import dut
from wbc.agent.budget import Budget
from wbc.agent.tools import Bench, build_toolbox
from wbc.config import ROOT, load_yaml

PINS = "firmware/dut_demo/include/pins.h"


class FakePort:
    """The demo board's serial port: hands out the given lines, then nothing."""

    def __init__(self, lines):
        self.lines = [line.encode() for line in lines]
        self.closed = False

    def readline(self):
        return self.lines.pop(0) if self.lines else b""

    def close(self):
        self.closed = True


def test_parse_reads_millivolts_and_the_firmware_pin():
    mv, gpio = dut.parse(["# reading GPIO 15\n", "mv 18\n", "garbage\n", "mv 22\r\n"])
    assert mv == [18, 22] and gpio == 15


def test_a_press_shows_as_a_response():
    flat = dut.summarize([20, 22, 19, 21] * 30, 15, 2.0)
    pressed = dut.summarize([140, 900, 1900, 1880, 300] * 20, 4, 2.0)
    assert flat["ok"] and not flat["responds"] and flat["mv"]["max"] == 22
    assert pressed["responds"] and pressed["firmware_gpio"] == 4 and len(pressed["trace_mv"]) == dut.TRACE_POINTS
    assert dut.summarize([], None, 2.0)["ok"] is False


def test_read_summarizes_the_lines_and_closes_the_port():
    port = FakePort(["# reading GPIO 4\n"] + [f"mv {v}\n" for v in (140, 1500, 1900, 160)])
    out = dut.read("/dev/cu.fake", seconds=0.05, opener=lambda p, b: port)
    assert out["ok"] and out["responds"] and out["firmware_gpio"] == 4 and out["port"] == "/dev/cu.fake"
    assert port.closed


def test_find_port_skips_the_arm(monkeypatch):
    found = {"/dev/cu.usb*": ["/dev/cu.usbmodem5B910297421", "/dev/cu.usbserial-110"], "/dev/cu.wchusbserial*": []}
    monkeypatch.setattr(dut.glob, "glob", lambda pattern: found[pattern])
    cfg = {"port": "auto", "skip_ports": ["/dev/cu.usbmodem5B910297421"]}
    assert dut.find_port(cfg) == "/dev/cu.usbserial-110"
    assert dut.find_port({"port": "/dev/cu.x"}) == "/dev/cu.x"
    found["/dev/cu.usb*"].append("/dev/cu.usbserial-120")
    with pytest.raises(RuntimeError, match="2 possible ports"):
        dut.find_port(cfg)


def test_flash_runs_the_upload_command_on_the_port(tmp_path):
    out = dut.flash(tmp_path, "echo upload to {port}", "/dev/cu.fake")
    assert out["ok"] and "upload to /dev/cu.fake" in out["log"]
    assert dut.flash(tmp_path, "false {port}", "/dev/cu.fake")["error"] == "the build or upload failed"
    assert "not on PATH" in dut.flash(tmp_path, "no-such-tool-here {port}", "/dev/cu.fake")["error"]


def test_sim_sensor_answers_only_after_the_fix_is_flashed(tmp_path):
    shutil.copytree(ROOT / "firmware" / "dut_demo", tmp_path / "firmware" / "dut_demo",
                    ignore=shutil.ignore_patterns(".pio"))
    said = []
    tb = build_toolbox(Bench(tmp_path, load_yaml("models.yaml"), None, Budget(10, {}), sim=True, say=said.append))
    before = tb.dispatch("read_sensor", {})
    assert before["ok"] and not before["responds"] and before["firmware_gpio"] == 15
    pins = tmp_path / PINS
    pins.write_text(pins.read_text().replace("#define SENSOR_PIN 15 ", "#define SENSOR_PIN 4  "))
    assert tb.dispatch("build_and_flash", {})["firmware_gpio"] == 4
    after = tb.dispatch("read_sensor", {"prompt": "Press the sensor now."})
    assert after["responds"] and after["firmware_gpio"] == 4 and said == ["Press the sensor now."]


def test_the_firmware_pin_is_read_from_pins_h():
    assert dut.firmware_gpio(ROOT) == 15
