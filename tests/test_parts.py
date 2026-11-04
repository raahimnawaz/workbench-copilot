"""add_part with Spark Open scripted: its wiring checked, sent back once when wrong, the skill saved, and the screen
drawn into a copy of the schematic that KiCad's own check and netlist accept. The drawing's tests need kicad-cli."""
import json
import shutil

import pytest
from fakes import FakeChatClient, SilentHud, chat_response

from wbc.agent import parts
from wbc.agent.budget import Budget
from wbc.agent.tools import Bench, build_toolbox
from wbc.config import ROOT, load_yaml
from wbc.perception import schematic

OPEN = "meta-models/muse-spark-open-1026"
SCREEN = load_yaml("parts.yaml")["modules"]["ssd1306_spi_7"]
GOOD = {"wiring": {"GND": "GND", "VCC": "3V3", "D0": "GPIO18", "D1": "GPIO23", "RES": "GPIO16", "DC": "GPIO17",
                   "CS": "GPIO5"},
        "nets": {"D0": "OLED_SCK", "D1": "OLED_MOSI", "RES": "OLED_RES", "DC": "OLED_DC", "CS": "OLED_CS"},
        "library": "olikraus/U8g2", "setup": "U8G2_SSD1306_128X64_NONAME_F_4W_HW_SPI u8g2(U8G2_R0, 5, 17, 16);",
        "notes": "D0 is the SPI clock and D1 its data.", "say": "Clock on GPIO18, data on 23, chip select on 5."}
FREE = [5, 13, 14, 16, 17, 18, 19, 21, 22, 23, 25, 26, 27, 32, 33]
USED = {4: "net GPIO4 in the schematic", 15: "SENSOR_PIN in pins.h"}


def budget():
    return Budget(10, load_yaml("models.yaml")["budget"]["prices"])


def test_a_wiring_on_free_pins_checks_out():
    wiring, problems = parts.check(GOOD, SCREEN["pins"], FREE, USED)
    assert problems == [] and wiring == GOOD["wiring"]


@pytest.mark.parametrize("pin, to, problem", [
    ("CS", "GPIO15", "GPIO15 is already used (SENSOR_PIN in pins.h)"),
    ("CS", "GPIO34", "GPIO34 can only read, so it can't drive CS"),
    ("CS", "GPIO2", "GPIO2 is a boot pin"),
    ("CS", "GPIO18", "GPIO18 already goes to D0"),
    ("VCC", "GPIO5", "VCC is a power pin, so it goes to 3V3, not GPIO5"),
    ("CS", "pin 9", "CS goes to 'pin 9', which isn't GND, 3V3 or a GPIO")])
def test_wiring_the_board_cant_give_is_sent_back(pin, to, problem):
    assert problem in parts.check({"wiring": {**GOOD["wiring"], pin: to}}, SCREEN["pins"], FREE, USED)[1]


def test_pins_are_read_the_ways_people_write_them():
    assert {parts.target(s) for s in ("IO18", "18", "D18", "gpio 18")} == {"GPIO18"}
    assert parts.target("3.3V") == "3V3" and parts.target("ground") == "GND" and parts.target("SCK") is None


def test_the_answer_is_found_after_the_reasoning():
    assert parts.parse("<think>{not json}</think>Here: " + json.dumps(GOOD) + " done") == GOOD
    assert parts.parse("no json at all") is None


def test_a_wrong_answer_goes_back_once_with_whats_wrong():
    bad = {**GOOD, "wiring": {**GOOD["wiring"], "CS": "GPIO15"}}
    client = FakeChatClient([chat_response(json.dumps(bad)), chat_response(json.dumps(GOOD))])
    wiring, _, problems = parts.learn(client, OPEN, budget(), "ssd1306_spi_7", SCREEN, "", "esp32", FREE, USED)
    assert problems == [] and wiring["CS"] == "GPIO5"
    first, second = client.chat.completions.requests
    assert "Free GPIOs: GPIO5, GPIO13," in first["messages"][1]["content"]
    assert "GPIO15: SENSOR_PIN in pins.h" in first["messages"][1]["content"]
    assert "GPIO15 is already used" in second["messages"][-1]["content"]


def test_two_wrong_answers_end_it():
    client = FakeChatClient([chat_response("no idea"), chat_response("still none")])
    problems = parts.learn(client, OPEN, budget(), "ssd1306_spi_7", SCREEN, "", "esp32", FREE, USED)[2]
    assert problems == ["the answer wasn't one JSON object"]


def bench(tmp_path, client):
    (tmp_path / "hardware").mkdir()
    shutil.copy(ROOT / "hardware" / "demo_sensor.kicad_sch", tmp_path / "hardware")
    (tmp_path / "pins.h").write_text("#define SENSOR_PIN 15\n")
    models = load_yaml("models.yaml")
    b = Bench(tmp_path, models, None, Budget(10, models["budget"]["prices"]), say=SilentHud().note,
              layout={"schematic": "hardware/demo_sensor.kicad_sch", "firmware": {"file": "pins.h"},
                      "modules": {"esp32": {"part": "esp32_devkit_30"}}},
              parts=load_yaml("parts.yaml"))
    b.open = (client, OPEN, "Spark Open") if client else None
    return b


def test_without_spark_open_it_says_so(tmp_path):
    result = build_toolbox(bench(tmp_path, None)).dispatch("add_part", {"part": "ssd1306_spi_7"})
    assert not result["ok"] and "BASETEN_API_KEY" in result["error"]


def test_the_tool_names_the_parts_it_knows(tmp_path):
    tool = next(t for t in build_toolbox(bench(tmp_path, None)).schemas() if t["name"] == "add_part")
    assert "ssd1306_spi_7 (SSD1306 OLED screen" in tool["description"]


@pytest.mark.skipif(schematic.kicad_cli() is None, reason="needs KiCad 10's kicad-cli")
def test_the_screen_is_drawn_into_a_copy_of_the_schematic_that_kicad_accepts(tmp_path):
    shown = []
    b = bench(tmp_path, FakeChatClient([chat_response("<think>pins…</think>" + json.dumps(GOOD))]))
    b.show = lambda path: shown.append(path) or "schematic: opened in KiCad"
    b.show_file = lambda path: shown.append(path) or "VS Code: opened"  # the new skill, as it's saved
    original = (tmp_path / "hardware" / "demo_sensor.kicad_sch").read_text()
    result = build_toolbox(b).dispatch("add_part", {"part": "ssd1306_spi_7", "use": "show the force reading"})
    assert result["ok"], result
    assert result["schematic"] == ".wbc/schematics/demo_sensor+ssd1306_spi_7.kicad_sch" == b.layout["schematic"]
    assert result["erc"] == {"errors": 0} and shown == [tmp_path / result["skill"], tmp_path / result["schematic"]]
    assert result["learned"].startswith("Spark Open learned a new skill for the SSD1306 OLED screen")
    nets = {n["net"]: n["connects"] for n in schematic.read(tmp_path / result["schematic"])["nets"]}
    assert nets["OLED_CS"] == ["U1 IO5", "U2 CS"] and nets["OLED_SCK"] == ["U1 IO18", "U2 D0"]
    assert "U2 GND" in nets["GND"] and "U2 VCC" in nets["+3V3"]
    assert nets["GPIO4"] == ["R1 pin 2 (TFS-100 force sensor)", "R2 pin 1 (10k)", "U1 IO4"]  # the rest as it was
    assert (tmp_path / "hardware" / "demo_sensor.kicad_sch").read_text() == original
    skill = (tmp_path / result["skill"]).read_text()
    assert "| CS | GPIO5 |" in skill and "olikraus/U8g2" in skill and "U8G2_SSD1306" in skill
