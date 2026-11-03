import pytest
from fakes import SilentHud

from wbc.agent.budget import Budget
from wbc.agent.tools import Bench, build_toolbox
from wbc.config import ROOT, load_yaml
from wbc.perception import schematic

NETLIST = """(export (version "E")
	(components
		(comp (ref "R1") (value "330") (footprint ""))
		(comp (ref "U1") (value "ESP32-WROOM-32") (footprint "")))
	(nets
		(net
			(code "1")
			(name "/GPIO4")
			(class "Default")
			(node
				(ref "R1")
				(pin "1")
				(pintype "passive")
			)
			(node
				(ref "U1")
				(pin "26")
				(pinfunction "IO4_26")
				(pintype "bidirectional")
			)
		)
		(net
			(code "2")
			(name "unconnected-(U1-IO5-Pad29)")
			(class "Default")
			(node
				(ref "U1")
				(pin "29")
				(pinfunction "IO5_29")
				(pintype "bidirectional+no_connect")
			)
		)
	)
)
"""


def test_a_netlist_gives_each_nets_pins_and_each_parts_value():
    nets, values = schematic.parse_netlist(NETLIST)
    assert nets == [("GPIO4", [("R1", "1", ""), ("U1", "26", "IO4")]), ("unconnected-(U1-IO5-Pad29)", [("U1", "29", "IO5")])]
    assert values == {"R1": "330", "U1": "ESP32-WROOM-32"}


def toolbox(layout):
    models = load_yaml("models.yaml")
    bench = Bench(ROOT, models, None, Budget(10, models["budget"]["prices"]), sim=False, say=SilentHud().note,
                  layout=layout, parts={}, robot_cfg={}, desk_cfg={})
    return build_toolbox(bench)


def test_read_schematic_says_when_bench_yaml_names_none():
    assert toolbox({}).dispatch("read_schematic", {}) == {"ok": False, "error": "bench.yaml names no schematic"}


@pytest.mark.skipif(schematic.kicad_cli() is None, reason="needs KiCad 10's kicad-cli")
def test_the_led_boards_schematic_runs_gpio_4_through_the_resistor_to_the_led():
    read = toolbox(load_yaml("bench.yaml")).dispatch("read_schematic", {})
    assert read["ok"] and read["erc"] == {"errors": 0}
    nets = {n["net"]: n["connects"] for n in read["nets"]}
    assert nets["GPIO4"] == ["R1 pin 1 (330)", "U1 IO4"] and nets["LED_A"] == ["D1 A", "R1 pin 2 (330)"]
    assert nets["GND"] == ["D1 K", "U1 GND"] and "U1 VDD" in nets["+3V3"]


@pytest.mark.skipif(schematic.kicad_cli() is None, reason="needs KiCad 10's kicad-cli")
def test_the_sensor_boards_schematic_puts_gpio_4_between_the_sensor_and_its_resistor():
    read = schematic.read(ROOT / "hardware" / "demo_sensor.kicad_sch")
    assert read["erc"] == {"errors": 0}
    nets = {n["net"]: n["connects"] for n in read["nets"]}
    assert nets["GPIO4"] == ["R1 pin 2 (TFS-100 force sensor)", "R2 pin 1 (10k)", "U1 IO4"]
    assert "U1 GND" in nets["GND"] and "U1 VDD" in nets["+3V3"]
