"""The demo boards' schematics as KiCad 10 files, drawn from KiCad's own symbols: the LED board or the sensor board.

    python scripts/make_schematic.py [led|sensor] [OUT.kicad_sch]   # led, to hardware/demo_<board>.kicad_sch

ESP32-WROOM-32 (the DOIT DevKit's module) as U1, its 3V3 and GND, EN tied high, and every other pin marked unused.
led: GPIO 4 -> 330 ohm -> LED anode, cathode to GND (the LED board: jumper a5-a26, resistor b26-b42, LED 42-43).
sensor: 3V3 -> TFS-100 -> row 26 (GPIO 4) -> 10 kohm -> GND. GPIO 4 runs to its part by a wire, named by a label on
it, and each part's text sits where KiCad's own library puts it. Each item's id comes from the board and its place in
the drawing, so the same script always draws the same file.
"""
import itertools
import re
import sys
import uuid
from pathlib import Path

from wbc.perception.schematic import HIDDEN, library_symbol, sexpr_block, turned

HARDWARE = Path(__file__).resolve().parent.parent / "hardware"


def pins(block: str) -> dict:
    """number -> (x, y, name) in symbol space, from a library symbol's text."""
    out = {}
    for m in re.finditer(r'\(pin \w+ \w+\s*\(at ([-\d.]+) ([-\d.]+) [-\d.]+\).*?\(name "([^"]*)".*?\(number "([^"]*)"',
                         block, re.S):
        x, y, name, number = float(m.group(1)), float(m.group(2)), m.group(3), m.group(4)
        out.setdefault(number, (x, y, name))
    return out


def fields(block: str) -> dict:
    """Reference and Value -> (x, y, angle, justify) in symbol space, where a library symbol puts them."""
    out = {}
    for name in ("Reference", "Value"):
        m = re.search(r'\(property "' + name + r'" "[^"]*"\s*\(at ([-\d.]+) ([-\d.]+)(?: ([-\d.]+))?\)', block)
        j = re.search(r'\(justify ([^)]*)\)', sexpr_block(block, m.start()))
        out[name] = (float(m.group(1)), float(m.group(2)), float(m.group(3) or 0), j.group(1) if j else "")
    return out


def effects(hidden: bool, justify: str = "") -> str:
    return ("(effects (font (size 1.27 1.27))" + (f" (justify {justify})" if justify else "")
            + (" (hide yes)" if hidden else "") + ")")


class Sheet:
    def __init__(self, board: str):
        self.board, self._ids = board, itertools.count()
        self.libs, self.items, self.pins = {}, [], {}
        self.root = self.u()

    def u(self) -> str:
        """The next item's id, from the board and the item's place in the drawing."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"workbench-copilot/demo_{self.board}/{next(self._ids)}"))

    def place(self, lib, name, ref, value, x, y, angle=0, text_at=None):
        """`text_at`: where its Reference and Value go, each (x, y, angle, justify) on the sheet, if not where the
        library puts them."""
        key = f"{lib}:{name}"
        if key not in self.libs:
            self.libs[key] = library_symbol(lib, name)
        p = pins(self.libs[key])
        ends = {}
        for number, (px, py, pname) in p.items():
            sx, sy = turned(px, py, angle)
            ends[number] = (round(x + sx, 2), round(y + sy, 2), pname)
        self.pins[ref] = ends
        hide = ref.startswith("#")
        if text_at is None:
            f = fields(self.libs[key])
            text_at = [(round(x + turned(fx, fy, angle)[0], 2), round(y + turned(fx, fy, angle)[1], 2), fa, fj)
                       for fx, fy, fa, fj in (f["Reference"], f["Value"])]
        (rx, ry, ra, rj), (vx, vy, va, vj) = text_at
        props = [f'(property "Reference" "{ref}" (at {rx} {ry} {ra}) {effects(hide, rj)})',
                 f'(property "Value" "{value}" (at {vx} {vy} {va}) {effects(hide and "FLAG" in name, vj)})',
                 f'(property "Footprint" "" (at {x} {y} 0) {HIDDEN})',
                 f'(property "Datasheet" "" (at {x} {y} 0) {HIDDEN})']
        pin_lines = " ".join(f'(pin "{n}" (uuid "{self.u()}"))' for n in p)
        self.items.append(
            f'(symbol (lib_id "{key}") (at {x} {y} {angle}) (unit 1) (exclude_from_sim no) '
            f'(in_bom {"no" if hide else "yes"}) (on_board yes) (dnp no) (uuid "{self.u()}") {" ".join(props)} '
            f'{pin_lines} (instances (project "demo" (path "/{self.root}" (reference "{ref}") (unit 1)))))')
        return ends

    def label(self, name, at):
        self.items.append(f'(label "{name}" (at {at[0]} {at[1]} 0) {effects(False, "left bottom")} '
                          f'(uuid "{self.u()}"))')

    def wire(self, a, b):
        self.items.append(f'(wire (pts (xy {a[0]} {a[1]}) (xy {b[0]} {b[1]})) (stroke (width 0) (type default)) '
                          f'(uuid "{self.u()}"))')

    def junction(self, at):
        self.items.append(f'(junction (at {at[0]} {at[1]}) (diameter 0) (color 0 0 0 0) (uuid "{self.u()}"))')

    def no_connect(self, at):
        self.items.append(f'(no_connect (at {at[0]} {at[1]}) (uuid "{self.u()}"))')

    def power(self, name, ref, at, flag=False):
        self.place("power", name, ref, name, at[0], at[1])
        if flag:
            self.place("power", "PWR_FLAG", "#FLG" + ref[4:], "PWR_FLAG", at[0], at[1])

    def text(self) -> str:
        libs = "\n".join(self.libs.values())
        body = "\n".join(self.items)
        return (f'(kicad_sch (version 20250114) (generator "eeschema") (generator_version "10.0") (uuid "{self.root}") '
                f'(paper "A4") (title_block (title "Workbench Co-Pilot demo board") (rev "1"))\n'
                f"(lib_symbols\n{libs}\n)\n{body}\n"
                f'(sheet_instances (path "/" (page "1"))) (embedded_fonts no))\n')


def main():
    board = sys.argv[1] if len(sys.argv) > 1 else "led"
    if board not in ("led", "sensor"):
        sys.exit("python scripts/make_schematic.py [led|sensor] [OUT.kicad_sch]")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else HARDWARE / f"demo_{board}.kicad_sch"
    s = Sheet(board)
    esp = s.place("RF_Module", "ESP32-WROOM-32", "U1", "ESP32-WROOM-32 (DOIT DevKit)", 101.6, 101.6)
    used = set()
    gnd_pins = [n for n, (_, _, name) in esp.items() if name == "GND"]
    vdd = next(n for n, (_, _, name) in esp.items() if name == "VDD")
    en = next(n for n, (_, _, name) in esp.items() if name == "EN")
    io4 = next(n for n, (_, _, name) in esp.items() if name == "IO4")
    s.power("GND", "#PWR01", esp[gnd_pins[0]], flag=True)
    used.update(gnd_pins)
    s.power("+3V3", "#PWR02", esp[vdd], flag=True)
    used.add(vdd)
    tie = (round(esp[en][0] - 5.08, 2), esp[en][1])  # EN tied high a little way out, clear of U1's name
    s.wire(esp[en], tie)
    s.power("+3V3", "#PWR03", tie)
    used.add(en)
    used.add(io4)
    x4, y4 = esp[io4][0], esp[io4][1]
    if board == "led":
        # GPIO 4 runs out to the resistor's top; the LED hangs from its bottom, anode up, and its cathode goes to GND
        r = s.place("Device", "R", "R1", "330", 152.4, round(y4 + 3.81, 2))  # vertical: pin 1 up, pin 2 down
        s.wire((x4, y4), r["1"])
        d = s.place("Device", "LED", "D1", "LED", 152.4, round(y4 + 11.43, 2), 90,  # turned: anode up, on R1's pin 2
                    text_at=[(158.75, round(y4 + 10.16, 2), 90, "left"), (158.75, round(y4 + 12.7, 2), 90, "left")])
        k = next(n for n, (_, _, name) in d.items() if name == "K")
        s.label("LED_A", r["2"])
        s.power("GND", "#PWR04", d[k])
    else:
        # GPIO 4 runs out to the middle of the divider: the sensor from 3V3 above it, 10 kohm to GND below
        f = s.place("Device", "R", "R1", "TFS-100 force sensor", 152.4, round(y4 - 3.81, 2))
        s.power("+3V3", "#PWR04", f["1"])
        r = s.place("Device", "R", "R2", "10k", 152.4, round(y4 + 3.81, 2))
        s.power("GND", "#PWR05", r["2"])
        s.wire((x4, y4), r["1"])
        s.junction(r["1"])
    s.label("GPIO4", (round(x4 + 5.08, 2), y4))  # on the wire
    for n, end in esp.items():
        if n not in used:
            s.no_connect(end)
    out.write_text(s.text())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
