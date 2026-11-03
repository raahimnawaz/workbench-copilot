"""The demo board's schematic as a KiCad 10 file, drawn from KiCad's own symbols.

    python scripts/make_schematic.py [OUT.kicad_sch]     # hardware/demo_sensor.kicad_sch by default

ESP32-WROOM-32 (the DOIT DevKit's module) as U1, its 3V3 and GND, EN tied high, and every other pin marked unused:
3V3 -> TFS-100 -> row 26 (GPIO 4) -> 10 kohm -> GND. Nets are joined by labels at pin ends. A schematic carries a copy
of every library symbol it uses, which KiCad writes one token a line; here each pin and shape takes one line instead,
and the ids come from each item's place in the drawing, so the same script always draws the same file.
"""
import itertools
import re
import sys
import uuid
from pathlib import Path

LIBS = Path("/Applications/KiCad/KiCad.app/Contents/SharedSupport/symbols")
OUT = Path(__file__).resolve().parent.parent / "hardware" / "demo_sensor.kicad_sch"
TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|[()]|[^\s()"]+')  # a quoted string, a parenthesis or a bare word
OWN_LINE = {"symbol", "property", "pin", "rectangle", "polyline", "circle", "arc", "text", "bezier", "embedded_fonts"}


def sexpr_block(text: str, start: int) -> str:
    """The balanced (...) starting at `start`, skipping parentheses inside quoted strings."""
    depth, i = 0, start
    while i < len(text):
        c = text[i]
        if c == '"':
            i += 1
            while text[i] != '"':
                i += 2 if text[i] == "\\" else 1
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
        i += 1
    raise ValueError("unbalanced")


def compact(block: str) -> str:
    """A library symbol's text with each of its units, properties, pins and shapes on a line of its own, instead of
    KiCad's one token a line. Only whitespace outside quoted strings changes, so KiCad reads the same symbol."""
    tokens = TOKEN.findall(block)
    out, depth = [], 0
    for k, tok in enumerate(tokens):
        if tok == "(" and k and depth <= 2 and tokens[k + 1] in OWN_LINE:
            out.append("\n" + "\t" * (depth + 1))
        elif tok != ")" and out and out[-1] != "(":
            out.append(" ")
        out.append(tok)
        depth += {"(": 1, ")": -1}.get(tok, 0)
    return "\t" + "".join(out)


def lib_symbol(lib: str, name: str) -> str:
    text = (LIBS / f"{lib}.kicad_sym").read_text()
    m = re.search(r'\n\t\(symbol "' + re.escape(name) + r'"\n', text)
    block = sexpr_block(text, m.start() + 2)
    return compact(block.replace(f'(symbol "{name}"', f'(symbol "{lib}:{name}"', 1))


def pins(block: str) -> dict:
    """number -> (x, y, name) in symbol space, from a library symbol's text."""
    out = {}
    for m in re.finditer(r'\(pin \w+ \w+\s*\(at ([-\d.]+) ([-\d.]+) [-\d.]+\).*?\(name "([^"]*)".*?\(number "([^"]*)"',
                         block, re.S):
        x, y, name, number = float(m.group(1)), float(m.group(2)), m.group(3), m.group(4)
        out.setdefault(number, (x, y, name))
    return out


IDS = itertools.count()


def u() -> str:
    """The next item's id, from its place in the drawing."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"workbench-copilot/demo_sensor/{next(IDS)}"))


ROOT_UUID = u()
FONT = "(effects (font (size 1.27 1.27)))"
HIDDEN = "(effects (font (size 1.27 1.27)) (hide yes))"


class Sheet:
    def __init__(self):
        self.libs, self.items, self.pins = {}, [], {}
        self.refs = {}

    def place(self, lib, name, ref, value, x, y, angle=0):
        key = f"{lib}:{name}"
        if key not in self.libs:
            self.libs[key] = lib_symbol(lib, name)
        p = pins(self.libs[key])
        ends = {}
        for number, (px, py, pname) in p.items():
            # symbol y is up, the sheet's down; rotation counterclockwise on screen
            if angle == 0:
                sx, sy = px, -py
            elif angle == 90:
                sx, sy = -py, -px
            elif angle == 180:
                sx, sy = -px, py
            else:
                sx, sy = py, px
            ends[number] = (round(x + sx, 2), round(y + sy, 2), pname)
        self.pins[ref] = ends
        hide = ref.startswith("#")
        value_font = HIDDEN if hide and "FLAG" in name else FONT
        props = [f'(property "Reference" "{ref}" (at {x + 2.54} {y - 2.54} 0) {HIDDEN if hide else FONT})',
                 f'(property "Value" "{value}" (at {x + 2.54} {y + 2.54} 0) {value_font})',
                 f'(property "Footprint" "" (at {x} {y} 0) {HIDDEN})',
                 f'(property "Datasheet" "" (at {x} {y} 0) {HIDDEN})']
        pin_lines = " ".join(f'(pin "{n}" (uuid "{u()}"))' for n in p)
        self.items.append(
            f'(symbol (lib_id "{key}") (at {x} {y} {angle}) (unit 1) (exclude_from_sim no) '
            f'(in_bom {"no" if hide else "yes"}) (on_board yes) (dnp no) (uuid "{u()}") {" ".join(props)} {pin_lines} '
            f'(instances (project "demo" (path "/{ROOT_UUID}" (reference "{ref}") (unit 1)))))')
        return ends

    def label(self, name, at):
        x, y = at[0], at[1]
        self.items.append(f'(label "{name}" (at {x} {y} 0) (effects (font (size 1.27 1.27)) (justify left bottom)) '
                          f'(uuid "{u()}"))')

    def no_connect(self, at):
        self.items.append(f'(no_connect (at {at[0]} {at[1]}) (uuid "{u()}"))')

    def power(self, name, ref, at, flag=False):
        self.place("power", name, ref, name, at[0], at[1])
        if flag:
            self.place("power", "PWR_FLAG", "#FLG" + ref[4:], "PWR_FLAG", at[0], at[1])

    def text(self) -> str:
        libs = "\n".join(self.libs.values())
        body = "\n".join(self.items)
        return (f'(kicad_sch (version 20250114) (generator "eeschema") (generator_version "10.0") (uuid "{ROOT_UUID}") '
                f'(paper "A4") (title_block (title "Workbench Co-Pilot demo board") (rev "1"))\n'
                f"(lib_symbols\n{libs}\n)\n{body}\n"
                f'(sheet_instances (path "/" (page "1"))) (embedded_fonts no))\n')


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    s = Sheet()
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
    s.power("+3V3", "#PWR03", esp[en])
    used.add(en)
    s.label("GPIO4", esp[io4])
    used.add(io4)
    f = s.place("Device", "R", "R1", "TFS-100 force sensor", 152.4, 83.82)  # vertical: pin 1 up, pin 2 down
    s.power("+3V3", "#PWR04", f["1"])
    s.label("GPIO4", f["2"])
    r = s.place("Device", "R", "R2", "10k", 152.4, 101.6)
    s.label("GPIO4", r["1"])
    s.power("GND", "#PWR05", r["2"])
    for n, end in esp.items():
        if n not in used:
            s.no_connect(end)
    out.write_text(s.text())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
