"""add_part: Muse Spark Open teaches itself a part from parts.yaml, and the part goes into the board's KiCad schematic.

Spark Open gets the part's pins, the DevKit's GPIOs and which of them the schematic and pins.h already use, and answers
how to wire the part and drive it: GND, 3V3 or a GPIO for each pin, the PlatformIO library and the setup code. Code
checks the wiring (every pin once, power pins to power, each signal on a free GPIO of its own that can drive an output
and won't stop the ESP32 booting) and sends it back once with what's wrong. The answer is saved as the part's skill in
.wbc/skills/, for writing its code later. Then the part is drawn into a copy of the schematic in .wbc/schematics/, a
box whose pins join the ESP32's by net labels, kept only if KiCad's electrical rules check passes and its netlist joins
every pin as wired. With --show, KiCad's window shows the copy. Nothing on the board changes, and nothing is flashed.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

from wbc.perception.schematic import FONT, HIDDEN, library_symbol, sexpr_block, turned

ASK_TOKENS = 4096  # the answer is a few hundred tokens; the rest is room for Spark Open's reasoning
ASK_TIMEOUT_S = 180  # its own, where a spoken question gives up at open.timeout_s: this waits for that reasoning
POWER = {"GND": "GND", "VCC": "3V3", "VDD": "3V3", "VIN": "3V3", "3V3": "3V3"}  # where a pin with that name goes
INPUT_ONLY = {34, 35, 36, 39}
AVOID = {0: "a boot pin", 1: "the USB serial's TX", 2: "a boot pin", 3: "the USB serial's RX",
         12: "a boot pin that sets the flash voltage"}
PITCH = 2.54  # KiCad's grid
SPREAD = 5.08  # between the new part's left pins, and between its wires' turns: twice the grid, to read on camera
LEAD_MM = 7.62  # from the ESP32's pin ends to its wires' first turn
PIN =re.compile(r'\(pin \w+ \w+\s*\(at ([-\d.]+) ([-\d.]+) ([-\d.]+)\).*?\(name "([^"]*)".*?\(number "([^"]*)"', re.S)
SYSTEM = (
    "You are the co-pilot of an electronics bench, teaching yourself a part you haven't used, so it can be wired to the "
    "bench's ESP32 DevKit and driven from its Arduino firmware. Decide where each of the part's pins goes and how to "
    "drive it, then answer with one JSON object and nothing else:\n"
    '{"wiring": {"<each of the part\'s pins>": "GND", "3V3" or "GPIO<n>"},\n'
    ' "nets": {"<each pin wired to a GPIO>": "a short net name in capitals, such as OLED_CS"},\n'
    ' "library": "the PlatformIO library that drives it, as owner/name",\n'
    ' "setup": "the C++ lines that create and start it in the sketch, with the pins you chose",\n'
    ' "notes": "one or two sentences someone wiring it should know",\n'
    ' "say": "one spoken sentence for the user saying which GPIO each signal pin goes to"}\n'
    "Power pins go to 3V3 or GND. Every other pin gets a GPIO of its own from the free ones listed, or 3V3 or GND if "
    "it must be tied. Where the part uses a bus the ESP32 has hardware for, use those pins if they're free: SPI clock "
    "GPIO18 and data GPIO23, I2C SDA GPIO21 and SCL GPIO22.")


def add(bench, key: str, use: str = "") -> dict:
    """The add_part tool: learn `key`, save its skill, and draw it into a copy of the schematic."""
    drawn = (bench.layout or {}).get("schematic")
    modules = (bench.parts or {}).get("modules") or {}
    part = modules.get(key)
    if not drawn:
        return {"ok": False, "error": "bench.yaml names no schematic"}
    if not isinstance(part, dict) or not part.get("pins"):
        return {"ok": False, "error": f"parts.yaml has no part {key!r} with its pins"}
    if bench.open is None:
        return {"ok": False, "error": "Spark Open is off (no BASETEN_API_KEY), so nothing can learn the part"}
    from wbc.perception import schematic

    src = bench.root / drawn
    before = schematic.read(src)
    fw = (bench.layout.get("firmware") or {}).get("file")
    try:
        header = (bench.root / fw).read_text(encoding="utf-8") if fw else ""
    except OSError:
        header = ""
    used = used_gpios(before["nets"], re.findall(r"#define\s+(\w+_PIN)\s+(\d+)", header))
    board_key = ((bench.layout.get("modules") or {}).get("esp32") or {}).get("part", "esp32_devkit_30")
    board = modules.get(board_key) or {}
    client, model, label = bench.open
    t0 = time.monotonic()
    wiring, answer, problems = learn(client, model, bench.budget, key, part, use, board.get("what", board_key),
                                     free_gpios(board, used), used)
    seconds = round(time.monotonic() - t0, 1)
    if problems:
        return {"ok": False, "error": f"{label}'s wiring didn't check out, twice: " + "; ".join(problems),
                "seconds": seconds}
    skill = save_skill(bench.root, key, part, wiring, answer, model)
    out = bench.root / ".wbc" / "schematics" / f"{src.stem}+{key}.kicad_sch"
    ref, esp, nets = draw(src, out, key, part, wiring, answer.get("nets"))
    after = schematic.read(out)
    joined = {n["net"]: set(n["connects"]) for n in after["nets"]}
    unjoined = [pin for pin, to in wiring.items()
                if f"{ref} {pin}" not in joined.get(nets.get(pin) or ("GND" if to == "GND" else "+3V3"), set())
                or (to.startswith("GPIO") and f"{esp} IO{to[4:]}" not in joined.get(nets[pin], set()))]
    rel = lambda p: str(p.relative_to(bench.root))  # noqa: E731
    if after["erc"]["errors"] or unjoined:
        return {"ok": False, "error": "the drawn schematic didn't check out: " + "; ".join(
            [f"{after['erc']['errors']} ERC errors {after['erc'].get('first', [])}"] * bool(after["erc"]["errors"])
            + [f"{', '.join(unjoined)} not joined as wired"] * bool(unjoined)), "schematic": rel(out)}
    bench.layout["schematic"] = rel(out)  # the design from now on, for read_schematic and another add_part
    shown = bench.show(out) if bench.show else None
    return {"ok": True, "part": key, "by": label, "seconds": seconds, "wiring": wiring, "nets": nets,
            "library": answer.get("library"), "notes": answer.get("notes"), "say": answer.get("say"),
            "skill": rel(skill), "schematic": rel(out), "erc": after["erc"], **({"shown": shown} if shown else {}),
            "board": "unchanged: nothing is wired or flashed"}


def target(value) -> str | None:
    """"GND", "3V3" or "GPIO<n>" for where a pin goes, as a model or a person writes it ("IO18", "D18", "3.3V")."""
    s = re.sub(r"\s+", "", str(value)).upper()
    if s in ("GND", "GROUND", "0V"):
        return "GND"
    if s in ("3V3", "+3V3", "3.3V", "+3.3V", "3.3"):
        return "3V3"
    m = re.fullmatch(r"(?:GPIO|IO|D|G)?(\d{1,2})", s)
    return f"GPIO{int(m.group(1))}" if m else None


def used_gpios(nets: list, defines: list) -> dict:
    """{gpio: what uses it}, from the schematic's nets ("U1 IO4") and pins.h's *_PIN defines."""
    used = {}
    for net in nets:
        for pin in net["connects"]:
            m = re.search(r" IO(\d+)$", pin)
            if m:
                used.setdefault(int(m.group(1)), f"net {net['net']} in the schematic")
    for name, number in defines:
        used.setdefault(int(number), f"{name} in pins.h")
    return used


def free_gpios(board: dict, used: dict) -> list:
    """The DevKit's header GPIOs a new part's signal can have: not used, not input-only, not a boot or serial pin."""
    gpios = {int(p[4:]) for side in ("left", "right") for p in board.get(side, []) if p.startswith("GPIO")}
    return sorted(n for n in gpios if n not in used and n not in INPUT_ONLY and n not in AVOID)


def check(answer, pins: list, free, used: dict) -> tuple:
    """(wiring, problems): each pin's "GND", "3V3" or "GPIO<n>", and what's wrong with the answer, if anything."""
    wiring = answer.get("wiring") if isinstance(answer, dict) else None
    if not isinstance(wiring, dict):
        return {}, ["there's no wiring"]
    problems = [f"{p} isn't one of the part's pins" for p in wiring if p not in pins]
    out, taken = {}, {}
    for pin in pins:
        if pin not in wiring:
            problems.append(f"{pin} isn't wired")
            continue
        to, power = target(wiring[pin]), POWER.get(pin.upper())
        if to is None:
            problems.append(f"{pin} goes to {wiring[pin]!r}, which isn't GND, 3V3 or a GPIO")
        elif power and to != power:
            problems.append(f"{pin} is a power pin, so it goes to {power}, not {to}")
        elif to.startswith("GPIO"):
            n = int(to[4:])
            if n in used:
                problems.append(f"{to} is already used ({used[n]})")
            elif n in INPUT_ONLY:
                problems.append(f"{to} can only read, so it can't drive {pin}")
            elif n in AVOID:
                problems.append(f"{to} is {AVOID[n]}")
            elif n not in free:
                problems.append(f"the DevKit has no free {to}")
            elif n in taken:
                problems.append(f"{to} already goes to {taken[n]}")
            taken.setdefault(n, pin)
        out[pin] = to
    return out, problems


def parse(text: str):
    """The first JSON object in a reply, after any reasoning; None if there isn't one."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    start = text.find("{")
    while start != -1:
        try:
            return json.JSONDecoder().raw_decode(text[start:])[0]
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
    return None


def learn(client, model: str, budget, key: str, part: dict, use: str, board: str, free: list, used: dict) -> tuple:
    """(wiring, answer, problems) from the model: asked once, and once more with what was wrong."""
    taken = "; ".join(f"GPIO{n}: {what}" for n, what in sorted(used.items())) or "nothing"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": (
        f"Part: {key}, {part.get('what', key)}. Its pins: {', '.join(part['pins'])}.\n"
        f"What it's for: {use or 'not said'}.\n"
        f"The board: {board}. Free GPIOs: {', '.join(f'GPIO{n}' for n in free)}.\n"
        f"Already used: {taken}.")}]
    answer, problems = None, []
    for _ in range(2):
        budget.check()
        r = client.chat.completions.create(model=model, max_tokens=ASK_TOKENS, messages=messages,
                                           timeout=ASK_TIMEOUT_S)
        budget.add_tokens(model, r.usage)
        text = r.choices[0].message.content or ""
        answer = parse(text)
        wiring, problems = (check(answer, part["pins"], free, used) if answer is not None
                            else ({}, ["the answer wasn't one JSON object"]))
        if not problems:
            return wiring, answer, []
        messages += [{"role": "assistant", "content": text[-6000:]}, {"role": "user", "content": (
            "That doesn't work: " + "; ".join(problems) + ". Answer again, as one JSON object.")}]
    return None, answer, problems


def save_skill(root: Path, key: str, part: dict, wiring: dict, answer: dict, model: str) -> Path:
    """The part's skill, .wbc/skills/<key>.md: its wiring, library and setup code, for writing its code later."""
    path = root / ".wbc" / "skills" / f"{key}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"| {pin} | {to} |" for pin, to in wiring.items())
    path.write_text(
        f"# {part.get('what', key)} ({key})\n\nLearned by {model} on {datetime.now():%Y-%m-%d %H:%M}.\n\n"
        f"| Pin | Goes to |\n|---|---|\n{rows}\n\nLibrary: `{answer.get('library', '')}`\n\n"
        f"```cpp\n{str(answer.get('setup', '')).strip()}\n```\n\n{str(answer.get('notes', '')).strip()}\n",
        encoding="utf-8")
    return path


def _snap(v: float) -> float:
    return round(round(v / PITCH) * PITCH, 2)


def _part_symbol(key: str, pins: list, left: list) -> tuple:
    """The part as a box: `left` down its left side, top first and SPREAD apart, its 3V3 pins on top and its GND pins
    underneath: (its library symbol, {pin: (x, y, angle) in symbol space}, the box's half height)."""
    top = (len(left) - 1) * SPREAD / 2 if left else 0
    half = top + PITCH
    at = {p: (-10.16, round(top - i * SPREAD, 2), 0) for i, p in enumerate(left)}
    for side, y, angle in (("3V3", half + PITCH, 270), ("GND", -half - PITCH, 90)):
        on_side = [p for p in pins if POWER.get(p.upper()) == side]  # right of middle, clear of the left pins' names
        at.update({p: (round(PITCH * (i + 1), 2), round(y, 2), angle) for i, p in enumerate(on_side)})
    lines = [f'(symbol "wbc:{key}" (pin_names (offset 1.016)) (exclude_from_sim no) (in_bom yes) (on_board yes)',
             f'(property "Reference" "U" (at 0 {half + 1.27} 0) {FONT})',
             f'(property "Value" "{key}" (at 0 {-half - 1.27} 0) {FONT})',
             f'(property "Footprint" "" (at 0 0 0) {HIDDEN})', f'(property "Datasheet" "" (at 0 0 0) {HIDDEN})',
             f'(symbol "{key}_0_1" (rectangle (start -7.62 {half}) (end 7.62 {-half}) (stroke (width 0.254) '
             f'(type default)) (fill (type background))))', f'(symbol "{key}_1_1"']
    for i, pin in enumerate(pins):
        kind = "power_in" if pin.upper() in POWER else "input"
        lines.append(f'(pin {kind} line (at {at[pin][0]} {at[pin][1]} {at[pin][2]}) (length 2.54) (name "{pin}" '
                     f'{FONT}) (number "{i + 1}" {FONT}))')
    return "\n".join(lines) + "))", at, half


def draw(src: Path, out: Path, key: str, part: dict, wiring: dict, nets) -> tuple:
    """Write `src` with the part added as `out`, below and right of what's there: each signal joined to its ESP32 pin
    by a wire that runs out, down and across, the top one turning furthest out so none cross, and named by a net label
    at the part; its power pins to power symbols; and the no-connect flags on the ESP32 pins it takes removed. (The
    part's reference, the ESP32's, {pin: net}.)"""
    text = src.read_text(encoding="utf-8")
    libs_start = text.index("(lib_symbols")
    libs_end = libs_start + len(sexpr_block(text, libs_start))
    placed = []
    for m in re.finditer(r'\(symbol\s*\(lib_id "([^"]+)"\)', text):
        block = sexpr_block(text, m.start())
        at = re.search(r'\(at ([-\d.]+) ([-\d.]+)(?: ([-\d.]+))?\)', block)
        ref = re.search(r'\(property "Reference" "([^"]*)"', block)
        placed.append((m.group(1), float(at.group(1)), float(at.group(2)), float(at.group(3) or 0),
                       ref.group(1) if ref else ""))
    esp = next(s for s in placed if "ESP32" in s[0])
    esp_lib = sexpr_block(text, text.index(f'(symbol "{esp[0]}"', libs_start))
    ends = {}
    for m in PIN.finditer(esp_lib):
        sx, sy = turned(float(m.group(1)), float(m.group(2)), esp[3])
        left = (float(m.group(3)) + esp[3]) % 360 == 0  # the pin points right, into a body to its right
        ends.setdefault(m.group(4), (round(esp[1] + sx, 2), round(esp[2] + sy, 2), left))
    root = re.search(r'\(kicad_sch.*?\(uuid "([^"]+)"\)', text, re.S).group(1)
    project = re.search(r'\(instances\s*\(project "([^"]*)"', text)
    project = project.group(1) if project else "demo"
    refs = {s[4] for s in placed}
    ref = next(f"U{n}" for n in range(2, 100) if f"U{n}" not in refs)
    power_refs = (f"#PWR{n:02d}" for n in range(1, 1000) if f"#PWR{n:02d}" not in refs)
    labels = set(re.findall(r'\(label "([^"]+)"', text))
    names = {}
    for pin, to in wiring.items():
        if to.startswith("GPIO"):
            want = str((nets or {}).get(pin, "")).strip().upper()
            if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,23}", want) or want in labels or want in names.values():
                want = f"{key.split('_')[0].upper()}_{pin}"
            names[pin] = want

    def instance(lib_id, x, y, ref_, value, numbers, ref_at, value_at, angle=0):
        hide = ref_.startswith("#")
        return (f'(symbol (lib_id "{lib_id}") (at {x} {y} {angle}) (unit 1) (exclude_from_sim no) '
                f'(in_bom {"no" if hide else "yes"}) (on_board yes) (dnp no) (uuid "{uuid.uuid4()}") '
                f'(property "Reference" "{ref_}" (at {ref_at[0]} {ref_at[1]} 0) {HIDDEN if hide else FONT}) '
                f'(property "Value" "{value}" (at {value_at[0]} {value_at[1]} 0) {FONT}) '
                f'(property "Footprint" "" (at {x} {y} 0) {HIDDEN}) (property "Datasheet" "" (at {x} {y} 0) {HIDDEN}) '
                + " ".join(f'(pin "{n}" (uuid "{uuid.uuid4()}"))' for n in numbers)
                + f' (instances (project "{project}" (path "/{root}" (reference "{ref_}") (unit 1)))))')

    def label(name, at, angle=0):
        return (f'(label "{name}" (at {at[0]} {at[1]} {angle}) (effects (font (size 1.27 1.27)) '
                f'(justify {"right" if angle == 180 else "left"} bottom)) (uuid "{uuid.uuid4()}"))')

    def wire(a, b):
        return (f'(wire (pts (xy {a[0]} {a[1]}) (xy {b[0]} {b[1]})) (stroke (width 0) (type default)) '
                f'(uuid "{uuid.uuid4()}"))')

    pins = list(part["pins"])
    gpio = {pin: int(to[4:]) for pin, to in wiring.items() if to.startswith("GPIO")}
    start = {pin: next(e for name, e in ends.items() if re.fullmatch(rf"(?:\S+/)?IO{n}(?:/\S+)?", name))
             for pin, n in gpio.items()}
    wired = sorted(gpio, key=lambda p: start[p][1])  # top first, as on the ESP32, so the wires don't cross
    left = wired + [p for p in pins if p.upper() not in POWER and p not in gpio]  # then any tied to 3V3 or GND
    lib_text, at, half = _part_symbol(key, pins, left)
    new_libs = [lib_text] + [library_symbol("power", name) for name in ("GND", "+3V3")
                             if f'(symbol "power:{name}"' not in text[libs_start:libs_end]]
    reach = max([e[0] for e in start.values()] or [esp[1]]) + LEAD_MM
    turns = {p: round(reach + (len(wired) - 1 - i) * SPREAD, 2) for i, p in enumerate(wired)}
    pins_x = _snap(max(max([reach, *turns.values()]) + 3 * SPREAD, max(s[1] for s in placed) + 25.4))
    first_y = _snap(max([e[1] for e in start.values()] or [esp[2]]) + 12.7)
    x, y = round(pins_x + 10.16, 2), round(first_y + (len(left) - 1) * SPREAD / 2, 2)
    value = str(part.get("what", key)).split(",")[0].replace('"', "'")
    items = [instance(f"wbc:{key}", x, y, ref, value, [str(i + 1) for i in range(len(pins))],
                      (round(x - 5.08, 2), round(y - half - 1.27, 2)), (x, round(y + half + 10.16, 2)))]  # under GND
    freed = []
    for pin in pins:
        px, py, side = at[pin]
        end = (round(x + px, 2), round(y - py, 2))
        to = wiring[pin]
        if to in ("GND", "3V3"):
            name = "GND" if to == "GND" else "+3V3"
            if side == 0:  # a signal pin tied to power, on the left: the symbol turned to point away from the box
                angle, value_at = (270 if name == "GND" else 90), (round(end[0] - 7.62, 2), end[1])
            else:
                angle, value_at = 0, (end[0], round(end[1] + (3.81 if name == "GND" else -3.81), 2))
            items.append(instance(f"power:{name}", end[0], end[1], next(power_refs), name, ["1"], end, value_at,
                                  angle))
            continue
        sx, sy, on_left = start[pin]
        if on_left:  # an ESP32 pin on its far side: a label at each end, not a wire across the ESP32
            items.append(label(names[pin], (sx, sy), 180))
        else:
            path = [(sx, sy), (turns[pin], sy), (turns[pin], end[1]), end]
            items += [wire(a, b) for a, b in zip(path, path[1:]) if a != b]
        items.append(label(names[pin], end, 180))
        freed.append((sx, sy))
    text = text[:libs_end - 1] + "\n" + "\n".join(new_libs) + "\n" + text[libs_end - 1:]

    def flagged(m):
        return any(abs(float(m.group(1)) - p[0]) < 0.01 and abs(float(m.group(2)) - p[1]) < 0.01 for p in freed)

    text = re.sub(r'\(no_connect\s*\(at ([-\d.]+) ([-\d.]+)\)\s*\(uuid "[^"]*"\)\s*\)\n?',
                  lambda m: "" if flagged(m) else m.group(0), text)
    tail = text.find("(sheet_instances")
    tail = tail if tail != -1 else text.rstrip().rfind(")")
    text = text[:tail] + "\n".join(items) + "\n" + text[tail:]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return ref, esp[4], names
