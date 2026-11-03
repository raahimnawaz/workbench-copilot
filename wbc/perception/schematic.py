"""The demo board's KiCad schematic: what each pin connects to, and KiCad's electrical rules check.

The schematic is the design: what the board should be. `read_schematic` gives Spark its nets beside the board read (what
the board is) and pins.h (what the firmware thinks), so a fault shows as the three disagreeing. Both come from KiCad's
own command line (kicad-cli, from KiCad 10 at /Applications/KiCad), so nothing here parses the drawing itself.
scripts/make_schematic.py draws hardware/demo_led.kicad_sch and demo_sensor.kicad_sch from KiCad's symbols.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

KICAD_CLI = Path("/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli")  # where KiCad's macOS installer puts it
TIMEOUT_S = 30


def kicad_cli() -> str | None:
    found = shutil.which("kicad-cli")
    return found or (str(KICAD_CLI) if KICAD_CLI.exists() else None)


def _run(args: list) -> None:
    cli = kicad_cli()
    if cli is None:
        raise RuntimeError("kicad-cli isn't installed (KiCad 10, from kicad.org)")
    p = subprocess.run([cli, *args], capture_output=True, text=True, timeout=TIMEOUT_S)
    if p.returncode not in (0, 5):  # 5: ERC found violations, which the report says
        raise RuntimeError(f"kicad-cli {args[0]} {args[1]} failed: {(p.stderr or p.stdout).strip()[-300:]}")


def parse_netlist(text: str) -> tuple:
    """(nets, values) from kicad-cli's kicadsexpr netlist: each net's name and its (ref, pin, function) nodes, and
    each part's value."""
    values = dict(re.findall(r'\(comp\s+\(ref "([^"]+)"\)\s+\(value "([^"]*)"\)', text))
    nets = []
    for block in re.split(r"\n\s*\(net\s*\n", text)[1:]:
        name = re.search(r'\(name "([^"]*)"\)', block).group(1)
        nodes = [(ref, pin, (fn or "").rsplit("_", 1)[0] if fn else "")
                 for ref, pin, fn in re.findall(r'\(node\s+\(ref "([^"]+)"\)\s+\(pin "([^"]+)"\)'
                                                r'(?:\s+\(pinfunction "([^"]*)"\))?', block)]
        nets.append((name.lstrip("/"), nodes))
    return nets, values


def read(path: Path) -> dict:
    """The schematic's nets that join two or more pins, each pin as "U1 IO4" or "R1 pin 1 (330)", and its ERC."""
    path = Path(path)
    with tempfile.TemporaryDirectory() as tmp:
        net_file, erc_file = Path(tmp) / "board.net", Path(tmp) / "erc.json"
        _run(["sch", "export", "netlist", "--format", "kicadsexpr", "-o", str(net_file), str(path)])
        _run(["sch", "erc", "--format", "json", "--severity-error", "-o", str(erc_file), str(path)])
        nets, values = parse_netlist(net_file.read_text(encoding="utf-8"))
        report = json.loads(erc_file.read_text(encoding="utf-8"))
    errors = [v.get("description", "") for sheet in report.get("sheets", []) for v in sheet.get("violations", [])]
    joined = []
    for name, nodes in nets:
        if len(nodes) < 2 or name.startswith("unconnected-"):
            continue
        pins = [f"{ref} {fn}" if fn and not ref.startswith(("R", "C")) else f"{ref} pin {pin} ({values.get(ref, '')})"
                for ref, pin, fn in nodes]
        joined.append({"net": name, "connects": sorted(set(pins))})
    return {"file": path.name, "nets": joined,
            "erc": {"errors": len(errors), **({"first": errors[:3]} if errors else {})}}
