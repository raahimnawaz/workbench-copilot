"""The demo board's serial side: find its port, read its force sensor, flash it.

The Act 1 firmware (firmware/dut_demo) prints one line per reading, "mv 1873", every 20 ms, and every 50
readings a line naming the GPIO it reads, "# reading GPIO 15". So a couple of seconds of lines tell the router
both whether the sensor answers a press and which pin the firmware thinks it's on. Plain text rather than the
force sampler's binary frames: one sensor at 50 Hz needs no framing, and it reads in any serial monitor.

Opening the port with DTR and RTS asserted would reset the ESP32 through its auto-reset circuit, so both are
cleared before opening. Only one program can use a serial port at a time, so reads and flashes share a lock.
"""
from __future__ import annotations

import glob
import re
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path

RESPONDS_MV = 200  # a press through the 10 kOhm divider moves the reading far more than an idle pin wanders
TRACE_POINTS = 20
SIM_WIRED_GPIO = 4  # the recorded board: the sensor's jumper is on GPIO 4 ...
SIM_FIRMWARE_GPIO = 15  # ... while the firmware it shipped with reads GPIO 15

_LINE = re.compile(r"^mv (\d+)$")
_PIN = re.compile(r"^# reading GPIO (\d+)$")
_DEFINE = re.compile(r"#define\s+SENSOR_PIN\s+(\d+)")
_port_lock = threading.Lock()


def find_port(cfg: dict) -> str:
    """bench.yaml's firmware.port, or with "auto" the one USB serial port that isn't the arm's or the sampler's."""
    port = cfg.get("port", "auto")
    if port != "auto":
        return port
    skip = set(cfg.get("skip_ports", []))
    found = sorted({p for pattern in ("/dev/cu.usb*", "/dev/cu.wchusbserial*") for p in glob.glob(pattern)} - skip)
    if len(found) != 1:
        raise RuntimeError(f"found {len(found)} possible ports for the demo board ({', '.join(found) or 'none'}): "
                           "plug it in, or set firmware.port in bench.yaml")
    return found[0]


def parse(lines) -> tuple:
    """(millivolt readings, the GPIO the firmware says it reads, or None) from the board's serial lines."""
    mv, gpio = [], None
    for line in lines:
        line = line.strip()
        if m := _LINE.match(line):
            mv.append(int(m.group(1)))
        elif m := _PIN.match(line):
            gpio = int(m.group(1))
    return mv, gpio


def summarize(mv: list, gpio, seconds: float) -> dict:
    if not mv:
        return {"ok": False, "error": "no readings came from the demo board", "firmware_gpio": gpio}
    lo, hi = min(mv), max(mv)
    step = max(1, len(mv) // TRACE_POINTS)
    return {"ok": True, "firmware_gpio": gpio, "seconds": round(seconds, 1), "readings": len(mv),
            "mv": {"min": lo, "max": hi, "mean": round(sum(mv) / len(mv)), "last": mv[-1]},
            "responds": hi - lo >= RESPONDS_MV, "trace_mv": mv[::step][:TRACE_POINTS]}


def _open(port: str, baud: int):
    import serial  # pyserial, from the arm extra; imported here so the rest works without it

    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, baud, 0.2
    s.dtr = s.rts = False  # don't reset the board by opening its port
    s.open()
    return s


def read(port: str, seconds: float = 2.0, baud: int = 115200, opener=_open) -> dict:
    """Read the board's lines for `seconds` and summarize them."""
    lines = []
    with _port_lock:
        t0 = time.monotonic()
        s = opener(port, baud)
        try:
            while time.monotonic() - t0 < seconds:
                raw = s.readline()
                if raw:
                    lines.append(raw.decode("ascii", "replace"))
        finally:
            s.close()
        took = time.monotonic() - t0
    out = summarize(*parse(lines), took)
    out["port"] = port
    return out


def flash(root: Path, cmd: str, port: str, timeout_s: float = 180) -> dict:
    """Build the firmware as it is in the working tree, staged patches included, and upload it."""
    args = shlex.split(cmd.format(port=port))
    if shutil.which(args[0]) is None:
        return {"ok": False, "error": f"{args[0]} is not on PATH: activate the venv first"}
    with _port_lock:
        t0 = time.monotonic()
        try:
            p = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"the upload took over {timeout_s:.0f} s", "port": port}
    log = (p.stdout + p.stderr)[-800:]
    out = {"ok": p.returncode == 0, "port": port, "seconds": round(time.monotonic() - t0, 1), "log": log}
    if p.returncode:
        out["error"] = "the build or upload failed"
    return out


def firmware_gpio(root: Path) -> int | None:
    """SENSOR_PIN as pins.h says it now, staged patches included."""
    m = _DEFINE.search((root / "firmware" / "dut_demo" / "include" / "pins.h").read_text(encoding="utf-8"))
    return int(m.group(1)) if m else None


def sim_read(flashed_gpio: int | None) -> dict:
    """What read_sensor returns with --sim. The recorded board's sensor is on GPIO 4, so it answers a press only
    once firmware reading GPIO 4 has been flashed; before that, the board runs the firmware that reads GPIO 15."""
    gpio = SIM_FIRMWARE_GPIO if flashed_gpio is None else flashed_gpio
    if gpio == SIM_WIRED_GPIO:
        mv = [142, 141, 143, 520, 1310, 1840, 1902, 1915, 1890, 1876, 1650, 900, 300, 145, 142, 141]
    else:
        mv = [18, 22, 19, 21, 20, 23, 19, 18, 22, 20, 21, 19, 20, 22, 18, 21]
    out = summarize(mv, gpio, 2.0)
    out["sim"] = True
    return out
