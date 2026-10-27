"""The force sensor link, calibration fit and creep analysis, plus the calibrate and creep scripts against a
fake sensor. No hardware needed."""
import math
import random
import time
from pathlib import Path

import numpy as np
import pytest

from wbc.config import ROOT
from wbc.reflex import fsr_cal, fsr_link
from wbc.reflex.fsr_link import Frame

import importlib.util

_spec = importlib.util.spec_from_file_location("calibrate_fsr", ROOT / "scripts" / "calibrate_fsr.py")
calibrate_fsr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calibrate_fsr)

K, B, ZERO_MV = 0.02, 1.1, 20.0  # the fake sensor: F = 0.02 * G**1.1, reading 20 mV unloaded


def mv_for(newtons, k=K, b=B, zero_mv=ZERO_MV, supply=fsr_cal.SUPPLY_MV, r_fixed=fsr_cal.R_FIXED_OHM):
    """The pin voltage the fake sensor gives at a force (the inverse of the calibration model)."""
    g = (newtons / k) ** (1 / b) if newtons > 0 else 0.0
    g += fsr_cal.conductance_us(zero_mv, supply, r_fixed)
    r_sensor = 1e6 / g
    return supply * r_fixed / (r_sensor + r_fixed)


# --- link ---------------------------------------------------------------------------------------------

def test_crc_matches_the_standard_check_value():
    assert fsr_link.crc16(b"123456789") == 0x29B1  # CRC-16/CCITT-FALSE


def test_frames_survive_junk_and_any_chunking():
    frames = [fsr_link.encode(i, 1000 + 2000 * i, (i, 2 * i, 3 * i, 4000 - i)) for i in range(200)]
    stream = b"ets Jun  8 2016 00:22:57\r\nrst:0x1 (POWERON_RESET)\r\n" + b"".join(frames)
    parser, out, rng, pos = fsr_link.Parser(), [], random.Random(1), 0
    while pos < len(stream):
        n = rng.randint(1, 40)
        out += parser.feed(stream[pos:pos + n])
        pos += n
    assert [f.seq for f in out] == list(range(200))
    assert out[7].mv == (7, 14, 21, 3993) and out[7].t_us == 15000
    assert parser.dropped == 0 and parser.bad_crc == 0


def test_a_damaged_frame_is_skipped_and_counted():
    good = [fsr_link.encode(i, 0, (100, 100, 100, 100)) for i in range(3)]
    bad = bytearray(good[1])
    bad[9] ^= 0x40
    parser = fsr_link.Parser()
    out = parser.feed(good[0] + bytes(bad) + good[2])
    assert [f.seq for f in out] == [0, 2]
    assert parser.bad_crc >= 1 and parser.dropped == 1


def test_gaps_wraparound_and_restarts():
    parser = fsr_link.Parser()
    parser.feed(b"".join(fsr_link.encode(s, 0, (0, 0, 0, 0)) for s in (65534, 65535, 0, 3)))
    assert parser.dropped == 2  # 1 and 2 are missing; 65535 -> 0 is just the counter wrapping
    parser.feed(fsr_link.encode(4, 0, (0, 0, 0, 0)) + fsr_link.encode(0, 0, (0, 0, 0, 0)))
    assert parser.restarts == 1 and parser.dropped == 2


def test_unwrap_seconds_handles_the_32_bit_clock():
    t = fsr_link.unwrap_seconds([(1 << 32) - 2000, 0, 2000])
    assert t == pytest.approx([0.0, 0.002, 0.004])


class FakeSerial:
    def __init__(self, data: bytes, chunk: int = 7):
        self.data, self.chunk, self.closed = bytearray(data), chunk, False

    @property
    def in_waiting(self):
        return min(len(self.data), self.chunk)

    def read(self, n):
        if not self.data:
            time.sleep(0.001)
            return b""
        out = bytes(self.data[:n])
        del self.data[:n]
        return out

    def close(self):
        self.closed = True


def test_sampler_collects_frames_on_a_thread():
    port = FakeSerial(b"".join(fsr_link.encode(i, 2000 * i, (i, 0, 0, 0)) for i in range(300)))
    sampler = fsr_link.Sampler("fake", serial_factory=lambda p, **kw: port).start()
    try:
        assert sampler.wait_for(300, timeout=3.0)
        assert sampler.latest().seq == 299
        assert len(sampler.window(0.0, float("inf"))) == 300
    finally:
        sampler.close()
    assert port.closed


def test_firmware_sends_the_frame_the_host_reads():
    src = (ROOT / "firmware" / "fsr_sampler" / "src" / "main.cpp").read_text()
    assert f"uint8_t frame[{fsr_link.FRAME_LEN}] = {{0xA5, 0x5A}}" in src
    assert "{" + ", ".join(str(p) for p in fsr_link.PINS.values()) + "}" in src
    assert f"PERIOD_US = {1_000_000 // fsr_link.RATE_HZ}" in src
    assert "0x1021" in src and "0xFFFF" in src


# --- calibration ---------------------------------------------------------------------------------------

def test_conductance_of_a_divider_at_half_supply():
    assert fsr_cal.conductance_us(1650) == pytest.approx(100.0)  # sensor = 10 kΩ
    assert fsr_cal.conductance_us(0) == 0.0
    assert fsr_cal.sensor_ohms(0) == math.inf


def test_fit_recovers_the_sensor_law():
    points = [(g, mv_for(fsr_cal.grams_to_newtons(g))) for g in (20, 50, 100, 200, 500)]
    cal = fsr_cal.fit(points, ZERO_MV)
    assert cal.k == pytest.approx(K, rel=1e-6) and cal.b == pytest.approx(B, rel=1e-6)
    assert cal.r2 == pytest.approx(1.0)
    assert cal.newtons(mv_for(1.5)) == pytest.approx(1.5, rel=1e-6)
    assert cal.newtons(ZERO_MV) == 0.0 and cal.newtons(5.0) == 0.0  # at or below unloaded reads zero
    assert all(abs(row[-1]) < 1e-6 for row in fsr_cal.fit_rows(cal))
    assert fsr_cal.Calibration(k=K, b=B, zero_mv=ZERO_MV).ohms_at(1.0) == pytest.approx(
        fsr_cal.sensor_ohms(mv_for(1.0)), rel=1e-6)


def test_fit_needs_two_readable_weights():
    with pytest.raises(ValueError):
        fsr_cal.fit([(100, 500.0)], ZERO_MV)
    with pytest.raises(ValueError):
        fsr_cal.fit([(50, 10.0), (100, 15.0)], ZERO_MV)  # both below the unloaded reading


def test_calibrations_save_and_load(tmp_path):
    cal = fsr_cal.fit([(g, mv_for(fsr_cal.grams_to_newtons(g))) for g in (50, 200)], ZERO_MV)
    path = fsr_cal.save({"SD": cal}, tmp_path / "fsr_cal.yaml")
    back = fsr_cal.load(path)["SD"]
    assert back.k == pytest.approx(cal.k, rel=1e-5)
    assert np.ravel(back.points) == pytest.approx(np.ravel(cal.points), rel=1e-5)
    assert fsr_cal.load(tmp_path / "missing.yaml") == {}


# --- creep ---------------------------------------------------------------------------------------------

def creep_trace(t_on=4.0, t_off=130.0, end=150.0, creep=0.05, tau=20.0, residual=0.03, noise=2.0):
    """A recording: unloaded, then about 1 N whose conductance creeps up 5% (time constant 20 s), then a
    3% residue after removal that fades in 3 s."""
    t = np.arange(0.0, end, 0.002)
    g1 = (1.0 / K) ** (1 / B)
    g = np.zeros_like(t)
    on = (t >= t_on) & (t < t_off)
    g[on] = g1 * (1 + creep * (1 - np.exp(-(t[on] - t_on) / tau)))
    off = t >= t_off
    g[off] = g1 * residual * np.exp(-(t[off] - t_off) / 3.0)
    g += fsr_cal.conductance_us(ZERO_MV)
    mv = fsr_cal.SUPPLY_MV * fsr_cal.R_FIXED_OHM / (1e6 / g + fsr_cal.R_FIXED_OHM)
    mv += np.random.default_rng(0).normal(0, noise, t.size)
    return t, mv, g1


def test_creep_found_and_measured_in_conductance():
    t, mv, g1 = creep_trace()
    r = fsr_cal.analyze_creep(t, mv)
    assert r.unit == "µS" and r.t_on == pytest.approx(4.0, abs=0.01) and r.t_off == pytest.approx(130.0, abs=0.01)
    shape = lambda s: 1 + 0.05 * (1 - math.exp(-s / 20))  # noqa: E731
    assert r.ref == pytest.approx(g1 * shape(1), rel=0.01)
    assert [c for c, _, _ in r.checkpoints] == [3, 10, 30, 60, 120]
    short = fsr_cal.analyze_creep(*creep_trace(t_off=90.0)[:2])
    assert [c for c, _, _ in short.checkpoints] == [3, 10, 30, 60]  # 120 s would run past the removal
    assert r.pct_at(60) == pytest.approx((shape(60) / shape(1) - 1) * 100, abs=0.3)
    assert r.rate is not None and 0 < r.rate < 0.01
    assert r.residual[0][0] == 1 and r.residual[0][2] == pytest.approx(3 * math.exp(-1 / 3) / shape(1), abs=0.3)


def test_creep_in_newtons_with_a_calibration():
    t, mv, _ = creep_trace(creep=0.15)
    cal = fsr_cal.Calibration(k=K, b=B, zero_mv=ZERO_MV)
    r = fsr_cal.analyze_creep(t, mv, cal=cal, grams=102.0)
    assert r.unit == "N" and r.ref == pytest.approx(1.0, rel=0.02)
    assert r.expected_n == pytest.approx(1.0003, rel=1e-3)
    text = "\n".join(r.lines())
    assert "slip cue" in text and "lock the jaw position" in text and "re-zero" not in text.split("Lift:")[0]


def test_creep_ignores_fumbled_placements():
    # Two touches that fell off (like a coin stack that toppled) before the stack finally sat still at 20 s
    t, mv, _ = creep_trace(t_on=20.0, t_off=130.0)
    mv[((t > 5.0) & (t < 6.0)) | ((t > 10.0) & (t < 11.5))] = 900.0
    r = fsr_cal.analyze_creep(t, mv)
    assert r.t_on == pytest.approx(20.0, abs=0.01) and r.t_off == pytest.approx(130.0, abs=0.01)
    assert [c for c, _, _ in r.checkpoints] == [3, 10, 30, 60]


def test_longest_run_bridges_short_dips():
    t = np.arange(0, 10, 0.1)
    mask = (t > 1) & (t < 2) | (t > 4) & (t < 9)
    mask[(t > 6) & (t < 6.25)] = False  # a 0.2 s dip inside the long stretch
    i0, i1 = fsr_cal.longest_run(t, mask)
    assert t[i0] == pytest.approx(4.1) and t[i1 - 1] == pytest.approx(8.9)
    assert fsr_cal.longest_run(t, np.zeros(t.size, bool)) is None


def test_creep_needs_a_load_step():
    t = np.arange(0, 10, 0.002)
    with pytest.raises(ValueError):
        fsr_cal.analyze_creep(t, np.full(t.size, 20.0))


# --- the script, against a fake sensor -------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeSampler:
    """Serves frames at 500 Hz from a function of time, like a sampler that has been streaming all along."""

    def __init__(self, clock, mv_at):
        self.clock, self.mv_at, self.parser = clock, mv_at, fsr_link.Parser()

    def frame(self, k):
        t = k / fsr_link.RATE_HZ
        return Frame(k & 0xFFFF, (k * 2000) & 0xFFFFFFFF, tuple(int(round(v)) for v in self.mv_at(t)), t)

    def window(self, t0, t1):
        t1 = min(t1, self.clock())
        return [self.frame(k) for k in range(max(math.ceil(t0 * 500), 0), math.ceil(t1 * 500))]

    def latest(self):
        return self.frame(int(self.clock() * 500))


def test_calibrate_script_fits_and_saves(tmp_path, capsys):
    clock = FakeClock()
    load = {"grams": 0.0}
    sampler = FakeSampler(clock, lambda t: (mv_for(fsr_cal.grams_to_newtons(load["grams"])), 0, 0, 0))
    answers = iter(["", "20", "fifty", "50", "100", "200", "500", ""])

    def ask(prompt):
        a = next(answers)
        if a.replace(".", "").isdigit():
            load["grams"] = float(a)
        return a

    cal, marks = calibrate_fsr.calibrate(sampler, "SD", tmp_path / "fsr_cal.yaml", ask=ask, clock=clock,
                                         sleep=clock.sleep)
    assert cal.k == pytest.approx(K, rel=0.02) and cal.b == pytest.approx(B, rel=0.02)  # mV is rounded
    assert [m for _, m in marks] == ["zero", "20 g", "50 g", "100 g", "200 g", "500 g"]
    assert "SD" in fsr_cal.load(tmp_path / "fsr_cal.yaml")
    out = capsys.readouterr().out
    assert "Type a number" in out and "saved SD" in out


def test_creep_script_reports_and_logs(tmp_path, capsys):
    clock = FakeClock()
    g1 = (1.0 / K) ** (1 / B)
    t_on, t_off = clock.t + 3.3, clock.t + 3.0 + 20.0 + 0.4  # the "hand" is a little late each time

    def mv_at(t):
        if t_on <= t < t_off:
            g = g1 * (1 + 0.04 * math.log1p(t - t_on)) + fsr_cal.conductance_us(ZERO_MV)
            return (fsr_cal.SUPPLY_MV * fsr_cal.R_FIXED_OHM / (1e6 / g + fsr_cal.R_FIXED_OHM), 0, 0, 0)
        return (ZERO_MV, 0, 0, 0)

    report, frames, marks = calibrate_fsr.creep(FakeSampler(clock, mv_at), "SD", 102, seconds=20, after=12,
                                                clock=clock, sleep=clock.sleep)
    assert report.t_on == pytest.approx(3.3, abs=0.01)
    shape = lambda s: 1 + 0.04 * math.log1p(s)  # noqa: E731
    assert report.pct_at(10) == pytest.approx((shape(10) / shape(1) - 1) * 100, abs=0.3)
    assert [m for _, m in marks] == ["place", "remove"]
    path = calibrate_fsr.write_log(frames, marks, "creep", folder=tmp_path)
    lines = path.read_text().splitlines()
    assert lines[0] == "t,seq,SD,SP,MD,MP,mark" and len(lines) == len(frames) + 1
    assert sum(1 for line in lines if line.endswith((",place", ",remove"))) == 2
    assert "isn't calibrated" in capsys.readouterr().out
