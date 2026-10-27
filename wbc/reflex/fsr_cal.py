"""Force sensor calibration and creep: turn divider voltages into newtons, and measure how far a steady
reading drifts.

Each sensor is one half of a divider: sensor from the supply to the ADC pin, a fixed resistor from the pin
to ground. A thin-film sensor's conductance (1/R) rises roughly in proportion to force, so each sensor gets
a power law, F = k * G**b, fitted as a straight line in log-log through a few known weights. G is the
conductance in microsiemens above the unloaded reading, which is why re-zeroing with the jaws open cancels
whatever offset the sensor has drifted to.

Creep is the slow part of drift: under a steady load the reading keeps changing for minutes. A slip is fast
(the slip cue fires below -3 N/s), so creep matters most during the long compliant hold.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import yaml

GRAVITY = 9.80665
SUPPLY_MV = 3300.0
R_FIXED_OHM = 10_000.0
ADC_GOOD_MV = (150.0, 2450.0)  # the classic ESP32's ADC is accurate between these at 11 dB
SLIP_N_PER_S = 3.0             # the slip detector's drop cue (blueprint sheet 04)


def grams_to_newtons(grams: float) -> float:
    return grams * GRAVITY / 1000.0


def conductance_us(mv, supply_mv: float = SUPPLY_MV, r_fixed: float = R_FIXED_OHM):
    """Sensor conductance in microsiemens from the pin voltage. Takes a number or a numpy array."""
    v = np.clip(np.asarray(mv, dtype=float), 0.0, supply_mv - 1.0)
    # V = Vs * Rf / (Rs + Rf), so Rs = Rf * (Vs - V) / V and G = 1 / Rs = V / (Rf * (Vs - V))
    g = 1e6 * v / (r_fixed * (supply_mv - v))
    return float(g) if g.ndim == 0 else g


def sensor_ohms(mv, supply_mv: float = SUPPLY_MV, r_fixed: float = R_FIXED_OHM) -> float:
    g = conductance_us(mv, supply_mv, r_fixed)
    return math.inf if g <= 0 else 1e6 / g


@dataclass
class Calibration:
    k: float                  # F = k * G**b, with F in N and G in µS above the unloaded reading
    b: float
    zero_mv: float = 0.0      # the unloaded reading while calibrating; re-zero before each grasp in use
    supply_mv: float = SUPPLY_MV
    r_fixed: float = R_FIXED_OHM
    r2: float = float("nan")  # of the log-log line
    points: list = field(default_factory=list)  # the [grams, mV] pairs it was fitted on
    date: str = ""

    def newtons(self, mv, zero_mv: float | None = None):
        """Force for a reading. Pass zero_mv to re-zero on a fresh unloaded reading."""
        z = self.zero_mv if zero_mv is None else zero_mv
        g = conductance_us(mv, self.supply_mv, self.r_fixed) - conductance_us(z, self.supply_mv, self.r_fixed)
        f = self.k * np.clip(np.asarray(g, dtype=float), 0.0, None) ** self.b
        return float(f) if f.ndim == 0 else f

    def ohms_at(self, newtons: float) -> float:
        """The sensor's resistance at a given force, from the fit."""
        g = (newtons / self.k) ** (1.0 / self.b) + conductance_us(self.zero_mv, self.supply_mv, self.r_fixed)
        return 1e6 / g


def fit(points, zero_mv: float, supply_mv: float = SUPPLY_MV, r_fixed: float = R_FIXED_OHM) -> Calibration:
    """Fit F = k * G**b through (grams, mV) points. Needs two or more loads that read above unloaded."""
    g0 = conductance_us(zero_mv, supply_mv, r_fixed)
    used = [(grams_to_newtons(gr), conductance_us(mv, supply_mv, r_fixed) - g0) for gr, mv in points]
    used = [(f, g) for f, g in used if f > 0 and g > 0]
    if len(used) < 2:
        raise ValueError("need at least two weights that read above the unloaded level")
    x = np.log([g for _, g in used])
    y = np.log([f for f, _ in used])
    if np.ptp(y) < 1e-9:
        raise ValueError("every reading used the same weight: a fit needs at least two different weights")
    if np.ptp(x) < 1e-9:
        raise ValueError("every weight gave the same reading: check the wiring")
    b, a = np.polyfit(x, y, 1)
    ss_res = float(np.sum((y - (a + b * x)) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return Calibration(k=float(math.exp(a)), b=float(b), zero_mv=float(zero_mv), supply_mv=supply_mv,
                       r_fixed=r_fixed, r2=1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
                       points=[[float(gr), float(mv)] for gr, mv in points], date=date.today().isoformat())


def fit_rows(cal: Calibration) -> list:
    """Per point: grams, newtons, mV, sensor ohms, fitted newtons, error in %."""
    rows = []
    for grams, mv in cal.points:
        n = grams_to_newtons(grams)
        fitted = cal.newtons(mv)
        err = (fitted - n) / n * 100 if n > 0 else float("nan")
        rows.append((grams, n, mv, sensor_ohms(mv, cal.supply_mv, cal.r_fixed), fitted, err))
    return rows


def save(cals: dict, path: Path) -> Path:
    def tidy(v):
        if isinstance(v, float):
            return float(f"{v:.6g}")
        if isinstance(v, list):
            return [tidy(x) for x in v]
        return v

    data = {ch: {k: tidy(v) for k, v in asdict(c).items()} for ch, c in cals.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Force sensor calibrations, one per channel, written by scripts/calibrate_fsr.py.\n"
                    "# F [N] = k * G**b, where G [µS] is the sensor's conductance above the unloaded reading.\n"
                    + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {ch: Calibration(**fields) for ch, fields in data.items()}


@dataclass
class CreepReport:
    unit: str                # "N" when calibrated, otherwise "µS" (conductance above unloaded)
    base_mv: float
    noise_mv: float
    t_on: float              # when the load went on, s from the start of the recording
    ref: float               # the reading 1 s after the load went on; drift is measured from here
    checkpoints: list        # (s after the load went on, reading, % change from ref)
    rate: float | None       # drift over the last 10 s before removal, in unit/s
    t_off: float | None      # when the load came off
    residual: list           # (s after removal, reading, % of ref)
    expected_n: float | None = None

    def pct_at(self, seconds: float):
        return next((p for c, _, p in self.checkpoints if c == seconds), None)

    def lines(self) -> list:
        u = self.unit
        out = [f"unloaded: {self.base_mv:.0f} mV (noise {self.noise_mv:.1f} mV)"]
        ref = f"load on at {self.t_on:.2f} s; one second later: {self.ref:.3f} {u}"
        if self.expected_n:
            ref += f" (the weight is {self.expected_n:.3f} N, so {(self.ref / self.expected_n - 1) * 100:+.0f}%)"
        out.append(ref)
        notes = {3: "  a lift lasts about this long", 60: "  the compliant hold"}
        for c, v, p in self.checkpoints:
            out.append(f"  {c:4.0f} s after loading: {v:.3f} {u}  {p:+.1f}%{notes.get(c, '')}")
        if self.rate is not None:
            line = f"drift rate at the end of the hold: {self.rate:+.4f} {u}/s"
            if u == "N" and self.rate:
                line += f", {SLIP_N_PER_S / abs(self.rate):,.0f}x slower than the {SLIP_N_PER_S:g} N/s slip cue"
            out.append(line)
        if self.residual:
            line = "after removal: " + ", ".join(f"{c:g} s later {p:+.1f}% of the load" for c, _, p in self.residual)
            if self.base_mv <= ADC_GOOD_MV[0]:
                line += " (the ESP32 reads nothing below about 142 mV, so a small leftover wouldn't show)"
            out.append(line)

        lift = self.pct_at(3)
        if lift is not None:
            if abs(lift) < 5:
                out.append(f"Lift: the reading moves {lift:+.1f}% over a lift, so re-zeroing before each grasp is enough.")
            else:
                out.append(f"Lift: the reading moves {lift:+.1f}% over a lift. Compare grip forces at the same delay "
                           f"after contact, or tune the grip targets with that in mind.")
        long = [(c, p) for c, _, p in self.checkpoints if c >= 30]
        if long:
            c, p = long[-1]
            if abs(p) >= 10:
                out.append(f"Hold: it moves {p:+.1f}% over {c:.0f} s. During the compliant hold, lock the jaw position "
                           f"once the grip is reached instead of regulating force on this reading.")
            else:
                out.append(f"Hold: it moves {p:+.1f}% over {c:.0f} s, so regulating force during the hold is fine.")
        if self.residual and abs(self.residual[0][2]) > 3:
            out.append(f"Unloading: 1 s after removal it still read {self.residual[0][2]:+.1f}% of the load, "
                       f"so re-zero with the jaws open before every grasp.")
        return out


def longest_run(t, mask, bridge_s: float = 0.5):
    """(first, one past last) indices of the longest stretch where mask holds, bridging gaps under bridge_s."""
    edges = np.diff(np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if starts.size == 0:
        return None
    runs = [[starts[0], ends[0]]]
    for s, e in zip(starts[1:], ends[1:]):
        if t[s] - t[runs[-1][1] - 1] < bridge_s:
            runs[-1][1] = e
        else:
            runs.append([s, e])
    return tuple(max(runs, key=lambda r: t[r[1] - 1] - t[r[0]]))


def analyze_creep(t, mv, cal: Calibration | None = None, grams: float | None = None,
                  checkpoints=(3, 10, 30, 60, 120), baseline_s: float = 2.0, ref_s: float = 1.0,
                  supply_mv: float = SUPPLY_MV, r_fixed: float = R_FIXED_OHM) -> CreepReport:
    """Find the load step in an unloaded-loaded-unloaded recording and measure the drift from 1 s after it.

    t is in seconds, mv is one channel. The first baseline_s must be unloaded. The load is the longest
    stretch above the unloaded level, so a fumbled placement or two before it are ignored. Without a
    calibration the drift is measured in conductance (supply_mv and r_fixed describe the divider), which
    is close to proportional to force.
    """
    t = np.asarray(t, dtype=float)
    mv = np.asarray(mv, dtype=float)
    base = t < t[0] + baseline_s
    if base.sum() < 20:
        raise ValueError(f"need at least {baseline_s:g} s of unloaded readings at the start")
    base_mv = float(np.median(mv[base]))
    noise = float(np.std(mv[base]))
    hold = longest_run(t, ~base & (mv > base_mv + max(10 * noise, 30.0)))
    if hold is None:
        raise ValueError("no load step found: the reading never rose above the unloaded level")
    i_on, i_end = hold
    t_on = float(t[i_on])

    if cal is not None:
        sig, unit = cal.newtons(mv, zero_mv=base_mv), "N"  # re-zeroed on this run, as a grasp would be
    else:
        sig, unit = conductance_us(mv, supply_mv, r_fixed) - conductance_us(base_mv, supply_mv, r_fixed), "µS"

    def at(tc: float):
        sel = np.abs(t - tc) <= 0.1
        return float(np.median(sig[sel])) if sel.any() else None

    ref = at(t_on + ref_s)
    if not ref or ref <= 0:
        raise ValueError("the load was too light to read")
    mv_ref = float(np.median(mv[np.abs(t - (t_on + ref_s)) <= 0.1]))

    t_off = None
    if i_end < t.size:  # the load came off: removal is where the final drop crossed halfway down
        mid = base_mv + 0.5 * (mv_ref - base_mv)
        j = i_end
        while j > i_on and mv[j - 1] < mid:
            j -= 1
        t_off = float(t[j])
    loaded_end = t_off if t_off is not None else float(t[-1])

    marks = []
    for c in checkpoints:
        v = at(t_on + c) if t_on + c + 0.1 <= loaded_end - 0.3 else None  # clear of the hand lifting it off
        if v is not None:
            marks.append((c, v, (v - ref) / ref * 100))

    rate = None
    sel = (t >= max(t_on + ref_s, loaded_end - 10.5)) & (t <= loaded_end - 0.5)
    if sel.sum() >= 50 and np.ptp(t[sel]) >= 3.0:
        rate = float(np.polyfit(t[sel], sig[sel], 1)[0])

    residual = []
    if t_off is not None:
        for c in (1, 10):
            v = at(t_off + c) if t_off + c + 0.1 <= t[-1] else None
            if v is not None:
                residual.append((c, v, v / ref * 100))

    expected = grams_to_newtons(grams) if (grams and cal is not None) else None
    return CreepReport(unit, base_mv, noise, t_on, ref, marks, rate, t_off, residual, expected)
