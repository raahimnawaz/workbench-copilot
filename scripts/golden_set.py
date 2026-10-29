"""Score the board reader on the golden set (free): replay each labeled photo's saved SAM reply.

    python scripts/golden_set.py                      # with bench.yaml's tip_inset_mm
    python scripts/golden_set.py --inset -0.8 -1.0 -1.3   # compare insets

Labels live in config/golden.yaml; the photos and saved reads in .wbc/vision/. The gate is 20 photos with
at least 98% of wire ends on their true hole.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from wbc import config  # noqa: E402
from wbc.perception import golden  # noqa: E402

VISION = config.ROOT / ".wbc" / "vision"


def run(labels: dict, bench: dict, parts: dict, vision: Path, verbose: bool) -> tuple:
    scores, skipped = [], []
    for entry in labels.get("photos") or []:
        photo, read = vision / entry["photo"], vision / entry["read"]
        if not (photo.exists() and read.exists()):
            skipped.append(entry["photo"])
            continue
        s = golden.score(entry["photo"], entry["wires"], golden.replay(photo, read, bench, parts))
        scores.append(s)
        if verbose:
            print(f"  {s.photo}: {s.right}/{s.ends} ends right" + (f", {s.extra} extra wires" if s.extra else ""))
            for truth, got in s.misses:
                print(f"    {' to '.join(truth)} read as {' to '.join(h or '?' for h in got)}")
    return scores, skipped


def main(argv=None, vision: Path = VISION) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inset", type=float, nargs="*", help="tip_inset_mm values to try instead of bench.yaml's")
    args = ap.parse_args(argv)
    bench, parts, labels = (config.load_yaml(n) for n in ("bench.yaml", "parts.yaml", "golden.yaml"))
    gate = labels["gate"]
    insets = args.inset or [bench["reader"].get("tip_inset_mm", 0.0)]
    passed = False
    for inset in insets:
        bench["reader"]["tip_inset_mm"] = inset
        print(f"tip_inset_mm {inset:+.1f}:")
        scores, skipped = run(labels, bench, parts, vision, verbose=True)
        right, ends, share = golden.total(scores)
        resid = [r for s in scores for r in s.residuals_mm]
        margin = (f"; right ends sit {np.median(resid):.2f} mm (median), {max(resid):.2f} mm at worst, "
                  f"from their holes (flagged past {bench['reader'].get('snap_mm', 1.2)} mm)") if resid else ""
        print(f"  {right}/{ends} ends right ({share:.1%}) on {len(scores)} photos{margin}")
        if skipped:
            print(f"  ! skipped, photo or saved read not found: {', '.join(skipped)}")
        ok = len(scores) >= gate["photos"] and share >= gate["ends_right"]
        print(f"  gate ({gate['photos']} photos, {gate['ends_right']:.0%} right): "
              + ("PASS" if ok else f"not yet ({len(scores)} photos)" if share >= gate["ends_right"] else "FAIL"))
        passed |= ok
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
