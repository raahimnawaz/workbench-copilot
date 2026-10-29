"""Score the board reader against hand-labeled photos: how many wire ends land on their true hole.

Each labeled wire is matched to the read wire that shares the most ends with it (either order), one to one.
An end counts as right only on the exact hole; a missed wire counts both its ends wrong, and a read wire
that matches no label counts as an extra.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from . import netlist, reader, sam_client


@dataclass
class PhotoScore:
    photo: str
    ends: int = 0
    right: int = 0
    extra: int = 0  # read wires that match no label
    misses: list = field(default_factory=list)  # (true holes, read holes) for each wire with a wrong end
    residuals_mm: list = field(default_factory=list)  # of the right ends, to show the margin under snap_mm


def _right(truth: list, read: list) -> int:
    """How many of a labeled wire's two ends a read wire got, taking the better of its two orders."""
    a, b = truth
    x, y = (read + [None, None])[:2]
    return max((a == x) + (b == y), (a == y) + (b == x))


def match(truth: list, read: list) -> list:
    """For each labeled wire, the index of its read wire (or None), maximizing the ends right."""
    # Pair both-ends-right first, then one-end-right: a wire shares a hole with at most a few others.
    pairs = sorted(((_right(t, r), ti, ri) for ti, t in enumerate(truth) for ri, r in enumerate(read)), reverse=True)
    best, used = [None] * len(truth), set()
    for n, ti, ri in pairs:
        if n and best[ti] is None and ri not in used:
            best[ti] = ri
            used.add(ri)
    return best


def score(photo: str, truth: list, wires: list) -> PhotoScore:
    """wires: the reader's Wire objects (or dicts with 'ends' and 'residual_mm')."""
    holes = [[e.hole for e in w.ends] if hasattr(w, "ends") else list(w["ends"]) for w in wires]
    resid = [[e.residual_mm for e in w.ends] if hasattr(w, "ends") else list(w["residual_mm"]) for w in wires]
    out = PhotoScore(photo, ends=2 * len(truth))
    picks = match(truth, holes)
    for t, i in zip(truth, picks):
        got = holes[i] if i is not None else [None, None]
        n = _right(t, got)
        out.right += n
        if n < 2:
            out.misses.append((list(t), got))
        if i is not None:
            out.residuals_mm += [r for h, r in zip(got, resid[i]) if h in t and r is not None]
    out.extra = len(wires) - sum(i is not None for i in picks)
    return out


def replay(photo_path: Path, read_path: Path, bench: dict, parts: dict) -> list:
    """The reader's wires for a saved photo, from the SAM reply saved with its read (free)."""
    image = cv2.imread(str(photo_path))
    if image is None:
        raise RuntimeError(f"couldn't read {photo_path}")
    _, boards, area, view = reader.locate_boards(image, bench, parts)
    registered = [b for b, _ in reader.register_all(boards, view, area, bench["reader"])]
    instances, _ = sam_client.parse(json.loads(read_path.read_text())["sam_raw"], bench["reader"]["wire_phrase"])
    return netlist.trace(instances, view, area, bench["reader"]["px_per_mm"], registered, bench["reader"])


def total(scores: list) -> tuple:
    ends = sum(s.ends for s in scores)
    right = sum(s.right for s in scores)
    return right, ends, (right / ends if ends else 0.0)
