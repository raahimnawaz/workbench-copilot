"""Make the carrier: the printed sheet the breadboard lies on, with the AprilTags the reader maps every frame through.

    python scripts/make_carrier.py            # writes .wbc/vision/carrier.pdf (and a .png preview)

1. Print the PDF at 100% ("Actual size", not "Fit to page") on US Letter, landscape.
2. Measure the 100 mm bar with a ruler. If it isn't 100 mm, the printer scaled the page: print again.
3. Tape the sheet flat under the camera, and lay the breadboard on its outline with row 1 at the end marked
   "row 1" and column a on the side marked "a". Stick it down: the reader finds the board's holes to within
   about 1.2 mm of the outline, and a board further off than that reads a row out.

Everything comes from config/bench.yaml (the tags, their size and where they go, and each board's place)
and config/parts.yaml (the board's size), so change those and run this again, never the other way round.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from wbc import config  # noqa: E402
from wbc.perception import breadboard, tags  # noqa: E402

DPI = 300
GRAY = (150, 150, 150)


def text(img, s: float, words: str, xy_mm, size_mm: float = 2.5, color=(0, 0, 0), centered=False) -> None:
    """Words whose baseline starts at xy_mm, or whose middle sits on it when centered."""
    scale = size_mm * s / 22  # Hershey simplex capitals are about 22 px tall at scale 1
    thick = max(1, int(scale * 1.6))
    (w, h), _ = cv2.getTextSize(words, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = xy_mm[0] * s, xy_mm[1] * s
    if centered:
        x, y = x - w / 2, y + h / 2
    cv2.putText(img, words, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def sheet(bench: dict, parts: dict) -> np.ndarray:
    s = DPI / 25.4
    carrier = bench["carrier"]
    w_mm, h_mm = carrier["paper_mm"]
    img = np.full((int(round(h_mm * s)), int(round(w_mm * s)), 3), 255, np.uint8)
    tags.draw(img, carrier, s)
    half = carrier["tag_size_mm"] / 2
    for tag_id, (cx, cy) in carrier["tags"].items():
        text(img, s, f"tag {tag_id}", (cx - half, cy + half + 5), 2.2, GRAY)
    for b in breadboard.boards_from(bench, parts):
        body = b.body()
        cv2.polylines(img, [np.round(body * s).astype(np.int32)], True, GRAY, max(1, int(0.3 * s)), cv2.LINE_AA)
        u, v = np.asarray(b.u), np.asarray(b.v)
        length, width = b.spec["body_mm"]
        for spot, label in ((b.center() - (length / 2 + 9) * u, "row 1"),
                            (b.center() - (width / 2 + 5) * v, f"{b.name}: column a side")):
            text(img, s, label, spot, 2.5, centered=True)
    x0, y = 20.0, h_mm - 14  # the 100 mm check bar
    cv2.rectangle(img, (int(x0 * s), int(y * s)), (int((x0 + 100) * s), int((y + 1.5) * s)), (0, 0, 0), -1)
    text(img, s, "this bar is 100 mm: print at 100% (Actual size)", (x0 + 104, y + 1.5), 2.5)
    text(img, s, f"Workbench Co-Pilot carrier  {carrier['tag_family']}  ids {', '.join(map(str, carrier['tags']))}"
                 f"  {carrier['tag_size_mm']} mm", (20.0, 12.0), 2.5, GRAY)
    return img


def main() -> int:
    from PIL import Image

    img = sheet(config.load_yaml("bench.yaml"), config.load_yaml("parts.yaml"))
    out = config.ROOT / ".wbc" / "vision"
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "carrier.png"), img)
    Image.fromarray(img[:, :, ::-1]).convert("RGB").save(out / "carrier.pdf", resolution=DPI)
    print(f"wrote {out.relative_to(config.ROOT)}/carrier.pdf ({img.shape[1]}x{img.shape[0]} at {DPI} dpi). "
          f"Print it at 100%, then check the 100 mm bar with a ruler.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
