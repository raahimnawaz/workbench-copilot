"""SAM 3.1 on the Meta Model API: a short noun phrase and one image in, a box and a mask for every match out.

From Meta's docs (dev.meta.ai/docs/sam, read Oct 28):
- One concept per request. A list or an "and" doesn't make two, so several phrases go out as parallel
  requests, each billed as one image ($2.50 per 1,000).
- Name the thing ("jumper wire"). Commands and questions often find nothing, and no match is a normal,
  empty reply, not an error.
- The reply is one line of special tokens: an object id, a box and a mask per match. Each mask covers only
  its own box. Masks come back in the compact one_bit encoding and are decoded with Meta's meta-sam-parser,
  which is pure Python: about 1.4 µs per pixel of box, so a long diagonal wire takes about 0.15 s.
"""
from __future__ import annotations

import base64
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format

JPEG_QUALITY = 95
# BGR, picked to stay apart from each other on a white breadboard.
PALETTE = [(232, 158, 47), (60, 76, 231), (113, 204, 46), (182, 89, 155), (15, 196, 241),
           (34, 126, 230), (166, 166, 26), (141, 68, 44), (80, 127, 255), (0, 0, 0)]


@dataclass
class Instance:
    """One match. The box is in image pixels, with right and bottom exclusive; the mask covers only the box."""

    phrase: str
    object_id: str
    box: tuple  # (left, top, right, bottom)
    mask: np.ndarray  # bool, shape (bottom - top, right - left)
    score: float | None = None  # the model's confidence, 0 to 1

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def full_mask(self, shape) -> np.ndarray:
        """The mask placed in an empty image of `shape` (the image it came from)."""
        out = np.zeros(shape[:2], bool)
        left, top, right, bottom = self.box
        out[top:bottom, left:right] = self.mask
        return out


@dataclass
class Segmentation:
    phrase: str
    instances: list
    problems: list  # records the parser had to drop: each one may be an object SAM found but we lost
    raw: str  # the reply text, so a paid run can be parsed again offline
    seconds: float = 0.0


def data_url(image) -> str:
    """An image as a data URL: a BGR array (sent as JPEG), encoded image bytes, or a file path."""
    if isinstance(image, np.ndarray):
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise ValueError("couldn't encode the image as JPEG")
        data = buf.tobytes()
    else:
        data = bytes(image) if isinstance(image, (bytes, bytearray)) else Path(image).read_bytes()
    mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def parse(text: str, phrase: str = "") -> tuple:
    """SAM's reply line as (instances, problems). An empty reply means nothing matched."""
    parser = image_segmentation_format().create_parser()
    parser.push(text)
    result = parser.finish(CompletedOutcome()).result
    instances, problems = [], []
    for rec in result.records:
        if rec.kind == "text":
            problems.append(f"text instead of records: {rec.text[:120]!r}")
        elif rec.kind == "mask":
            b = rec.bounds
            box = (int(b.left), int(b.top), int(b.right), int(b.bottom))
            raster = np.frombuffer(decode_mask_to_raster(rec.mask), np.uint8)
            mask = raster.reshape(rec.mask.height, rec.mask.width).astype(bool)
            size = (box[2] - box[0], box[3] - box[1])
            if mask.shape != size[::-1]:  # Meta's examples send masks at the box's size, but don't promise it
                mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
            instances.append(Instance(phrase, rec.object_id, box, mask, rec.confidence))
    # A box whose mask couldn't be decoded has no instance, so its diagnostic is the only trace of it.
    problems += [f"{d.code}: {d.message}" for d in result.diagnostics if d.severity == "error"]
    return instances, problems


def segment(client, cfg: dict, budget, image, phrase: str) -> Segmentation:
    """One phrase on one image. API errors raise; an empty `instances` means nothing matched."""
    url = image if isinstance(image, str) and image.startswith("data:") else data_url(image)
    budget.check()
    t0 = time.monotonic()
    r = client.responses.create(
        model=cfg["model"],
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": phrase},
            {"type": "input_image", "image_url": url}]}],
        # The API takes metadata values as strings. one_bit masks are several times smaller than the default.
        metadata={"mask_encoding": "one_bit", "include_confidence": "true"},
        timeout=cfg.get("timeout_s", 30))
    seconds = time.monotonic() - t0
    budget.add_images(cfg["model"], 1)
    text = getattr(r, "output_text", "") or ""
    instances, problems = parse(text, phrase)
    return Segmentation(phrase, instances, problems, text, seconds)


def segment_many(client, cfg: dict, budget, image, phrases: list) -> dict:
    """Several phrases on one image as parallel requests, since SAM takes one concept per request."""
    url = image if isinstance(image, str) and image.startswith("data:") else data_url(image)
    phrases = list(dict.fromkeys(phrases))
    with ThreadPoolExecutor(max_workers=max(1, len(phrases))) as pool:
        results = list(pool.map(lambda p: segment(client, cfg, budget, url, p), phrases))
    return {s.phrase: s for s in results}


def overlay(image: np.ndarray, instances: list, alpha: float = 0.45) -> np.ndarray:
    """A copy of `image` with each mask tinted and outlined, and its box labeled with id and score."""
    out = image.copy()
    for i, inst in enumerate(instances):
        color = PALETTE[i % len(PALETTE)]
        left, top, right, bottom = inst.box
        region = out[top:bottom, left:right]
        region[inst.mask] = (region[inst.mask] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        contours, _ = cv2.findContours(inst.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 1, offset=(left, top))
        cv2.rectangle(out, (left, top), (right - 1, bottom - 1), color, 1)
        label = inst.object_id if inst.score is None else f"{inst.object_id} {inst.score:.2f}"
        origin = (left, max(12, top - 4))
        cv2.putText(out, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out
