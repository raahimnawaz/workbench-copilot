"""The desk camera: where things on the desk are, in the arm's frame.

A second webcam looks down at the desk and sees the arm's whole reach. Tags taped flat around the arm (config/desk.yaml)
are touched once each with the arm's tool point, motors off, which puts every tag's center in the arm's frame
(scripts/desk_camera.py touch). From then on each frame is mapped through the tags it shows, as the board reader maps
the carrier: a homography from image pixels to the table, here straight into the arm's x and y in mm. A bumped camera
only needs the tags still in view; a moved tag needs touching again.

One camera can't see height, and something standing up looks further from the point under the camera than it is: a
cup's rim 95 mm up, seen from 800 mm, lands 13% further out. The homography and the camera's focal length place the
camera itself (camera_from), so a point said to be h mm up can be moved back along its ray (DeskFit.lift).

locate shows SAM 3.1 a top-down view of the arm's reach (top_down), one request per phrase ($0.0025), and turns each
match into a center, a size and a heading in the arm's frame (found). Something standing up is placed from its
outline's near edge, its base on the table, and its far edge, its top pushed out by its height: exact for an upright
cylinder or box, given its height.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import tags as tagmap

TOUCHED = Path(".wbc") / "desk_tags.json"  # where touch saves the tags' centers, under the repo root
NOMINAL_WIDTH = 1920  # config/desk.yaml's focal_px is for frames this wide
STRAIGHT_DOWN = 20.0  # degrees of tilt under which one view can't tell the focal length from the camera's height


@dataclass
class DeskFit:
    """One frame mapped onto the table, in the arm's frame."""

    H: np.ndarray       # image pixels -> the arm's (x, y) on the table, mm
    tags: list          # the touched tags it used
    missing: list       # touched tags it didn't see
    errors: dict        # {tag: mm} how far a fit from the other tags puts each tag's center from where it was touched
    rest: dict          # {tag: mm} how well the other tags agree among themselves without it, rms (6 or more in view)
    camera: np.ndarray  # the camera's (x, y) in the arm's frame and its height above the table, mm
    table_z: float      # the table's height in the arm's frame, mm: the touched centers' mean
    focal_px: float     # the focal length the camera was placed with

    @property
    def rms_mm(self) -> float | None:
        """How far the tags land from their touches when the others place them, rms over all of them; None with 4."""
        return float(np.sqrt(np.mean(np.square(list(self.errors.values()))))) if self.errors else None

    @property
    def suspect(self) -> int | None:
        """The tag that's off, if one is: leaving it out leaves the rest agreeing at least twice as well as leaving
        out any other. Its own entry in `errors` needn't be the largest, because a bad tag pulls the fits that place
        the others. Needs 6 tags in view."""
        if len(self.rest) < 3:
            return None
        (tag, best), (_, runner_up) = sorted(self.rest.items(), key=lambda kv: kv[1])[:2]
        return tag if best < runner_up / 2 else None

    def to_table(self, pixels) -> np.ndarray:
        """Image pixels to the arm's (x, y) on the table, mm."""
        return tagmap.to_mm(self.H, pixels)

    def lift(self, xy, height_mm: float) -> np.ndarray:
        """Where something the camera sees over table point(s) xy really is, if it's height_mm above the table: back
        along its ray, toward the point under the camera."""
        k = (self.camera[2] - height_mm) / self.camera[2]
        return self.camera[:2] + (np.asarray(xy, float) - self.camera[:2]) * k


@dataclass
class Found:
    """One match for a phrase, in the arm's frame, mm."""

    phrase: str
    score: float | None
    center: np.ndarray  # (x, y) of its middle
    height_mm: float    # how tall it was taken to be
    top_z: float        # its top, in the arm's frame
    length_mm: float    # along its long axis
    width_mm: float     # across it
    heading_deg: float  # its long axis, from the arm's straight ahead (+x) toward its left (+y), -90 to 90
    reach_mm: float     # its center's distance from the arm's base, on the table
    pixels: int         # its mask's size in the top-down view

    def summary(self) -> dict:
        """What the router sees."""
        return {"phrase": self.phrase, "score": None if self.score is None else round(self.score, 2),
                "x_mm": round(float(self.center[0]), 1), "y_mm": round(float(self.center[1]), 1),
                "top_z_mm": round(self.top_z, 1), "height_mm": self.height_mm,
                "length_mm": round(self.length_mm, 1), "width_mm": round(self.width_mm, 1),
                "heading_deg": round(self.heading_deg, 1), "reach_mm": round(self.reach_mm, 1)}


def center(corners) -> np.ndarray:
    """A tag's center in the image: where its diagonals cross. Perspective keeps that point, while the corners' mean
    drifts toward the tag's near side."""
    p0, p1, p2, p3 = np.asarray(corners, float)
    d1, d2 = p2 - p0, p3 - p1
    t = np.linalg.solve(np.column_stack([d1, -d2]), p1 - p0)[0]
    return p0 + t * d1


def load_touched(path) -> dict:
    """{tag id: (x, y, z) mm in the arm's frame}, as touch saved them."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no desk tags at {path}: run scripts/desk_camera.py touch first")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): np.asarray(v, float) for k, v in data["tags"].items()}


def save_touched(path, touched: dict, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"tags": {str(k): [round(float(v), 2) for v in p] for k, p in sorted(touched.items())},
            "saved": datetime.now().isoformat(timespec="seconds"), **(extra or {})}
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


def focal_for(cfg: dict, width: int) -> float:
    """config/desk.yaml's focal length, in pixels of a frame `width` wide."""
    return cfg["focal_px"] * width / NOMINAL_WIDTH


def _homography(src, dst) -> np.ndarray:
    H, _ = cv2.findHomography(np.asarray(src, float), np.asarray(dst, float), 0)
    if H is None:
        raise ValueError("the tags' centers don't fix a homography: are three of them in a line?")
    return H


def _intrinsics(size, focal_px: float) -> np.ndarray:
    w, h = size
    return np.array([[focal_px, 0, (w - 1) / 2], [0, focal_px, (h - 1) / 2], [0, 0, 1]], float)


def camera_from(H: np.ndarray, size, focal_px: float, near) -> np.ndarray:
    """Where the camera is: its (x, y) in the arm's frame and its height above the table, mm.

    H's inverse takes the table's (x, y) to image pixels, which for a camera with focal length focal_px and its optical
    center mid-frame is K [r1 r2 t] up to scale: r1 and r2 are the arm's x and y axes seen from the camera, and t the
    arm's origin. `near` is a table point in view, which has to come out in front of the camera."""
    B = np.linalg.inv(_intrinsics(size, focal_px)) @ np.linalg.inv(H)
    scale = 2 / (np.linalg.norm(B[:, 0]) + np.linalg.norm(B[:, 1]))
    if (B @ np.array([near[0], near[1], 1.0]))[2] * scale < 0:
        scale = -scale
    r1, r2, t = scale * B[:, 0], scale * B[:, 1], scale * B[:, 2]
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    C = -(U @ Vt).T @ t
    if C[2] <= 0:
        raise ValueError("the tags put the camera under the table: check that each tag was touched under its own id")
    return C


def tilt_deg(H: np.ndarray, size, focal_px: float) -> float:
    """How far from straight down the camera looks, in degrees."""
    B = np.linalg.inv(_intrinsics(size, focal_px)) @ np.linalg.inv(H)
    n = np.cross(B[:, 0], B[:, 1])  # the table's normal, seen from the camera
    return math.degrees(math.acos(min(1.0, abs(n[2]) / np.linalg.norm(n))))


def focal_from(H: np.ndarray, size) -> float | None:
    """The focal length the tags imply, in pixels, or None when the camera looks nearly straight down: there height
    and focal length trade off, and one view can't tell them apart.

    With the optical center mid-frame, K^-1 applied to H's inverse has to give two columns that are at right angles
    and as long as each other, which fixes 1/f^2."""
    w, h = size
    G = np.linalg.inv(H)
    G = G / np.linalg.norm(G)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    a, b, c = G[0, :2] - cx * G[2, :2], G[1, :2] - cy * G[2, :2], G[2, :2]
    p = np.array([a[0] * a[1] + b[0] * b[1], a[0] ** 2 + b[0] ** 2 - a[1] ** 2 - b[1] ** 2])
    q = np.array([c[0] * c[1], c[0] ** 2 - c[1] ** 2])
    inv_f2 = -(p @ q) / (p @ p)
    if inv_f2 <= 0:
        return None
    f = 1 / math.sqrt(inv_f2)
    if tilt_deg(H, size, f) < STRAIGHT_DOWN:
        return None
    return f


def fit(detections: dict, touched: dict, size, focal_px: float, min_tags: int = 4) -> DeskFit:
    """Map a frame onto the table from the touched tags it shows: at least min_tags of them, and never fewer than 4."""
    used = sorted(set(detections) & set(touched))
    need = max(4, min_tags)
    if len(used) < need:
        raise ValueError(f"the camera sees {len(used)} of the {len(touched)} touched desk tags {sorted(touched)} (tags "
                         f"in view: {sorted(detections) or 'none'}). It needs at least {need}: check they're in view, "
                         "lit and not covered.")
    src = np.array([center(detections[i]) for i in used])
    dst = np.array([touched[i][:2] for i in used])
    H = _homography(src, dst)
    errors, rest = {}, {}
    if len(used) > 4:  # with 4, the fit passes through every tag and can't check any
        for k, tag in enumerate(used):
            keep = [j for j in range(len(used)) if j != k]
            Hk = _homography(src[keep], dst[keep])
            errors[tag] = float(np.linalg.norm(tagmap.to_mm(Hk, src[k])[0] - dst[k]))
            if len(keep) > 4:
                rest[tag] = float(np.sqrt(np.mean(np.sum((tagmap.to_mm(Hk, src[keep]) - dst[keep]) ** 2, axis=1))))
    camera = camera_from(H, size, focal_px, dst.mean(axis=0))
    table_z = float(np.mean([touched[i][2] for i in used]))
    return DeskFit(H, used, sorted(set(touched) - set(used)), errors, rest, camera, table_z, focal_px)


def view_matrix(region_mm, px_per_mm: float) -> np.ndarray:
    """The arm's (x, y) mm to pixels of a top-down view of region_mm = (x0, y0, x1, y1): the arm's straight ahead (+x)
    up and its left (+y) to the left, as seen from above."""
    x0, y0, x1, y1 = region_mm
    s = px_per_mm
    return np.array([[0, -s, s * y1], [-s, 0, s * x1], [0, 0, 1]], float)


def top_down(frame: np.ndarray, desk: DeskFit, region_mm, px_per_mm: float) -> tuple:
    """The top-down view of region_mm that locate shows SAM, and the map from its pixels to the arm's (x, y) mm."""
    x0, y0, x1, y1 = region_mm
    M = view_matrix(region_mm, px_per_mm)
    size = (int(round((y1 - y0) * px_per_mm)), int(round((x1 - x0) * px_per_mm)))
    view = cv2.warpPerspective(frame, M @ desk.H, size, flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return view, np.linalg.inv(M)


def _along(points: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Each point's distance along a unit axis (written out: matmul here sets off spurious BLAS warnings on macOS)."""
    return points[:, 0] * axis[0] + points[:, 1] * axis[1]


def _span(values: np.ndarray) -> tuple:
    """The near and far ends of a spread of values, ignoring the outermost 1% either way."""
    lo, hi = np.percentile(values, [1, 99])
    return float(lo), float(hi)


def _heading(axis) -> float:
    """A line's heading from the arm's +x toward its +y, -90 to 90 degrees."""
    return (math.degrees(math.atan2(axis[1], axis[0])) + 90) % 180 - 90


def _lying(shadow: np.ndarray) -> tuple:
    """(middle, length, width, heading) of something flat: its mask is its footprint."""
    middle = shadow.mean(axis=0)
    d = shadow - middle
    values, vectors = np.linalg.eigh(np.cov(d.T))
    major = vectors[:, np.argmax(values)]
    (a0, a1), (b0, b1) = _span(_along(d, major)), _span(_along(d, np.array([-major[1], major[0]])))
    return middle, a1 - a0, b1 - b0, _heading(major)


def _standing(shadow: np.ndarray, desk: DeskFit, height_mm: float) -> tuple:
    """(middle, length, width, heading) of something height_mm tall, from its outline on the table.

    Seen from the camera, its outline reaches from its base's near edge, which is on the table and so where it looks,
    out to its top's far edge, which the table puts k = H / (H - h) times as far from the point under the camera (H the
    camera's height). Along the line from that point through the outline, the near edge is c - r and the far edge
    k (c + r), which gives the middle c and the footprint's half-size r. Across that line, the top is widest, k times
    its size."""
    foot, k = desk.camera[:2], desk.camera[2] / (desk.camera[2] - height_mm)
    out = shadow.mean(axis=0) - foot
    if np.linalg.norm(out) < 1.0:  # right under the camera: the top hides the base
        return _lying(desk.lift(shadow, height_mm))
    radial = out / np.linalg.norm(out)
    side = np.array([-radial[1], radial[0]])
    (near, far), (left, right) = _span(_along(shadow - foot, radial)), _span(_along(shadow - foot, side))
    middle_r, middle_s = (far / k + near) / 2, (left + right) / (2 * k)
    deep, wide = 2 * (middle_r - near), (right - left) / k
    middle = foot + middle_r * radial + middle_s * side
    return middle, max(deep, wide), min(deep, wide), _heading(radial if deep >= wide else side)


def found(mask: np.ndarray, to_arm: np.ndarray, desk: DeskFit, height_mm: float = 0.0, phrase: str = "",
          score: float | None = None) -> Found:
    """What a mask in the top-down view says about the thing: where it stands, how big it is and which way it lies.

    The view is the table, so a mask pixel is where the camera's ray through that bit of the thing meets the table:
    for a flat thing (height 0) its footprint, and for one standing up an outline stretched away from the camera."""
    v, u = np.nonzero(mask)
    if len(u) < 3:
        raise ValueError(f"the mask for {phrase or 'it'} is empty")
    shadow = tagmap.to_mm(to_arm, np.column_stack([u, v]).astype(float))
    middle, length, width, heading = (_standing(shadow, desk, height_mm) if height_mm > 0 else _lying(shadow))
    return Found(phrase, score, middle, float(height_mm), desk.table_z + height_mm, float(length), float(width),
                 heading, float(np.hypot(*middle)), int(len(u)))


def locate(frame: np.ndarray, desk: DeskFit, cfg: dict, segment, phrase: str, height_mm: float = 0.0) -> tuple:
    """SAM's matches for `phrase` on the table as Found, most confident first, with the top-down view SAM saw and its
    Segmentation. `segment(image, phrase)` sends the request: scripts/desk_camera.py binds sam_client.segment to its
    client, config and budget."""
    view, to_arm = top_down(frame, desk, cfg["region_mm"], cfg["px_per_mm"])
    seg = segment(view, phrase)
    hits = [found(i.full_mask(view.shape), to_arm, desk, height_mm, phrase, i.score) for i in seg.instances if i.area]
    hits.sort(key=lambda f: -(f.score or 0.0))
    return hits, view, seg


def find_camera(cfg: dict, touched: dict, capture, indices=range(4)) -> tuple:
    """The desk camera's index and a frame from it: the configured camera if it shows enough touched tags, otherwise
    the first that does. macOS can renumber cameras when one is plugged in, so the desk camera's number can change."""
    cam = cfg["camera"]
    need = max(4, cfg.get("min_tags", 4))
    tried = []
    for index in [cam["index"]] + [i for i in indices if i != cam["index"]]:
        try:
            frame = capture(index, cam["width"], cam["height"])
        except RuntimeError as e:
            tried.append(f"camera {index}: {e}")
            continue
        seen = sorted(set(tagmap.detect(frame, cfg["tags"]["family"])) & set(touched))
        if len(seen) >= need:
            return index, frame
        tried.append(f"camera {index} shows {len(seen)} of them")
    raise RuntimeError(f"no camera shows {need} of the touched desk tags {sorted(touched)}. " + "; ".join(tried))


def draw(view: np.ndarray, to_arm: np.ndarray, touched: dict, hits=()) -> np.ndarray:
    """The top-down view with a ring at each touched tag's center (it should sit on its tag), a cross at the arm's
    base, and each match's middle and long axis."""
    out = view.copy()
    M = np.linalg.inv(to_arm)

    def px(xy):
        return tuple(int(round(v)) for v in tagmap.to_mm(M, np.asarray(xy, float))[0])
    for tag, p in sorted(touched.items()):
        cv2.circle(out, px(p[:2]), 9, (0, 140, 255), 2, cv2.LINE_AA)
        cv2.putText(out, str(tag), px(np.asarray(p[:2]) + (14, -14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1,
                    cv2.LINE_AA)
    base = px((0, 0))
    cv2.drawMarker(out, base, (200, 0, 200), cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
    for n, f in enumerate(hits, 1):
        a = math.radians(f.heading_deg)
        half = np.array([math.cos(a), math.sin(a)]) * f.length_mm / 2
        cv2.line(out, px(f.center - half), px(f.center + half), (0, 200, 0), 2, cv2.LINE_AA)
        cv2.circle(out, px(f.center), 5, (0, 200, 0), -1, cv2.LINE_AA)
        cv2.putText(out, f"{n}", px(f.center + (12, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA)
    return out
