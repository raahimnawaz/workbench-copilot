"""AprilTags on the printed carrier: find them in a camera frame and map the frame onto the carrier, in mm.

Four tag36h11 tags sit at known spots on the sheet the breadboard lies on (config/bench.yaml, printed by
scripts/make_carrier.py). Every frame is mapped through its own tags, so a bumped camera or a refocus
doesn't move the hole grid, and there's no camera calibration to keep. OpenCV's ArUco module reads them.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FAMILIES = {"tag36h11": cv2.aruco.DICT_APRILTAG_36h11}


def detect(image: np.ndarray, family: str = "tag36h11") -> dict:
    """{tag id: its 4 corners in image pixels}, clockwise from the tag's top-left as printed."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(FAMILIES[family]), params)
    corners, ids, _ = detector.detectMarkers(gray)
    return {} if ids is None else {int(i): c.reshape(4, 2).astype(float) for i, c in zip(ids.ravel(), corners)}


def draw(image: np.ndarray, carrier: dict, px_per_mm: float) -> None:
    """Draw the carrier's tags, in place, on an image of the carrier where pixel i covers mm [i, i+1) / px_per_mm.

    Each tag is its 8 x 8 cells: a black border around 6 x 6 data cells. The white paper around it is the
    quiet zone the detector needs, so keep other ink a cell's width away.
    """
    size = carrier["tag_size_mm"]
    dictionary = cv2.aruco.getPredefinedDictionary(FAMILIES[carrier["tag_family"]])
    for tag_id, (cx, cy) in carrier["tags"].items():
        cells = cv2.aruco.generateImageMarker(dictionary, int(tag_id), 8, borderBits=1)
        cell = size / cells.shape[0]
        for r, c in zip(*np.nonzero(cells == 0)):
            x0, y0 = cx - size / 2 + c * cell, cy - size / 2 + r * cell
            image[int(round(y0 * px_per_mm)):int(round((y0 + cell) * px_per_mm)),
                  int(round(x0 * px_per_mm)):int(round((x0 + cell) * px_per_mm))] = 0


def corners_mm(center, size: float) -> np.ndarray:
    """A tag's corners on the carrier, in the order detect() gives them."""
    cx, cy = center
    h = size / 2
    return np.array([[cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h], [cx - h, cy + h]], float)


@dataclass
class CarrierFit:
    H: np.ndarray  # image pixels -> carrier mm
    tags: list  # the tag ids it used
    missing: list  # the carrier's tags it didn't see
    rms_mm: float  # how far the seen corners land from where they're printed


def fit(detections: dict, carrier: dict) -> CarrierFit:
    """Map the frame onto the carrier from every corner of every carrier tag it shows (at least 2 tags)."""
    known = {int(k): v for k, v in carrier["tags"].items()}
    used = sorted(set(detections) & set(known))
    if len(used) < 2:
        seen = sorted(detections) or "none"
        raise ValueError(f"the camera sees {len(used)} of the carrier's tags {sorted(known)} (tags seen: {seen}). "
                         f"It needs at least 2, and all 4 is best: check the sheet is in view and lit.")
    src = np.vstack([detections[i] for i in used])
    dst = np.vstack([corners_mm(known[i], carrier["tag_size_mm"]) for i in used])
    H, _ = cv2.findHomography(src, dst, 0)
    rms = float(np.sqrt(np.mean(np.sum((to_mm(H, src) - dst) ** 2, axis=1))))
    return CarrierFit(H, used, sorted(set(known) - set(used)), rms)


def to_mm(H: np.ndarray, points) -> np.ndarray:
    """Points through a homography: image pixels to carrier mm with H, or back with its inverse."""
    pts = np.asarray(points, float).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def view_transform(region_mm, px_per_mm: float) -> np.ndarray:
    """Carrier mm to the pixels of a top-down view of region_mm = (x0, y0, x1, y1)."""
    x0, y0 = region_mm[:2]
    return np.array([[px_per_mm, 0, -px_per_mm * x0], [0, px_per_mm, -px_per_mm * y0], [0, 0, 1]], float)


def rectify(image: np.ndarray, H: np.ndarray, region_mm, px_per_mm: float) -> tuple:
    """The top-down view of region_mm on the carrier, and the transform from image pixels to its pixels."""
    M = view_transform(region_mm, px_per_mm) @ H
    x0, y0, x1, y1 = region_mm
    size = (int(round((x1 - x0) * px_per_mm)), int(round((y1 - y0) * px_per_mm)))
    return cv2.warpPerspective(image, M, size, flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)), M
