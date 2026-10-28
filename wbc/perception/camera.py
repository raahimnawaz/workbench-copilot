"""The bench camera: find it and grab a still.

macOS asks once whether the terminal may use the camera, and a terminal without that permission can't open
it. This app's terminal has neither camera nor mic permission, so capture from Terminal.app, as with voice.
"""
from __future__ import annotations

import json
import subprocess
import time

import cv2
import numpy as np

NOT_ALLOWED = ("If macOS hasn't let this terminal use the camera, allow it under System Settings → "
               "Privacy & Security → Camera, or run from Terminal.app.")


def camera_names() -> list:
    """Camera names macOS reports, such as "HD Pro Webcam C920". Their order needn't match OpenCV's indices."""
    try:
        out = subprocess.run(["system_profiler", "-json", "SPCameraDataType"], capture_output=True, text=True,
                             timeout=30)
        return [c.get("_name", "?") for c in json.loads(out.stdout).get("SPCameraDataType", [])]
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def open_camera(index: int, width: int, height: int):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"camera {index} didn't open. {NOT_ALLOWED}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def capture(index: int = 0, width: int = 1920, height: int = 1080, settle_s: float = 1.5) -> np.ndarray:
    """One BGR still, taken after `settle_s` of frames so exposure and focus can settle."""
    cap = open_camera(index, width, height)
    frame, t_end = None, time.monotonic() + settle_s
    try:
        for _ in range(600):  # 20 s at 30 fps: a camera that stalls can't hang the caller
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
                if time.monotonic() >= t_end:
                    break
    finally:
        cap.release()
    if frame is None:
        raise RuntimeError(f"camera {index} opened but sent no frames. {NOT_ALLOWED}")
    if not frame.any():
        raise RuntimeError(f"camera {index} sent an all-black frame. {NOT_ALLOWED}")
    return frame


def list_cameras(max_index: int = 4) -> list:
    """Each OpenCV camera index that opens, with the frame size it gives when asked for 1920x1080."""
    found = []
    for index in range(max_index):
        try:
            cap = open_camera(index, 1920, 1080)
        except RuntimeError:
            continue
        ok, frame = cap.read()
        cap.release()
        found.append({"index": index, "size": None if not ok else (frame.shape[1], frame.shape[0])})
    return found
