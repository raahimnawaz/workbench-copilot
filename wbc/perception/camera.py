"""The bench camera: find it and grab a still.

macOS asks once whether the terminal may use the camera, and a terminal without that permission can't open
it. This app's terminal has neither camera nor mic permission, so capture from Terminal.app, as with voice.
"""
from __future__ import annotations

import collections
import json
import subprocess
import threading
import time

import cv2
import numpy as np

NOT_ALLOWED = ("If macOS hasn't let this terminal use the camera, allow it under System Settings → "
               "Privacy & Security → Camera, or run from Terminal.app.")
FRESH_S = 0.4  # seconds of a kept-open camera's newest frames that its still is the sharpest of
STALLED_S = 2.0  # seconds a kept-open camera may go without a frame before a still from it fails


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


def sharpness(frame: np.ndarray) -> float:
    """How in focus a frame is: the variance of the Laplacian on a half-size gray copy. Higher is sharper."""
    gray = cv2.cvtColor(cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def capture(index: int = 0, width: int = 1920, height: int = 1080, settle_s: float = 1.5,
            pick_s: float = 1.5) -> np.ndarray:
    """One BGR still: after `settle_s` for exposure, the sharpest frame seen over the next `pick_s`.

    The C920 refocuses each time it's opened, and the frame at 1.5 s can still be soft, so this keeps
    looking while the autofocus settles instead of taking whatever frame arrives first.
    """
    cap = open_camera(index, width, height)
    best, best_score, last = None, -1.0, None
    t0 = time.monotonic()
    try:
        for _ in range(900):  # 30 s at 30 fps: a camera that stalls can't hang the caller
            ok, f = cap.read()
            if not ok or f is None:
                continue
            last = f
            t = time.monotonic() - t0
            if t >= settle_s:
                score = sharpness(f)
                if score > best_score:
                    best, best_score = f, score
                if t >= settle_s + pick_s:
                    break
    finally:
        cap.release()
    frame = best if best is not None else last
    if frame is None:
        raise RuntimeError(f"camera {index} opened but sent no frames. {NOT_ALLOWED}")
    if not frame.any():
        raise RuntimeError(f"camera {index} sent an all-black frame. {NOT_ALLOWED}")
    return frame


class Stream:
    """A camera kept open, a thread reading every frame it sends, so a still from it needn't wait the 3 s that
    capture() gives exposure and focus to settle each time it opens a camera: it's the sharpest of the frames from the
    last FRESH_S, once one has come since it was asked for. The first waits settle_s + pick_s from opening, as capture()
    does. close() lets the camera go."""

    def __init__(self, index: int, width: int = 1920, height: int = 1080, settle_s: float = 1.5,
                 pick_s: float = 1.5):
        self.index, self._size, self._wait = index, (width, height), settle_s + pick_s
        self._frames = collections.deque()  # (when, frame) for the last FRESH_S
        self._lock = threading.Lock()
        self._start()

    def _start(self) -> None:
        self._cap = open_camera(self.index, *self._size)
        self._ready = time.monotonic() + self._wait
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._read, args=(self._cap, self._done), name=f"camera {self.index}",
                                        daemon=True)
        self._thread.start()

    def _read(self, cap, done) -> None:
        try:
            while not done.is_set():
                ok, f = cap.read()
                if not ok or f is None:
                    time.sleep(0.01)
                    continue
                now = time.monotonic()
                with self._lock:
                    self._frames.append((now, f))
                    while now - self._frames[0][0] > FRESH_S:
                        self._frames.popleft()
        finally:
            cap.release()

    def capture(self, index: int, width: int = 1920, height: int = 1080) -> np.ndarray:
        """A still from camera `index`, as capture() gives one: this camera's from its stream, another's from
        capture()."""
        if index != self.index:
            return capture(index, width, height)
        asked, restarted = time.monotonic(), False
        while True:
            now = time.monotonic()
            with self._lock:
                recent = [(t, f) for t, f in self._frames if t >= now - FRESH_S]
            if now >= self._ready and recent and recent[-1][0] > asked:
                break
            if now > max(asked, self._ready) + STALLED_S:
                if not restarted:  # a stream that stalls (Nov 4, 12:16) opens again once before giving up
                    self.close()
                    self._start()
                    asked, restarted = time.monotonic(), True
                    continue
                raise RuntimeError(f"camera {index} stopped sending frames. {NOT_ALLOWED}")
            time.sleep(0.01)
        frame = max((f for _, f in recent), key=sharpness)
        if not frame.any():
            raise RuntimeError(f"camera {index} sent an all-black frame. {NOT_ALLOWED}")
        return frame.copy()

    def close(self) -> None:
        self._done.set()
        self._thread.join(timeout=2.0)


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
