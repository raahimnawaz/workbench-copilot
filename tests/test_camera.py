import time

import cv2
import numpy as np
import pytest

from wbc.perception import camera


def _frames():
    """A stripe pattern, blurred except for one frame, as a camera's autofocus might deliver it."""
    sharp = np.zeros((120, 160, 3), np.uint8)
    sharp[:, ::8] = 255
    soft = cv2.GaussianBlur(sharp, (0, 0), 3)
    return [soft] * 5 + [sharp] + [soft] * 5


class FakeCap:
    def __init__(self, frames):
        self.frames = list(frames)

    def read(self):
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def release(self):
        pass


def test_sharpness_ranks_a_sharp_frame_above_a_soft_one():
    frames = _frames()
    assert camera.sharpness(frames[5]) > 10 * camera.sharpness(frames[0])


def test_capture_keeps_the_sharpest_frame(monkeypatch):
    frames = _frames()
    monkeypatch.setattr(camera, "open_camera", lambda index, width, height: FakeCap(frames))
    got = camera.capture(0, 160, 120, settle_s=0.0, pick_s=60.0)  # look at every frame
    assert np.array_equal(got, frames[5])


class SendingCap:
    """A camera sending a frame every 10 ms, every tenth one sharp, until it stalls (when `stalled`)."""

    def __init__(self, stalled=False):
        self.sharp, self.soft = _frames()[5], _frames()[0]
        self.n, self.stalled, self.released = 0, stalled, False

    def read(self):
        time.sleep(0.01)
        if self.stalled:
            return False, None
        self.n += 1
        return True, self.sharp if self.n % 10 == 0 else self.soft

    def release(self):
        self.released = True


def test_a_kept_open_camera_waits_to_settle_once_then_gives_the_sharpest_recent_frame_at_once(monkeypatch):
    cap = SendingCap()
    monkeypatch.setattr(camera, "open_camera", lambda index, width, height: cap)
    stream = camera.Stream(0, 160, 120, settle_s=0.2, pick_s=0.1)
    try:
        t0 = time.monotonic()
        assert np.array_equal(stream.capture(0, 160, 120), cap.sharp)
        assert time.monotonic() - t0 >= 0.25  # settle_s + pick_s from opening, as capture() waits
        t0 = time.monotonic()
        assert np.array_equal(stream.capture(0, 160, 120), cap.sharp)
        assert time.monotonic() - t0 < 0.2  # then only for a frame sent since it was asked
    finally:
        stream.close()
    assert cap.released


def test_a_kept_open_camera_leaves_the_other_cameras_to_capture(monkeypatch):
    monkeypatch.setattr(camera, "open_camera", lambda index, width, height: SendingCap())
    monkeypatch.setattr(camera, "capture", lambda index, width, height: ("captured", index))
    stream = camera.Stream(0, 160, 120, settle_s=0.0, pick_s=0.0)
    try:
        assert stream.capture(1, 160, 120) == ("captured", 1)
    finally:
        stream.close()


def test_a_still_from_a_kept_open_camera_that_stopped_sending_fails(monkeypatch):
    monkeypatch.setattr(camera, "open_camera", lambda index, width, height: SendingCap(stalled=True))
    monkeypatch.setattr(camera, "STALLED_S", 0.2)
    stream = camera.Stream(0, 160, 120, settle_s=0.0, pick_s=0.0)
    try:
        with pytest.raises(RuntimeError, match="camera 0 stopped sending frames"):
            stream.capture(0, 160, 120)
    finally:
        stream.close()
