import cv2
import numpy as np

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
