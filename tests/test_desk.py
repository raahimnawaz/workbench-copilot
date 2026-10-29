"""The desk camera: the table mapped through touched tags, the camera placed from that map, heights corrected and
locate's geometry, all on a synthetic pinhole camera; then the tags found in a rendered frame, touch on the fake arm,
and the printed tag sheet."""
import importlib.util
import math

import cv2
import numpy as np
import pytest

from test_check_joint_map import CHAIN, MAP, ROBOT_CFG, FakeLine
from test_joints import REST
from wbc.config import ROOT, load_yaml
from wbc.motion import sts
from wbc.motion.kinematics import solve_task
from wbc.perception import desk, sam_client
from wbc.perception import tags as tagmap

_spec = importlib.util.spec_from_file_location("desk_camera", ROOT / "scripts" / "desk_camera.py")
desk_camera = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(desk_camera)

CFG = load_yaml("desk.yaml")
F, SIZE = 1400.0, (1920, 1080)
TAGS = {20: (150, -300), 21: (380, -250), 22: (420, 0), 23: (380, 250), 24: (150, 300), 25: (250, 0)}  # arm mm
TOUCHED = {i: np.array([x, y, 0.0]) for i, (x, y) in TAGS.items()}
OVERHEAD = (np.array([150.0, 40.0, 820.0]), (260.0, 0.0, 0.0))  # where the camera is, and a table point it faces
OBLIQUE = (np.array([-250.0, 30.0, 600.0]), (260.0, 0.0, 0.0))
CUP = (np.array([300.0, 120.0]), 40.0, 95.0)  # its axis on the table, radius and height, mm


class Camera:
    """A pinhole camera at `center` (arm mm) facing `look`, the top of its image toward the arm's straight ahead."""

    def __init__(self, center, look, f=F, size=SIZE):
        self.C, self.f, self.size = np.asarray(center, float), f, size
        z = np.asarray(look, float) - self.C
        z /= np.linalg.norm(z)
        y = np.array([-1.0, 0.0, 0.0])
        y -= (y @ z) * z
        y /= np.linalg.norm(y)
        self.R = np.vstack([np.cross(y, z), y, z])  # rows: the camera's right, down and forward

    def project(self, points) -> np.ndarray:
        Xc = (np.atleast_2d(np.asarray(points, float)) - self.C) @ self.R.T
        w, h = self.size
        return np.column_stack([self.f * Xc[:, 0] / Xc[:, 2] + (w - 1) / 2,
                                self.f * Xc[:, 1] / Xc[:, 2] + (h - 1) / 2])


def square(center, size=50.0, turn=0.0) -> np.ndarray:
    """A printed square's corners on the table (arm mm, z 0), top-left first as printed, turned `turn` radians. Face
    up, the print's right is the arm's +x and its down the arm's -y."""
    h = size / 2
    c, s = math.cos(turn), math.sin(turn)
    sq = np.array([[-h, h], [h, h], [h, -h], [-h, -h]]) @ np.array([[c, -s], [s, c]]).T + np.asarray(center, float)
    return np.column_stack([sq, np.zeros(4)])


def detections(cam, tags=TAGS) -> dict:
    return {i: cam.project(square(c, turn=0.2 * i)) for i, c in tags.items()}


def rect(center, length, width, heading_deg) -> np.ndarray:
    """A flat bar's outline on the table, arm mm."""
    a = math.radians(heading_deg)
    u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    c = np.asarray(center, float)
    return np.array([c + sx * length / 2 * u + sy * width / 2 * v for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def cup_mask(cam, fit, center=CUP[0], radius=CUP[1], height=CUP[2]) -> np.ndarray:
    """The top-down view's mask of an upright cylinder: its outline in the camera image, warped onto the table."""
    a = np.linspace(0, 2 * math.pi, 120, endpoint=False)
    ring = np.column_stack([center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)])
    points = np.vstack([np.column_stack([ring, np.zeros(len(a))]), np.column_stack([ring, np.full(len(a), height)])])
    image = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    cv2.fillConvexPoly(image, np.round(cv2.convexHull(cam.project(points).astype(np.float32))).astype(np.int32), 255)
    x0, y0, x1, y1 = CFG["region_mm"]
    size = (int(round((y1 - y0) * CFG["px_per_mm"])), int(round((x1 - x0) * CFG["px_per_mm"])))
    M = desk.view_matrix(CFG["region_mm"], CFG["px_per_mm"]) @ fit.H
    return cv2.warpPerspective(image, M, size, flags=cv2.INTER_NEAREST, borderValue=0) > 127


def render(cam, tags=TAGS) -> np.ndarray:
    """A gray frame with each desk tag printed at its spot, with its white border, as the camera sees it."""
    frame = np.full((SIZE[1], SIZE[0], 3), 170, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    for i, c in tags.items():
        cell = 50
        marker = cv2.copyMakeBorder(cv2.aruco.generateImageMarker(dictionary, i, 8 * cell, borderBits=1),
                                    cell, cell, cell, cell, cv2.BORDER_CONSTANT, value=255)
        n = marker.shape[0]
        src = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
        dst = cam.project(square(c, 50.0 * 10 / 8, turn=0.2 * i)).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR), M, SIZE, flags=cv2.INTER_LINEAR)
        inside = cv2.warpPerspective(np.full((n, n), 255, np.uint8), M, SIZE, flags=cv2.INTER_NEAREST) > 0
        frame[inside] = warped[inside]
    return frame


def test_a_tags_center_is_where_its_diagonals_cross_not_its_corners_mean():
    cam = Camera(*OBLIQUE)
    corners = cam.project(square(TAGS[22], turn=0.4))
    truth = cam.project([[*TAGS[22], 0.0]])[0]
    assert np.linalg.norm(desk.center(corners) - truth) < 1e-6
    assert np.linalg.norm(corners.mean(axis=0) - truth) > 0.2


@pytest.mark.parametrize("where", [OVERHEAD, OBLIQUE], ids=["overhead", "oblique"])
def test_the_fit_maps_the_table_into_the_arms_frame_and_places_the_camera(where):
    cam = Camera(*where)
    fit = desk.fit(detections(cam), TOUCHED, SIZE, F)
    spots = np.array([[100, -400, 0], [460, 0, 0], [0, 300, 0], [300, 150, 0]], float)
    assert np.abs(fit.to_table(cam.project(spots)) - spots[:, :2]).max() < 0.01
    assert np.linalg.norm(fit.camera - cam.C) < 0.5
    assert fit.tags == sorted(TAGS) and fit.missing == [] and max(fit.errors.values()) < 0.01


@pytest.mark.parametrize("bad", sorted(TAGS))
@pytest.mark.parametrize("where", [OVERHEAD, OBLIQUE], ids=["overhead", "oblique"])
def test_a_tag_touched_8_mm_off_is_the_one_check_names(where, bad):
    cam = Camera(*where)
    fit = desk.fit(detections(cam), {**TOUCHED, bad: TOUCHED[bad] + (8.0, 0.0, 0.0)}, SIZE, F)
    assert fit.suspect == bad and fit.rest[bad] < 0.01 and fit.rms_mm > 3
    assert fit.errors[bad] == pytest.approx(8.0, abs=0.01)  # placed by the good tags, it lands where it really is


def test_with_every_tag_right_none_is_named_and_with_4_none_can_be_checked():
    cam = Camera(*OVERHEAD)
    rng = np.random.default_rng(1)
    noisy = {i: p + np.append(rng.normal(0, 0.3, 2), 0.0) for i, p in TOUCHED.items()}  # touches within ~0.5 mm
    fit = desk.fit(detections(cam), noisy, SIZE, F)
    assert fit.rms_mm < 2
    four = desk.fit(detections(cam, {i: TAGS[i] for i in (20, 21, 23, 24)}), TOUCHED, SIZE, F)
    assert four.errors == {} and four.rest == {} and four.rms_mm is None and four.suspect is None
    assert four.missing == [22, 25]


def test_a_frame_showing_too_few_tags_is_refused_saying_what_it_saw():
    with pytest.raises(ValueError, match=r"sees 3 of the 6 touched desk tags .*\(tags in view: \[20, 21, 22\]\)"):
        desk.fit(detections(Camera(*OVERHEAD), {i: TAGS[i] for i in (20, 21, 22)}), TOUCHED, SIZE, F)


@pytest.mark.parametrize("where", [OVERHEAD, OBLIQUE], ids=["overhead", "oblique"])
def test_lift_puts_a_raised_point_back_over_its_spot(where):
    cam = Camera(*where)
    fit = desk.fit(detections(cam), TOUCHED, SIZE, F)
    rim = np.array([[300.0, 120.0, 95.0]])
    shadow = fit.to_table(cam.project(rim))[0]
    assert np.linalg.norm(shadow - rim[0, :2]) > 20  # taken as on the table, it reads well out
    assert np.linalg.norm(fit.lift(shadow, 95) - rim[0, :2]) < 0.05


def test_a_focal_length_5_percent_off_moves_a_cup_rim_by_only_a_few_mm():
    cam = Camera(*OVERHEAD)
    fit = desk.fit(detections(cam), TOUCHED, SIZE, F * 1.05)
    rim = np.array([[300.0, 120.0, 95.0]])
    assert np.linalg.norm(fit.lift(fit.to_table(cam.project(rim))[0], 95) - rim[0, :2]) < 3


def test_the_focal_length_is_read_from_an_oblique_view_but_not_from_one_looking_straight_down():
    oblique = desk.fit(detections(Camera(*OBLIQUE)), TOUCHED, SIZE, F)
    assert desk.focal_from(oblique.H, SIZE) == pytest.approx(F, rel=0.01)
    assert desk.tilt_deg(oblique.H, SIZE, F) > 30
    overhead = desk.fit(detections(Camera(*OVERHEAD)), TOUCHED, SIZE, F)
    assert desk.focal_from(overhead.H, SIZE) is None


def test_the_top_down_view_maps_back_into_the_arms_frame():
    cam = Camera(*OVERHEAD)
    fit = desk.fit(detections(cam), TOUCHED, SIZE, F)
    frame = np.full((SIZE[1], SIZE[0], 3), 255, np.uint8)
    spot = np.array([320.0, -140.0, 0.0])
    cv2.circle(frame, tuple(int(round(v)) for v in cam.project([spot])[0]), 5, (0, 0, 0), -1)
    view, to_arm = desk.top_down(frame, fit, CFG["region_mm"], CFG["px_per_mm"])
    assert view.shape[:2] == (580, 900)
    v, u = np.nonzero(view[:, :, 0] < 128)
    assert np.linalg.norm(tagmap.to_mm(to_arm, [[u.mean(), v.mean()]])[0] - spot[:2]) < 1.0


def test_a_flat_things_middle_size_and_heading():
    fit = desk.fit(detections(Camera(*OVERHEAD)), TOUCHED, SIZE, F)
    M = desk.view_matrix(CFG["region_mm"], CFG["px_per_mm"])
    mask = np.zeros((580, 900), np.uint8)
    cv2.fillConvexPoly(mask, np.round(tagmap.to_mm(M, rect((280, -60), 150, 22, 30))).astype(np.int32), 1)
    f = desk.found(mask.astype(bool), np.linalg.inv(M), fit, 0.0, "screwdriver", 0.9)
    assert np.linalg.norm(f.center - (280, -60)) < 1
    assert f.length_mm == pytest.approx(150, abs=4) and f.width_mm == pytest.approx(22, abs=3)
    assert f.heading_deg == pytest.approx(30, abs=1) and f.top_z == pytest.approx(0)
    assert f.reach_mm == pytest.approx(math.hypot(280, 60), abs=1)
    assert f.summary()["x_mm"] == pytest.approx(280, abs=1)


@pytest.mark.parametrize("where", [OVERHEAD, OBLIQUE], ids=["overhead", "oblique"])
def test_an_upright_cups_middle_comes_from_its_outline_and_its_height(where):
    cam = Camera(*where)
    fit = desk.fit(detections(cam), TOUCHED, SIZE, F)
    mask = cup_mask(cam, fit)
    to_arm = np.linalg.inv(desk.view_matrix(CFG["region_mm"], CFG["px_per_mm"]))
    told = desk.found(mask, to_arm, fit, CUP[2], "cup")
    flat = desk.found(mask, to_arm, fit, 0.0, "cup")
    off, flat_off = np.linalg.norm(told.center - CUP[0]), np.linalg.norm(flat.center - CUP[0])
    assert off < 3 and flat_off > 10  # without its height, its middle reads out toward where the camera looks
    # A little under its 80 mm: the outline's outermost 1% either way is trimmed, and a circle's edge is thin
    assert told.length_mm == pytest.approx(76, abs=4) and told.width_mm == pytest.approx(76, abs=6)
    assert told.top_z == pytest.approx(95)


def test_a_wrong_height_moves_a_cups_middle_by_a_fraction_of_the_error():
    cam = Camera(*OBLIQUE)
    fit = desk.fit(detections(cam), TOUCHED, SIZE, F)
    to_arm = np.linalg.inv(desk.view_matrix(CFG["region_mm"], CFG["px_per_mm"]))
    guessed = desk.found(cup_mask(cam, fit), to_arm, fit, 110.0, "cup")  # guessed 110 mm for a 95 mm cup
    assert np.linalg.norm(guessed.center - CUP[0]) < 15


def test_locate_returns_the_most_confident_match_first_and_skips_empty_masks():
    cam = Camera(*OVERHEAD)
    fit = desk.fit(detections(cam), TOUCHED, SIZE, F)
    frame = np.full((SIZE[1], SIZE[0], 3), 255, np.uint8)
    mask = cup_mask(cam, fit)
    v, u = np.nonzero(mask)
    box = (int(u.min()), int(v.min()), int(u.max()) + 1, int(v.max()) + 1)
    crop = mask[box[1]:box[3], box[0]:box[2]]

    def segment(image, phrase):
        assert image.shape[:2] == mask.shape
        return sam_client.Segmentation(phrase, [
            sam_client.Instance(phrase, "a", box, crop, 0.5),
            sam_client.Instance(phrase, "b", box, np.zeros_like(crop), 0.99),
            sam_client.Instance(phrase, "c", box, crop, 0.9)], [], "")
    hits, view, seg = desk.locate(frame, fit, CFG, segment, "cup", CUP[2])
    assert [h.score for h in hits] == [0.9, 0.5] and view.shape[:2] == mask.shape
    assert np.linalg.norm(hits[0].center - CUP[0]) < 3


def test_tags_found_in_a_rendered_frame_map_the_table_and_find_the_desk_camera():
    cam = Camera(*OVERHEAD)
    frame = render(cam)
    seen = tagmap.detect(frame)
    assert sorted(seen) == sorted(TAGS)
    fit = desk.fit(seen, TOUCHED, SIZE, F)
    spots = np.array([[100, -400, 0], [460, 0, 0], [300, 150, 0]], float)
    assert np.abs(fit.to_table(cam.project(spots)) - spots[:, :2]).max() < 1.0
    assert fit.camera[2] == pytest.approx(cam.C[2], rel=0.03) and max(fit.errors.values()) < 1.0

    board = np.full_like(frame, 170)  # the board camera: no desk tags in its view
    configured = CFG["camera"]["index"]
    other = 1 if configured == 0 else 0  # macOS renumbered them: the desk camera is the other one now
    shots = {configured: board, other: frame}

    def capture(index, width, height):
        if index not in shots:
            raise RuntimeError(f"camera {index} didn't open")
        return shots[index]
    index, got = desk.find_camera(CFG, TOUCHED, capture)
    assert index == other and got is frame
    with pytest.raises(RuntimeError, match="no camera shows 4 of the touched desk tags"):
        desk.find_camera(CFG, TOUCHED, lambda index, width, height: board)


def test_the_tag_sheet_prints_every_tag_at_its_size_and_they_read_back():
    img = desk_camera.sheet(CFG)
    seen = tagmap.detect(img, CFG["tags"]["family"])
    assert sorted(seen) == CFG["tags"]["ids"]
    s = desk_camera.DPI / 25.4
    for corners in seen.values():
        assert np.linalg.norm(corners[1] - corners[0]) / s == pytest.approx(CFG["tags"]["size_mm"], abs=0.5)


def pose_at(spot_mm) -> dict:
    """Raw positions that put the tool point on `spot_mm` (arm frame), the gripper as near straight down as it goes."""
    for degrees in (90, 80, 70, 60, 50, 40):
        res = solve_task(CHAIN, *(np.asarray(spot_mm, float) / 1000), math.radians(degrees), 0.0)
        if res.ok:
            return {**REST, **MAP.raw(res.q)}
    raise AssertionError(f"no pose reaches {spot_mm}")


def press(line, raw):
    """A reply that moves the arm by hand to `raw`, then presses Enter."""
    def act():
        line.pose(raw)
        return ""
    return act


def touch_session(line, ids, path, replies=()):
    replies, said, record = list(replies), [], {}

    def ask(prompt):
        said.append(prompt)
        if not replies:
            raise EOFError
        reply = replies.pop(0)
        return reply() if callable(reply) else reply
    bus = sts.Bus("fake", serial_factory=lambda port, **kwargs: line)
    ok = desk_camera.run_touch(bus, ROBOT_CFG, CFG, ids, path, record, ask=ask, say=said.append)
    return ok, record, "\n".join(said)


SPOTS = {20: (200, -200, 0), 21: (320, -120, 0), 22: (340, 60, 0), 23: (230, 220, 0), 24: (130, 140, 0)}  # reachable


def test_touch_saves_each_tags_center_in_the_arms_frame_and_writes_nothing_to_the_servos(tmp_path):
    line, path = FakeLine(), tmp_path / "desk_tags.json"
    ok, record, said = touch_session(line, list(SPOTS), path, [press(line, pose_at(p)) for p in SPOTS.values()])
    assert ok and line.written() == [] and "Saved 5 tags" in said
    saved = desk.load_touched(path)
    assert sorted(saved) == sorted(SPOTS)
    for tag, p in SPOTS.items():
        assert np.linalg.norm(saved[tag] - p) < 2.0
    assert set(record["touched"]) == {str(t) for t in SPOTS}


def test_touch_with_fewer_than_4_tags_saves_nothing(tmp_path):
    line, path = FakeLine(), tmp_path / "desk_tags.json"
    ok, _, said = touch_session(line, [20, 21, 22, 23], path, [press(line, pose_at(SPOTS[20])), "s", "s", "q"])
    assert not ok and "1 tag touched" in said and "Nothing saved" in said and not path.exists()


def test_touch_refuses_while_a_motor_is_on(tmp_path):
    line = FakeLine(torque=("elbow_flex",))
    ok, _, said = touch_session(line, [20, 21, 22, 23], tmp_path / "desk_tags.json")
    assert not ok and "Torque is on for the elbow" in said and line.written() == []
