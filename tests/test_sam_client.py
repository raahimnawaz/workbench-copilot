import cv2
import numpy as np
import pytest
from fakes import FakeSam, sam_reply

from wbc.agent.budget import Budget, BudgetExceeded
from wbc.perception import sam_client

EYES = {"model": "sam-3.1", "timeout_s": 30}


def budget(limit=10):
    return Budget(limit, {"sam-3.1": {"per_1k_images": 2.50}})


def band(rows, cols, width_px=3):
    """A diagonal band across a rows x cols box: the shape of a wire's mask."""
    yy, xx = np.mgrid[0:rows, 0:cols]
    d = np.abs((rows - 1) * xx - (cols - 1) * yy) / np.hypot(rows - 1, cols - 1)
    return (d <= width_px / 2).astype(np.uint8)


def photo(rows=120, cols=200):
    return np.full((rows, cols, 3), 200, np.uint8)


def wire_reply():
    return sam_reply([((10, 40, 70, 70), band(30, 60), 0.96)], 200, 120)


def test_parse_reads_boxes_masks_and_scores():
    m0, m1 = band(30, 60), band(20, 25)
    text = sam_reply([((10, 40, 70, 70), m0, 0.96), ((100, 5, 125, 25), m1, 0.5)], 200, 120)
    instances, problems = sam_client.parse(text, "jumper wire")
    assert problems == []
    assert [i.box for i in instances] == [(10, 40, 70, 70), (100, 5, 125, 25)]
    assert np.array_equal(instances[0].mask, m0.astype(bool)) and np.array_equal(instances[1].mask, m1.astype(bool))
    assert instances[0].score == pytest.approx(0.96) and instances[0].phrase == "jumper wire"
    assert instances[0].area == m0.sum()
    full = instances[0].full_mask((120, 200))
    assert full.sum() == m0.sum() and full[40:70, 10:70].sum() == m0.sum()


def test_no_match_is_an_empty_result():
    assert sam_client.parse("", "jumper wire") == ([], [])


def test_a_mask_that_fails_to_decode_is_reported_and_the_rest_kept():
    text = sam_reply([((10, 40, 70, 70), band(30, 60), 0.9), ((100, 5, 125, 25), band(20, 25), 0.8)], 200, 120)
    head, marker, tail = text.rpartition("data=20,25,!")
    instances, problems = sam_client.parse(head + marker + "AAAAA" + tail)  # breaks the payload's length prefix
    assert [i.box for i in instances] == [(10, 40, 70, 70)]
    assert len(problems) == 1 and problems[0].startswith("invalid_mask_payload")


def test_text_instead_of_records_is_reported():
    instances, problems = sam_client.parse("I can't segment that.")
    assert instances == [] and problems[0].startswith("text instead of records")


def test_a_mask_smaller_than_its_box_is_scaled_to_the_box():
    text = sam_reply([((0, 0, 40, 20), np.ones((10, 20), np.uint8), None)], 100, 100)
    (inst,), _ = sam_client.parse(text)
    assert inst.mask.shape == (20, 40) and inst.mask.all() and inst.score is None


def test_segment_sends_one_phrase_with_the_photo_and_charges_one_image():
    client, b = FakeSam({"jumper wire": wire_reply()}), budget()
    seg = sam_client.segment(client, EYES, b, photo(), "jumper wire")
    (req,) = client.requests
    text_part, image_part = req["input"][0]["content"]
    assert req["model"] == "sam-3.1" and text_part == {"type": "input_text", "text": "jumper wire"}
    assert image_part["type"] == "input_image" and image_part["image_url"].startswith("data:image/jpeg;base64,")
    assert req["metadata"] == {"mask_encoding": "one_bit", "include_confidence": "true"}
    assert len(seg.instances) == 1 and seg.raw.startswith("<0f>") and seg.problems == []
    assert b.spent == pytest.approx(0.0025)


def test_segment_stops_at_the_budget_before_any_request():
    client = FakeSam({"jumper wire": wire_reply()})
    with pytest.raises(BudgetExceeded):
        sam_client.segment(client, EYES, budget(limit=0), photo(), "jumper wire")
    assert client.requests == []


def test_a_failed_request_raises_and_costs_nothing():
    b = budget()
    with pytest.raises(RuntimeError, match="boom"):
        sam_client.segment(FakeSam({"jumper wire": RuntimeError("boom")}), EYES, b, photo(), "jumper wire")
    assert b.spent == 0


def test_segment_many_sends_each_phrase_once_with_the_same_photo():
    client, b = FakeSam({"jumper wire": wire_reply(), "microcontroller board": ""}), budget()
    results = sam_client.segment_many(client, EYES, b, photo(),
                                      ["jumper wire", "microcontroller board", "jumper wire"])
    assert sorted(results) == ["jumper wire", "microcontroller board"]
    assert len(results["jumper wire"].instances) == 1 and results["microcontroller board"].instances == []
    assert len(client.requests) == 2
    assert len({r["input"][0]["content"][1]["image_url"] for r in client.requests}) == 1
    assert b.spent == pytest.approx(0.005)


def test_data_url_keeps_png_as_png(tmp_path):
    png = cv2.imencode(".png", photo())[1].tobytes()
    path = tmp_path / "board.png"
    path.write_bytes(png)
    assert sam_client.data_url(png).startswith("data:image/png;base64,")
    assert sam_client.data_url(path).startswith("data:image/png;base64,")


def test_overlay_tints_the_masks_and_leaves_the_rest_alone():
    image = photo()
    instances, _ = sam_client.parse(wire_reply())
    out = sam_client.overlay(image, instances)
    assert out.shape == image.shape and not np.array_equal(out[40:70, 10:70], image[40:70, 10:70])
    assert np.array_equal(out[100:, 150:], image[100:, 150:])
    assert np.array_equal(image, photo())  # the photo itself is untouched


def load_probe():
    import importlib.util

    from wbc.config import ROOT

    spec = importlib.util.spec_from_file_location("sam_probe", ROOT / "scripts" / "sam_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_segments_a_crop_saves_the_run_and_replays_it_free(tmp_path, monkeypatch, capsys):
    probe = load_probe()
    path = tmp_path / "photo.png"
    cv2.imwrite(str(path), np.vstack([photo(), photo()]))  # the wire sits in the top half, which is cropped out
    client, b = FakeSam({"jumper wire": wire_reply()}), budget()
    monkeypatch.setattr(probe, "OUT", tmp_path / "vision")
    monkeypatch.setattr(probe.config, "meta_client", lambda models: client)
    monkeypatch.setattr(probe.Budget, "from_config", classmethod(lambda cls, models, root: b))

    assert probe.main(["segment", str(path), "--phrase", "jumper wire", "--crop", "0,0,200,120"]) == 0
    out = capsys.readouterr().out
    assert "1 request(s) to sam-3.1 on a 200x120 image, about $0.0025" in out and '"jumper wire": 1 match ' in out
    (run,) = (tmp_path / "vision").glob("sam-*.json")
    assert run.with_suffix(".png").exists() and len(client.requests) == 1
    assert b.spent == pytest.approx(0.0025)

    assert probe.main(["replay", str(run)]) == 0
    assert '"jumper wire": 1 match ' in capsys.readouterr().out
    assert run.with_name(run.stem + "-replay.png").exists()
    assert len(client.requests) == 1 and b.spent == pytest.approx(0.0025)  # replaying costs nothing
