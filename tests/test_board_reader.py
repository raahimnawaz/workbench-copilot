"""The board reader on a synthetic bench photo: tags, the top-down view, the hole grid, wire ends, the netlist
and the diff against pins.h, with SAM's wire masks replayed by a fake."""
import cv2
import numpy as np
import pytest
from board_scene import PARTS, WIRES, bar, bench_config, camera_homography, photo, wire_masks
from fakes import FakeSam, sam_reply

from wbc.agent.budget import Budget
from wbc.config import ROOT
from wbc.perception import breadboard, netlist, reader, sam_client, tags

EYES = {"model": "sam-3.1"}


def board():
    (b,) = breadboard.boards_from(bench_config(), PARTS)
    return b


def sam_for(bench, actual_board, wires=WIRES):
    masks, size = wire_masks(bench, actual_board, wires)
    return FakeSam({bench["reader"]["wire_phrase"]: sam_reply(masks, *size)})


def budget():
    return Budget(10, {"sam-3.1": {"per_1k_images": 2.50}})


# --- tags and the top-down view ---

def test_tags_map_the_frame_onto_the_carrier():
    frame, _, bench = photo()
    fit = tags.fit(tags.detect(frame), bench["carrier"])
    assert fit.tags == [10, 11, 12, 13] and fit.missing == [] and fit.rms_mm < 0.2
    on_board = np.array([[60, 80], [220, 80], [60, 135], [220, 135], [140, 108]], float)
    seen = tags.to_mm(camera_homography(), on_board)  # where the camera sees them
    assert np.abs(tags.to_mm(fit.H, seen) - on_board).max() < 0.05


def test_two_tags_are_enough_and_one_is_not():
    frame, _, bench = photo()
    found = tags.detect(frame)
    for tag_id in (11, 12):  # cover two tags with white paper
        cv2.fillConvexPoly(frame, np.round(found[tag_id]).astype(np.int32), (255, 255, 255))
    fit = tags.fit(tags.detect(frame), bench["carrier"])
    assert fit.tags == [10, 13] and fit.missing == [11, 12]
    cv2.fillConvexPoly(frame, np.round(found[13]).astype(np.int32), (255, 255, 255))
    with pytest.raises(ValueError, match="sees 1 of the carrier's tags"):
        tags.fit(tags.detect(frame), bench["carrier"])


# --- breadboard geometry ---

def test_holes_strips_and_rails():
    b = board()
    assert np.allclose(b.hole("a1"), b.a1) and np.allclose(b.hole("j1") - b.hole("a1"), 11 * 2.54 * np.asarray(b.v))
    hit = b.locate(b.hole("f10") + [0.4, 0.3])
    assert (hit.hole, hit.node, hit.row) == ("f10", "10f-j", 10) and hit.distance_mm == pytest.approx(0.5)
    assert b.locate(b.hole("c41")).node == "41a-e"
    rail = b.locate(b.at(30.3, PARTS["breadboard"]["rails"]["a+"]))
    assert (rail.hole, rail.node, rail.row) == ("a+", "rail a+", None) and rail.distance_mm < 1e-9
    assert b.locate(b.at(30, 5.5)).hole in ("e30", "f30")  # mid-channel goes to the nearer side
    length, width = np.ptp(b.body() @ np.array([np.asarray(b.u), np.asarray(b.v)]).T, axis=0)
    assert (length, width) == pytest.approx(tuple(PARTS["breadboard"]["body_mm"]))


def test_several_boards_name_their_holes():
    bench = bench_config()
    bench["boards"]["right"] = {"a1_mm": [60.96, 160.0], "rows_toward": [1, 0], "letters_toward": [0, -1]}
    left, right = breadboard.boards_from(bench, PARTS)
    assert left.locate(left.hole("c5")).hole == "main:c5" and right.locate(right.hole("c5")).node == "right:5a-e"


def test_registration_finds_a_board_off_its_outline():
    frame, actual, bench = photo(shift_mm=(0.9, -0.7), turn_deg=-0.6)
    _, (configured,), area, view = reader.locate_boards(frame, bench, PARTS)
    found, report = breadboard.register(configured, view, area, bench["reader"]["px_per_mm"])
    assert "problem" not in report and report["contrast"] > 3
    assert np.abs(found.holes()[1] - actual.holes()[1]).max() < 0.15


def test_registration_finds_how_much_larger_the_board_looks():
    # Oct 28, the real bench: the board's top stands about 10 mm above the sheet, so it looked 1.9% larger, and an
    # unscaled grid fitted the middle of the board but sat a millimeter off the holes at both ends.
    frame, actual, bench = photo(shift_mm=(-0.3, 0.5), turn_deg=0.3, scale=1.03)
    _, (configured,), area, view = reader.locate_boards(frame, bench, PARTS)
    found, report = breadboard.register(configured, view, area, bench["reader"]["px_per_mm"])
    assert report["scale"] == pytest.approx(1.03, abs=0.003) and "problem" not in report
    assert np.abs(found.holes()[1] - actual.holes()[1]).max() < 0.15
    assert found.locate(actual.hole("j63")).hole == "j63" and found.locate(actual.hole("a1")).distance_mm < 0.15


def test_reader_reads_a_board_that_looks_larger_than_life():
    frame, actual, bench = photo(shift_mm=(0.4, -0.3), turn_deg=-0.2, scale=1.03)
    result = reader.read_board(frame, bench, PARTS, sam_for(bench, actual), EYES, budget(), ROOT)
    assert {tuple(sorted(w["ends"])) for w in result["wires"]} == {tuple(sorted((a, z))) for a, z, _ in WIRES}
    assert result["mismatches"][0]["hole"] == "a22" and result["endpoint_residual_mm"]["max"] < 0.6


def test_registration_without_holes_keeps_the_printed_position():
    b = board()
    area = breadboard.region([b], 5)
    blank = np.full((360, 970, 3), 230, np.uint8)
    found, report = breadboard.register(b, blank, area, 5.5)
    assert found == b and "couldn't find its hole grid" in report["problem"]


# --- wire ends ---

def view_of(b, wires_mm, s=5.5, thickness_mm=2.0, curve_mm=0.0):
    """Masks for wires drawn straight or bowed between holes ("c12") or (row, step) points, as SAM instances
    in a top-down view of b. Their round caps reach half a wire's width past each end."""
    area = breadboard.region([b], 5)
    size = (int(round((area[2] - area[0]) * s)), int(round((area[3] - area[1]) * s)))
    instances = []
    for i, (a, z) in enumerate(wires_mm):
        pa, pz = (b.hole(p) if isinstance(p, str) else b.at(*p) for p in (a, z))
        d = (pz - pa) / np.linalg.norm(pz - pa)
        t = np.linspace(0, 1, 60)[:, None]
        pts = pa + t * (pz - pa) + 4 * t * (1 - t) * curve_mm * np.array([-d[1], d[0]])
        full = np.zeros(size[::-1], np.uint8)
        cv2.polylines(full, [np.round((pts - area[:2]) * s * 16).astype(np.int32)], False, 1,
                      max(1, int(round(thickness_mm * s))), cv2.LINE_8, 4)
        ys, xs = np.nonzero(full)
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        instances.append(sam_client.Instance("wire", str(i), box, full[box[1]:box[3], box[0]:box[2]] > 0))
    return instances, np.full((size[1], size[0], 3), 200, np.uint8), area


def test_a_bowed_wire_still_lands_on_its_holes():
    b = board()
    instances, view, area = view_of(b, [("c12", "h30"), ("a40", "b55")], curve_mm=6)
    wires = netlist.trace(instances, view, area, 5.5, [b], {"snap_mm": 1.2, "tip_inset_mm": 1.0})  # the caps
    assert sorted(tuple(sorted(e.hole for e in w.ends)) for w in wires) == [("a40", "b55"), ("c12", "h30")]
    assert all(e.snapped for w in wires for e in w.ends) and not any(w.problems for w in wires)
    assert wires[0].width_mm == pytest.approx(2.0, abs=0.4)


def test_tip_inset_moves_the_ends_back_along_the_wire():
    b = board()
    instances, view, area = view_of(b, [("c12", "c30")])
    plain = netlist.trace(instances, view, area, 5.5, [b], {})[0]
    inset = netlist.trace(instances, view, area, 5.5, [b], {"tip_inset_mm": 1.0})[0]
    moved = [np.linalg.norm(np.subtract(p.xy_mm, q.xy_mm)) for p, q in zip(plain.ends, inset.ends)]
    assert moved == pytest.approx([1.0, 1.0], abs=0.05)


def test_wires_that_touch_are_flagged():
    b = board()
    (one,), view, area = view_of(b, [("c12", "c30")])
    (other,), _, _ = view_of(b, [("a20", "h22")])
    full = one.full_mask(view.shape) | other.full_mask(view.shape)
    ys, xs = np.nonzero(full)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    crossed = sam_client.Instance("wire", "0", box, full[box[1]:box[3], box[0]:box[2]])
    (wire,) = netlist.trace([crossed], view, area, 5.5, [b], {})
    assert any("two wires that touch" in p for p in wire.problems)


@pytest.mark.parametrize("bgr, name", [((0, 215, 255), "yellow"), ((200, 90, 30), "blue"), ((0, 140, 255), "orange"),
                                       ((60, 170, 40), "green"), ((235, 235, 235), "white"), ((30, 30, 30), "black"),
                                       ((40, 40, 200), "red")])
def test_color_names(bgr, name):
    assert netlist.color_name(bgr) == name


# --- netlist and pins.h ---

def test_pins_h_defines():
    defines = netlist.read_defines(ROOT / "firmware/dut_demo/include/pins.h")
    assert defines["OLED_CS"] == (5, 8) and defines["OLED_SCK"] == (18, 6)


def test_module_pins_follow_their_headers():
    pins = netlist.module_pins(bench_config()["modules"], PARTS, [board()])
    assert pins["esp32.EN"] == "10a-e" and pins["esp32.GPIO13"] == "22a-e" and pins["esp32.GPIO5"] == "17f-j"
    assert pins["oled.D0"] == "42a-e" and pins["oled.CS"] == "46a-e"


def _wire(wid, a, z):
    return netlist.Wire(wid, [netlist.End((0, 0), (1, 0), a, a, 0.1, True), netlist.End((0, 0), (-1, 0), z, z, 0.1, True)],
                        10.0, 2.0, "red", 0.9)


def test_check_flags_unwired_and_doubly_wired_signals():
    bench = bench_config()
    pins = netlist.module_pins(bench["modules"], PARTS, [board()])
    defines = netlist.read_defines(ROOT / bench["firmware"]["file"])
    wires = [_wire("wire:1", "42a-e", "16f-j"),  # SCK to GPIO18: right
             _wire("wire:2", "43a-e", "10f-j"), _wire("wire:3", "10f-j", "11f-j")]  # MOSI to GPIO23 and GPIO22
    rows, misses = netlist.check(wires, pins, bench["firmware"], defines, "pins.h")
    by_signal = {m["signal"]: m for m in misses}
    assert "OLED_SCK" not in by_signal and rows[0] == {"signal": "OLED_SCK", "wire": "wire:1", "display_pin": "D0",
                                                       "esp32_gpio": 18}
    assert by_signal["OLED_MOSI"]["wired_to"] == [22, 23] and by_signal["OLED_MOSI"]["problem"] == "wired to several GPIOs"
    assert by_signal["OLED_CS"] == {"signal": "OLED_CS", "pins_h": 5, "wired_to": None, "wire": None, "file": "pins.h",
                                    "line": 8, "problem": "not wired to any GPIO"}


# --- the whole read ---

def test_reader_finds_the_planted_cs_bug(tmp_path):
    frame, actual, bench = photo(shift_mm=(0.6, -0.5), turn_deg=0.4)
    client, b = sam_for(bench, actual), budget()
    result = reader.read_board(frame, bench, PARTS, client, EYES, b, ROOT, out_dir=tmp_path)
    assert result["ok"] and result["problems"] == []
    assert {tuple(sorted(w["ends"])) for w in result["wires"]} == {tuple(sorted((a, z))) for a, z, _ in WIRES}
    assert {w["color"] for w in result["wires"]} == {"yellow", "blue", "orange", "green", "white"}
    assert [r["esp32_gpio"] for r in result["netlist"]] == [18, 23, 13, 16, 17]
    (miss,) = result["mismatches"]
    assert {k: miss[k] for k in ("signal", "pins_h", "wired_to", "hole", "row", "file", "line")} == {
        "signal": "OLED_CS", "pins_h": 5, "wired_to": 13, "hole": "a22", "row": "row:22",
        "file": "firmware/dut_demo/include/pins.h", "line": 8}
    assert result["endpoint_residual_mm"]["max"] < 0.6
    assert len(client.requests) == 1 and b.spent == pytest.approx(0.0025)
    assert result["_image"][:8] == b"\x89PNG\r\n\x1a\n" and set(result["_timing"]) == {"tags", "sam", "trace"}
    saved = {p.name.split("-")[-1] for p in tmp_path.iterdir()}
    assert len(saved) == 3 and "view.png" in saved and any(name.endswith(".json") for name in saved)


def test_reader_without_placed_modules_still_traces_the_wires():
    frame, actual, bench = photo()
    bench["modules"] = {}
    result = reader.read_board(frame, bench, PARTS, sam_for(bench, actual), EYES, budget(), ROOT)
    assert len(result["wires"]) == 5 and result["netlist"] == [] and result["mismatches"] == []
    assert any("places no modules" in p for p in result["problems"])


def test_a_wire_end_between_holes_is_flagged_and_left_out_of_the_nets():
    b = board()
    # On a 2.54 mm grid nearly every point is within 1.2 mm of some hole; the middle of four holes isn't.
    instances, view, area = view_of(b, [("c12", (20.5, 2.5))])
    (wire,) = netlist.trace(instances, view, area, 5.5, [b], {"snap_mm": 1.2, "tip_inset_mm": 1.0})
    assert [e.snapped for e in wire.ends] == [True, False] and wire.ends[1].residual_mm > 1.2
    assert any("mm from the nearest hole" in p for p in wire.problems)
    rows, _ = netlist.check([wire], {"oled.D0": "12a-e", "esp32.GPIO18": "20a-e"},
                            {"gpio_module": "esp32", "signals": {"OLED_SCK": "oled.D0"}}, {"OLED_SCK": (18, 6)}, "pins.h")
    assert rows == [{"signal": "OLED_SCK", "wire": None, "display_pin": "D0", "esp32_gpio": None}]


def test_read_breadboard_tool_photographs_and_reads_the_board(tmp_path, monkeypatch):
    import shutil

    from wbc.agent.tools import Bench, build_toolbox
    from wbc.config import load_yaml
    from wbc.perception import camera

    shutil.copytree(ROOT / "firmware" / "dut_demo", tmp_path / "firmware" / "dut_demo")
    frame, actual, bench = photo()
    asked = []
    monkeypatch.setattr(camera, "capture", lambda index, width, height: asked.append((index, width, height)) or frame)
    tb = build_toolbox(Bench(tmp_path, load_yaml("models.yaml"), sam_for(bench, actual), budget(), layout=bench,
                             parts=PARTS))
    result = tb.dispatch("read_breadboard", {})
    assert asked == [(0, 1920, 1080)]
    assert result["ok"] and result["mismatches"][0]["wired_to"] == 13 and "capture" in result["_timing"]
    assert any((tmp_path / ".wbc" / "vision").glob("board-*.png"))


def _script(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_printed_carrier_puts_every_tag_where_bench_yaml_says():
    make_carrier = _script("make_carrier")
    bench = bench_config()
    sheet = make_carrier.sheet(bench, PARTS)
    s = make_carrier.DPI / 25.4
    assert sheet.shape[:2] == (round(bench["carrier"]["paper_mm"][1] * s), round(bench["carrier"]["paper_mm"][0] * s))
    found = tags.detect(sheet)
    assert sorted(found) == [10, 11, 12, 13]
    for tag_id, corners in found.items():  # pixel k covers mm [k, k + 1) / s, so its center is at (k + 0.5) / s
        printed = tags.corners_mm(bench["carrier"]["tags"][tag_id], bench["carrier"]["tag_size_mm"])
        assert np.abs((corners + 0.5) / s - printed).max() < 0.06


def test_read_board_script_checks_and_reads_a_saved_photo(tmp_path, monkeypatch, capsys):
    script = _script("read_board")
    frame, actual, bench = photo(shift_mm=(0.6, -0.5), turn_deg=0.4)
    path = tmp_path / "photo.png"
    cv2.imwrite(str(path), frame)
    client, b = sam_for(bench, actual), budget()
    monkeypatch.setattr(script, "OUT", tmp_path / "vision")
    monkeypatch.setattr(script.config, "meta_client", lambda models: client)
    monkeypatch.setattr(script.Budget, "from_config", classmethod(lambda cls, models, root: b))
    real = script.config.load_yaml
    monkeypatch.setattr(script.config, "load_yaml", lambda name: {"bench.yaml": bench, "parts.yaml": PARTS}.get(name)
                        or real(name))

    assert script.main(["check", "--photo", str(path)]) == 0
    out = capsys.readouterr().out
    import re

    shift = re.search(r"holes found ([-+][\d.]+), ([-+][\d.]+) mm and ([-+][\d.]+)", out)
    assert "saw [10, 11, 12, 13]" in out and client.requests == []
    assert [float(v) for v in shift.groups()] == pytest.approx([0.6, -0.5, 0.4], abs=0.1)  # as the photo was made

    assert script.main(["read", "--photo", str(path)]) == 0
    out = capsys.readouterr().out
    assert "MISMATCH OLED_CS: wired to a different GPIO, wired to 13, pins.h line 8 says 5 (at a22)" in out
    assert len(client.requests) == 1 and b.spent == pytest.approx(0.0025)
