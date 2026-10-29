from wbc.config import ROOT
from wbc.perception import golden

import yaml


def _read(*wires):
    return [{"ends": list(e), "residual_mm": [0.3, 0.5]} for e in wires]


def test_ends_count_in_either_order():
    s = golden.score("p", [["j3", "j33"], ["d28", "a-"]], _read(("a-", "d28"), ("j3", "j33")))
    assert (s.right, s.ends, s.extra, s.misses) == (4, 4, 0, [])
    assert s.residuals_mm == [0.3, 0.5, 0.3, 0.5]


def test_a_neighbor_hole_is_wrong():
    s = golden.score("p", [["j61", "a+"]], _read(("j62", "a+")))
    assert (s.right, s.ends) == (1, 2)
    assert s.misses == [(["j61", "a+"], ["j62", "a+"])]
    assert s.residuals_mm == [0.5]  # only the right end's


def test_missed_and_extra_wires():
    s = golden.score("p", [["j3", "j33"], ["d28", "a-"]], _read(("j3", "j33"), ("b5", "b9")))
    assert (s.right, s.ends, s.extra) == (2, 4, 1)
    assert s.misses == [(["d28", "a-"], [None, None])]


def test_each_read_wire_matches_one_label():
    # Two labels share j10; the read wire with both ends right must go to its own label.
    s = golden.score("p", [["j10", "j20"], ["j10", "j30"]], _read(("j10", "j30"), ("j10", "j20")))
    assert s.right == 4


def test_total():
    a, b = golden.PhotoScore("a", 6, 6), golden.PhotoScore("b", 4, 3)
    assert golden.total([a, b]) == (9, 10, 0.9)


def test_labels_name_real_holes():
    labels = yaml.safe_load((ROOT / "config" / "golden.yaml").read_text())
    assert labels["gate"] == {"photos": 20, "ends_right": 0.98}
    for entry in labels["photos"]:
        assert all(len(w) == 2 for w in entry["wires"])
