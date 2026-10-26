import shutil
from types import SimpleNamespace as NS

from fakes import FakeClient, usage

from wbc.agent import coder
from wbc.agent.budget import Budget
from wbc.config import ROOT

PINS = "firmware/dut_demo/include/pins.h"
FIX = """Here's the fix.
```diff
--- firmware/dut_demo/include/pins.h
+++ firmware/dut_demo/include/pins.h
@@ -7,3 +7,3 @@
 #define OLED_MOSI 23   // D1 on the display
-#define OLED_CS   5    // chip select
+#define OLED_CS   13   // chip select
 #define OLED_DC   16   // data/command
```
WHY: The CS jumper lands on GPIO 13, so pins.h has to match the wire.
"""


def project(tmp_path):
    """A throwaway copy of the demo firmware to patch."""
    dst = tmp_path / "firmware" / "dut_demo"
    shutil.copytree(ROOT / "firmware" / "dut_demo", dst)
    return tmp_path


def cfg(**overrides):
    base = {"model": "wonder", "effort": "low", "timeout_s": 30, "max_files": 4,
            "compile_cmd": "", "compile_dir": "firmware/dut_demo"}
    base.update(overrides)
    return base


def budget():
    return Budget(10, {"wonder": {"in": 4.0, "out": 20.0}})


def wonder_says(text):
    return FakeClient([NS(output_text=text, usage=usage())])


def test_split_diff_adds_prefixes_and_reads_why():
    diff, why = coder.split_diff(FIX)
    assert diff.startswith("--- a/firmware/") and "\n+++ b/firmware/" in diff
    assert why.startswith("The CS jumper")


def test_split_diff_without_a_fence():
    text = FIX.replace("```diff\n", "").replace("```\n", "")
    diff, why = coder.split_diff(text)
    assert "+#define OLED_CS   13" in diff and "WHY" not in diff and why


def test_split_diff_with_nothing_to_find():
    assert coder.split_diff("I can't do that.\nWHY: pins.h wasn't sent.") == ("", "pins.h wasn't sent.")


def test_ask_coder_returns_a_checked_diff(tmp_path):
    root = project(tmp_path)
    client = wonder_says(FIX)
    b = budget()
    out = coder.ask_coder(client, cfg(), b, root, "Fix CS", [PINS], {"OLED_CS": "GPIO 13"})
    assert out["ok"] and out["compiled"] is False and "skipped" in out["log"]
    assert "#define OLED_CS   5 " in (root / PINS).read_text(), "ask_coder must not touch the working tree"
    sent = client.responses.requests[0]
    assert sent["model"] == "wonder" and "GPIO 13" in sent["input"] and "=== firmware/dut_demo/include/pins.h" in sent["input"]
    assert b.spent > 0


def test_ask_coder_rejects_a_diff_that_does_not_apply(tmp_path):
    root = project(tmp_path)
    out = coder.ask_coder(wonder_says(FIX.replace("OLED_CS   5 ", "OLED_CS   7 ")), cfg(), budget(), root,
                          "Fix CS", [PINS])
    assert out["ok"] is False and out["error"] == "diff does not apply"


def test_ask_coder_refuses_files_outside_the_project(tmp_path):
    out = coder.ask_coder(wonder_says(FIX), cfg(), budget(), project(tmp_path), "x", ["../secrets.txt"])
    assert out["ok"] is False and "outside the project" in out["error"]


def test_ask_coder_reports_an_unavailable_coder(tmp_path):
    out = coder.ask_coder(FakeClient([TimeoutError("read timed out")]), cfg(), budget(), project(tmp_path),
                          "x", [PINS])
    assert out["ok"] is False and "coder unavailable" in out["error"]


def test_compile_runs_on_the_patched_sandbox_and_leaves_it_clean(tmp_path):
    root = project(tmp_path)
    check = "grep -q 'OLED_CS   13' firmware/dut_demo/include/pins.h"
    out = coder.ask_coder(wonder_says(FIX), cfg(compile_cmd=check), budget(), root, "Fix CS", [PINS])
    assert out["ok"] and out["compiled"] is True
    # A second run proves the sandbox was reverted: the same diff still applies there.
    again = coder.ask_coder(wonder_says(FIX), cfg(compile_cmd=check), budget(), root, "Fix CS", [PINS])
    assert again["ok"] and again["compiled"] is True


def test_compile_failure_is_reported(tmp_path):
    out = coder.ask_coder(wonder_says(FIX), cfg(compile_cmd="false"), budget(), project(tmp_path), "x", [PINS])
    assert out["ok"] is False and out["error"] == "patched firmware does not compile"


def test_propose_patch_applies_to_the_working_tree(tmp_path):
    root = project(tmp_path)
    diff, why = coder.split_diff(FIX)
    out = coder.propose_patch(root, diff, why)
    assert out["ok"] and out["files"] == [PINS] and out["line"] == 8
    assert "#define OLED_CS   13" in (root / PINS).read_text()
    assert coder.propose_patch(root, diff, why)["ok"] is False, "a second apply must fail cleanly"
