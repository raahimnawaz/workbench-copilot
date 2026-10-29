import shutil
from types import SimpleNamespace as NS

from fakes import FakeClient, usage

from wbc.agent import coder
from wbc.agent.budget import Budget
from wbc.agent.tools import Bench, build_toolbox
from wbc.config import ROOT

PINS = "firmware/dut_demo/include/pins.h"
FIX = """Here's the fix.
```diff
--- firmware/dut_demo/include/pins.h
+++ firmware/dut_demo/include/pins.h
@@ -6,2 +6,2 @@
-#define SENSOR_PIN 15    // the sensor's divider, through its jumper
+#define SENSOR_PIN 4     // the sensor's divider, through its jumper
 #define SAMPLE_MS  20    // one reading every 20 ms, printed over serial
```
WHY: The sensor's jumper lands on GPIO 4, so pins.h has to match the wire.
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
    assert why.startswith("The sensor's jumper")


def test_split_diff_without_a_fence():
    text = FIX.replace("```diff\n", "").replace("```\n", "")
    diff, why = coder.split_diff(text)
    assert "+#define SENSOR_PIN 4 " in diff and "WHY" not in diff and why


def test_split_diff_with_nothing_to_find():
    assert coder.split_diff("I can't do that.\nWHY: pins.h wasn't sent.") == ("", "pins.h wasn't sent.")


def test_ask_coder_returns_a_checked_diff(tmp_path):
    root = project(tmp_path)
    client = wonder_says(FIX)
    b = budget()
    out = coder.ask_coder(client, cfg(), b, root, "Fix the sensor pin", [PINS], {"SENSOR_PIN": "GPIO 4"})
    assert out["ok"] and out["compiled"] is False and "skipped" in out["log"]
    assert "#define SENSOR_PIN 15 " in (root / PINS).read_text(), "ask_coder must not touch the working tree"
    sent = client.responses.requests[0]
    assert sent["model"] == "wonder" and "GPIO 4" in sent["input"] and "=== firmware/dut_demo/include/pins.h" in sent["input"]
    assert b.spent > 0


def test_ask_coder_rejects_a_diff_that_does_not_apply(tmp_path):
    root = project(tmp_path)
    out = coder.ask_coder(wonder_says(FIX.replace("SENSOR_PIN 15 ", "SENSOR_PIN 17 ")), cfg(), budget(), root,
                          "Fix the sensor pin", [PINS])
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
    check = "grep -q 'SENSOR_PIN 4 ' firmware/dut_demo/include/pins.h"
    out = coder.ask_coder(wonder_says(FIX), cfg(compile_cmd=check), budget(), root, "Fix the sensor pin", [PINS])
    assert out["ok"] and out["compiled"] is True
    # A second run proves the sandbox was reverted: the same diff still applies there.
    again = coder.ask_coder(wonder_says(FIX), cfg(compile_cmd=check), budget(), root, "Fix the sensor pin", [PINS])
    assert again["ok"] and again["compiled"] is True


def test_compile_failure_is_reported(tmp_path):
    out = coder.ask_coder(wonder_says(FIX), cfg(compile_cmd="false"), budget(), project(tmp_path), "x", [PINS])
    assert out["ok"] is False and out["error"] == "patched firmware does not compile"


def test_ask_coder_times_wonder_and_the_compile(tmp_path):
    out = coder.ask_coder(wonder_says(FIX), cfg(compile_cmd="true"), budget(), project(tmp_path), "Fix the sensor pin", [PINS])
    assert out["ok"] and set(out["_timing"]) == {"wonder", "compile"}


def test_an_empty_diff_warms_the_sandbox(tmp_path):
    """The agent builds the unpatched firmware at start, so the first patch doesn't pay the cold build."""
    root = project(tmp_path)
    ok, compiled, _ = coder.compile_in_sandbox("", root, cfg(compile_cmd="true"))
    assert ok and compiled and (coder.sandbox_dir(root) / PINS).exists()


def test_auto_stage_puts_a_checked_patch_in_the_working_tree(tmp_path):
    """Oct 28: the router spent a whole round (11.2 s) just calling propose_patch."""
    root = project(tmp_path)
    tb = build_toolbox(Bench(root, {"coder": cfg(auto_stage=True)}, wonder_says(FIX), budget()))
    out = tb.dispatch("ask_coder", {"task": "Fix the sensor pin", "files": [PINS]})
    assert out["ok"] and out["staged"]["ok"] and out["staged"]["line"] == 6
    assert "#define SENSOR_PIN 4 " in (root / PINS).read_text()
    again = tb.dispatch("propose_patch", {"patch_id": out["patch_id"]})
    assert again["ok"] and again["already_staged"]


def test_without_auto_stage_the_tree_waits_for_propose_patch(tmp_path):
    root = project(tmp_path)
    tb = build_toolbox(Bench(root, {"coder": cfg()}, wonder_says(FIX), budget()))
    out = tb.dispatch("ask_coder", {"task": "Fix the sensor pin", "files": [PINS]})
    assert out["ok"] and "staged" not in out and "#define SENSOR_PIN 15 " in (root / PINS).read_text()
    assert tb.dispatch("propose_patch", {"patch_id": out["patch_id"]})["ok"]
    assert "#define SENSOR_PIN 4 " in (root / PINS).read_text()


def test_propose_patch_applies_to_the_working_tree(tmp_path):
    root = project(tmp_path)
    diff, why = coder.split_diff(FIX)
    out = coder.propose_patch(root, diff, why)
    assert out["ok"] and out["files"] == [PINS] and out["line"] == 6
    assert "#define SENSOR_PIN 4 " in (root / PINS).read_text()
    assert coder.propose_patch(root, diff, why)["ok"] is False, "a second apply must fail cleanly"
