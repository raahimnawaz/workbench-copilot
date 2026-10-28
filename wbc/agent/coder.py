"""ask_coder: the router's only path to code changes. Muse Wonder writes a diff, and code checks it.

A diff reaches the router only after it applies cleanly and, when a compile command is set,
builds in a sandbox copy of the firmware. The working tree is untouched until the patch is staged.
"""
from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path

CODER_SYS = (Path(__file__).parent / "prompts" / "coder.md").read_text(encoding="utf-8")
_FENCE = re.compile(r"```(?:diff|patch)?[ \t]*\n(.*?)```", re.S)
_WHY = re.compile(r"^WHY:\s*(.+)$", re.M)
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)", re.M)


def safe_path(root: Path, path: str) -> Path:
    """Resolve a repo-relative path and refuse anything outside the repo."""
    root = root.resolve()
    p = (root / path).resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"{path} is outside the project")
    return p


def normalize_paths(diff: str) -> str:
    """Give ---/+++ headers a/ and b/ prefixes, since models often leave them off and git apply expects -p1."""
    out = []
    for line in diff.splitlines():
        for marker, prefix in (("--- ", "a/"), ("+++ ", "b/")):
            if line.startswith(marker):
                path = line[4:].split("\t")[0].strip()
                if path != "/dev/null" and not path.startswith(("a/", "b/")):
                    line = marker + prefix + path
        out.append(line)
    return "\n".join(out) + "\n"


def split_diff(text: str) -> tuple:
    """Pull the unified diff and the WHY line out of the coder's reply."""
    why_match = _WHY.search(text)
    why = why_match.group(1).strip() if why_match else ""
    fenced = [block for block in _FENCE.findall(text) if "--- " in block or "diff --git" in block]
    if fenced:
        diff = fenced[0]
    else:
        lines = text.splitlines()
        start = next((i for i, l in enumerate(lines) if l.startswith(("diff --git", "--- "))), None)
        if start is None:
            return "", why
        body = [l for l in lines[start:] if not l.startswith("WHY:")]
        diff = "\n".join(body)
    return normalize_paths(diff.strip("\n")), why


def first_changed_line(diff: str) -> int:
    """Line number, in the new file, of the first added or removed line."""
    line = None
    for text in diff.splitlines():
        hunk = _HUNK.match(text)
        if hunk:
            line = int(hunk.group(1))
        elif line is not None and text.startswith((" ", "+", "-")) and not text.startswith(("+++", "---")):
            if text[0] != " ":
                return line
            line += 1
    return 1


def git_apply(diff: str, cwd: Path, check: bool = True, reverse: bool = False) -> tuple:
    # --recount: models often get the line counts in @@ headers wrong; git recomputes them.
    cmd = ["git", "apply", "--recount"] + (["--check"] if check else []) + (["-R"] if reverse else []) + ["-"]
    p = subprocess.run(cmd, input=diff, text=True, capture_output=True, cwd=cwd)
    return p.returncode == 0, (p.stdout + p.stderr).strip()


_sandbox_lock = threading.Lock()  # the start-up warm-up and a patch must not build in the sandbox at once


def sandbox_dir(root: Path) -> Path:
    """Where patches get compiled. It's under .wbc/ so its build cache survives reboots, unlike the temp folder."""
    return root / ".wbc" / "sandbox"


def compile_in_sandbox(diff: str, root: Path, cfg: dict) -> tuple:
    """Build the firmware with `diff` applied, in a copy of the project. Returns (ok, compiled, log).

    The copy keeps its own .pio build cache, so after the first build only the changed files recompile:
    about 1 s against 14 s cold for the pins.h fix (Oct 28). An empty diff builds the firmware as it is,
    which the agent does at start to warm that cache.
    """
    cmd = (cfg.get("compile_cmd") or "").strip()
    if not cmd:
        return True, False, "compile skipped: no compile_cmd in config/models.yaml"
    if shutil.which(shlex.split(cmd)[0]) is None:
        return True, False, f"compile skipped: {shlex.split(cmd)[0]} is not on PATH"
    sandbox = sandbox_dir(root)
    rel = cfg["compile_dir"]
    with _sandbox_lock:
        shutil.copytree(root / rel, sandbox / rel, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".pio"))
        if diff:
            ok, log = git_apply(diff, sandbox, check=False)
            if not ok:
                return False, False, log
        try:
            p = subprocess.run(shlex.split(cmd), cwd=sandbox, capture_output=True, text=True,
                               timeout=cfg.get("compile_timeout_s", 180))
            ok, log = p.returncode == 0, p.stdout + p.stderr
        except subprocess.TimeoutExpired:
            ok, log = False, "compile timed out"
        finally:
            if diff:
                git_apply(diff, sandbox, check=False, reverse=True)
    return ok, True, log


def ask_coder(client, cfg: dict, budget, root: Path, task: str, files: list, facts: dict | None = None) -> dict:
    if len(files) > cfg.get("max_files", 4):
        return {"ok": False, "error": f"send at most {cfg.get('max_files', 4)} files"}
    try:
        paths = [safe_path(root, f) for f in files]
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    sources = "\n\n".join(f"=== {p.relative_to(root.resolve())}\n{p.read_text(encoding='utf-8')}" for p in paths)
    prompt = (f"{task}\n\nMeasured on the bench (trust these over comments in the code):\n"
              f"{json.dumps(facts or {}, indent=1)}\n\n{sources}")
    budget.check()
    t0 = time.monotonic()
    try:
        r = client.responses.create(model=cfg["model"], instructions=CODER_SYS, input=prompt,
                                    reasoning={"effort": cfg.get("effort", "low")},
                                    timeout=cfg.get("timeout_s", 30))
    except Exception as e:
        return {"ok": False, "error": f"coder unavailable: {type(e).__name__}: {e}",
                "_timing": {"wonder": time.monotonic() - t0}}
    timing = {"wonder": time.monotonic() - t0}  # "_" keys go to the HUD, not to the router
    budget.add_tokens(cfg["model"], getattr(r, "usage", None))
    text = getattr(r, "output_text", "") or ""
    diff, why = split_diff(text)
    if not diff:
        return {"ok": False, "error": "coder returned no diff", "raw": text[:400], "_timing": timing}
    ok, log = git_apply(diff, root)
    if not ok:
        return {"ok": False, "error": "diff does not apply", "diff": diff, "why": why, "log": log[-800:],
                "_timing": timing}
    t0 = time.monotonic()
    ok, compiled, log = compile_in_sandbox(diff, root, cfg)
    if compiled:
        timing["compile"] = time.monotonic() - t0
    result = {"ok": ok, "diff": diff, "why": why, "compiled": compiled, "log": log[-800:], "_timing": timing}
    if not ok:
        result["error"] = "patched firmware does not compile"
    return result


def propose_patch(root: Path, diff: str, why: str) -> dict:
    """Apply a checked diff to the working tree and open the first changed line in VS Code. Never commits."""
    ok, log = git_apply(diff, root)
    if not ok:
        return {"ok": False, "error": "diff no longer applies", "log": log[-800:]}
    ok, log = git_apply(diff, root, check=False)
    if not ok:
        return {"ok": False, "error": "git apply failed", "log": log[-800:]}
    files = [l[6:] for l in diff.splitlines() if l.startswith("+++ b/")]
    line = first_changed_line(diff)
    opened = False
    if files and shutil.which("code"):
        subprocess.Popen(["code", "-g", f"{root / files[0]}:{line}"])
        opened = True
    return {"ok": True, "files": files, "line": line, "why": why, "opened_in_vscode": opened}
