"""--show: the HUD in a window of its own and the demo board's schematic in KiCad, on this Mac's screen, for a take.

The HUD opens as a Chrome app window (--app: no tabs and no address bar), in a Chrome profile of its own under .wbc,
so it starts its own Chrome whatever else is open, keeps where its window was last put, and closes when the agent
ends. The schematic opens in KiCad's Schematic Editor, and so does the copy add_part draws. An editor that's open is
closed first: for a take it holds nothing unsaved, and its window then comes back on the new file without asking
about a file open twice. The agent needs neither: without Chrome or KiCad it says so and goes on.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
SCHEMATIC_EDITOR = Path("/Applications/KiCad/Schematic Editor.app")  # KiCad's macOS installer puts it there
FIRST_SIZE = (1000, 760)  # points: the HUD window's size the first time; after that Chrome keeps it where it was put
CLOSE_TRIES = 30  # tenths of a second for an editor to close before it's killed outright, or to open
SETTLE_S = 1.0  # just after an editor quits, macOS still hands it an open, which loses the file


def hud_window(url: str, profile: Path, popen=subprocess.Popen, chrome: Path = CHROME):
    """Chrome showing `url` as an app window in its own `profile`: the process, to terminate() at the end, or None
    without Chrome."""
    if not chrome.exists():
        return None
    first = not profile.exists()
    profile.mkdir(parents=True, exist_ok=True)
    args = [str(chrome), f"--app={url}", f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check"]
    if first:
        args.append("--window-size={},{}".format(*FIRST_SIZE))
    return popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def schematic(path: Path, run=subprocess.run, editor: Path = SCHEMATIC_EDITOR, sleep=time.sleep) -> str:
    """Open `path` in KiCad's Schematic Editor, closing one that's running first and clearing the locks a closed
    editor can leave beside the file; what happened, for the HUD's notes."""
    if not editor.exists():
        return f"schematic: KiCad's Schematic Editor isn't in {editor.parent}"
    if not path.exists():
        return f"schematic: there's no {path}"
    replaced = _running(run)
    if replaced:
        run(["pkill", "-x", "eeschema"], capture_output=True)
        for _ in range(CLOSE_TRIES):
            if not _running(run):
                break
            sleep(0.1)
        else:
            run(["pkill", "-9", "-x", "eeschema"], capture_output=True)
        sleep(SETTLE_S)
    for lock in (path.with_name(f"~{path.name}.lck"), path.with_name(f"~{path.stem}.kicad_pro.lck")):
        lock.unlink(missing_ok=True)
    run(["open", "-a", str(editor), str(path)], capture_output=True)
    if replaced:  # and if the open went to the editor that was quitting after all, once more
        for _ in range(CLOSE_TRIES):
            sleep(0.1)
            if _running(run):
                break
        else:
            run(["open", "-a", str(editor), str(path)], capture_output=True)
    return f"schematic: {path.name} opened in KiCad"


def in_editor(path: Path, line: int | None = None, popen=subprocess.Popen, which=shutil.which) -> str:
    """Open `path` in VS Code (its `code` command), at `line` if given, so a take shows it as it changes: pins.h when
    Wonder's fix is staged, and the skill add_part saves. What happened, for the HUD's notes."""
    if not which("code"):
        return "VS Code: no `code` command on PATH, so nothing opened"
    popen(["code", "-g", f"{path}:{line}" if line else str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return f"VS Code: {path.name}"


def _running(run) -> bool:
    return run(["pgrep", "-x", "eeschema"], capture_output=True).returncode == 0
