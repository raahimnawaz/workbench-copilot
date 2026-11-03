"""--show: the HUD in a window of its own and the demo board's schematic in KiCad, on this Mac's screen, for a take.

The HUD opens as a Chrome app window (--app: no tabs and no address bar), in a Chrome profile of its own under .wbc,
so it starts its own Chrome whatever else is open, keeps where its window was last put, and closes when the agent
ends. The schematic opens in KiCad's Schematic Editor, unless an editor has it open already, which would ask about
the file being open twice. The agent needs neither: without Chrome or KiCad it says so and goes on.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
SCHEMATIC_EDITOR = Path("/Applications/KiCad/Schematic Editor.app")  # KiCad's macOS installer puts it there
FIRST_SIZE = (1000, 760)  # points: the HUD window's size the first time; after that Chrome keeps it where it was put


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


def schematic(path: Path, run=subprocess.run, editor: Path = SCHEMATIC_EDITOR) -> str:
    """Open `path` in KiCad's Schematic Editor, unless one's running already; what happened, for the HUD's notes."""
    if not editor.exists():
        return f"schematic: KiCad's Schematic Editor isn't in {editor.parent}"
    if not path.exists():
        return f"schematic: there's no {path}"
    if run(["pgrep", "-x", "eeschema"], capture_output=True).returncode == 0:
        return "schematic: KiCad's Schematic Editor is open already"
    run(["open", "-a", str(editor), str(path)], capture_output=True)
    return f"schematic: {path.name} opened in KiCad"
