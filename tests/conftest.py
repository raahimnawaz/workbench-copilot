import shutil
import sys
from pathlib import Path

import pytest

# Make the repo root and this folder importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_which = shutil.which


@pytest.fixture(autouse=True)
def no_editor(monkeypatch):
    """coder.propose_patch opens the patched file in VS Code when `code` is on PATH. On a machine with VS Code's
    command installed, that opened three tabs per test run, so tests never find it."""
    monkeypatch.setattr(shutil, "which", lambda cmd, *args, **kwargs: None if cmd == "code" else
                        _which(cmd, *args, **kwargs))
