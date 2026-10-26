import sys
from pathlib import Path

# Make the repo root and this folder importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
