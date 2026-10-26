"""Paths, config files and API clients."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name: str) -> dict:
    with open(ROOT / "config" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env(path: Path = ROOT / ".env") -> None:
    """Read KEY=value lines into the environment without overriding what's already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def api_key() -> str:
    load_env()
    key = os.environ.get("META_API_KEY", "")
    if not key:
        raise RuntimeError("META_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return key


def meta_client(models: dict):
    from openai import OpenAI

    return OpenAI(base_url=models["api"]["base_url"], api_key=api_key())


def backup_client(models: dict):
    """Client for the open-weights router on your own machine (any OpenAI-compatible server)."""
    from openai import OpenAI

    return OpenAI(base_url=models["backup"]["base_url"], api_key=os.environ.get("BACKUP_API_KEY", "local"))
