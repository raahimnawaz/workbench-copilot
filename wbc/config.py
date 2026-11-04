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


def open_client(models: dict):
    """Client for Muse Spark Open on Baseten (models.yaml's open), or None without its key in the environment."""
    from openai import OpenAI

    load_env()
    o = models.get("open") or {}
    key = os.environ.get(o.get("key_env", "BASETEN_API_KEY"), "")
    # A stalled Baseten mustn't hold a turn: the library's own defaults wait 10 min and retry twice. When it times
    # out, the router answers instead.
    return OpenAI(base_url=o["base_url"], api_key=key, timeout=o.get("timeout_s", 20), max_retries=0) if o and key \
        else None

