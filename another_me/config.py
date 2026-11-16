from __future__ import annotations

import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> dict:
    p = Path(path) if path else ROOT / "configs" / "default.toml"
    with p.open("rb") as f:
        cfg = tomllib.load(f)
    # Personal settings live in .env so the tracked config stays generic.
    load_env()
    if os.environ.get("PERSONA_NAME"):
        cfg["persona"]["name"] = os.environ["PERSONA_NAME"]
    return cfg


def load_env(path: str | os.PathLike | None = None) -> None:
    """Load KEY=VALUE lines from .env. Variables already set in the environment win."""
    p = Path(path) if path else ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def load_profile(cfg: dict) -> str:
    """Private notes about the person, or "" when the file does not exist."""
    path = cfg["persona"].get("profile")
    p = resolve(path) if path else None
    return p.read_text().strip() if p and p.exists() else ""


def resolve(path: str) -> Path:
    """Resolve a config path relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def display(path: str | os.PathLike) -> str:
    """Path relative to the repo root when inside it, otherwise absolute."""
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)
