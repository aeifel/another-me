"""The canonical training-pair format and its conversion to chat examples."""
from __future__ import annotations

import json
from pathlib import Path

REGISTERS = ("professional", "casual")
KINDS = ("answer", "fallback", "identity")
REQUIRED = ("id", "register", "kind", "prompt", "reply")

_REGISTER_HINT = {
    "professional": "You are talking to someone asking about your work, experience or opinions.",
    "casual": "You are chatting with a friend.",
}


def validate_pair(obj: dict) -> list[str]:
    """Return a list of problems; empty means the pair is valid."""
    errors = [f"missing '{k}'" for k in REQUIRED if not str(obj.get(k, "")).strip()]
    if obj.get("register") not in REGISTERS:
        errors.append(f"register must be one of {REGISTERS}")
    if obj.get("kind") not in KINDS:
        errors.append(f"kind must be one of {KINDS}")
    return errors


def system_prompt(name: str, register: str, with_knowledge: bool = False,
                  profile: str = "") -> str:
    text = (
        f"You are {name}'s text twin. Reply in first person exactly as {name} would. "
        f"{_REGISTER_HINT[register]}"
    )
    if profile:
        text += (
            f"\n\nWhat you know about {name}:\n{profile}\n\n"
            "Use these notes for facts. Do not invent personal facts beyond them."
        )
    if with_knowledge:
        text += (
            f" Use the tools to look up facts about {name} before answering. "
            "If the tools return nothing relevant, say you don't know and suggest asking "
            f"{name} directly. Never invent facts."
        )
    return text


def to_chat(pair: dict, name: str, profile: str = "") -> dict:
    """Convert a canonical pair into the chat format used for training."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt(name, pair["register"], profile=profile)},
            {"role": "user", "content": pair["prompt"].strip()},
            {"role": "assistant", "content": pair["reply"].strip()},
        ]
    }


def load_paraphrases(path: str | Path, name: str) -> dict[str, list[str]]:
    """Read `ID | reworded question` lines into {sheet id: [questions]}."""
    out: dict[str, list[str]] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        key, text = (part.strip() for part in line.split("|", 1))
        text = text.replace("{name_lower}", name.lower()).replace("{name}", name)
        if key and text:
            out.setdefault(key, []).append(text)
    return out


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open() as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON ({e})") from e
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
