#!/usr/bin/env python3
"""The question sheet: one text file holding every question and your answer to it.

    python3 scripts/qa.py init            # build or refresh datasets/raw/qa.txt from the bank
    python3 scripts/qa.py status          # how many are answered, per section
    python3 scripts/qa.py set P012        # save an answer for one question (text on stdin)
    python3 scripts/qa.py sync            # write answered questions to datasets/raw/pairs.jsonl

The sheet is the single source of truth. Edit it by hand or with `set`; `init` keeps
existing answers when the bank changes. Both files are gitignored.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from another_me.config import ROOT, display, load_config, resolve  # noqa: E402
from another_me.schema import write_jsonl  # noqa: E402

BANK = ROOT / "datasets" / "question_bank"
SHEET = ROOT / "datasets" / "raw" / "qa.txt"

# Sheet order. prefix -> (register, kind, section title)
SECTIONS = {
    "P": ("professional", "answer", "PROFESSIONAL: answer as you would to a recruiter or engineer"),
    "C": ("casual", "answer", "CASUAL: answer as you would text a friend"),
    "PF": ("professional", "fallback", "PROFESSIONAL, DECLINE: do not give the detail, say how you would refuse"),
    "CF": ("casual", "fallback", "CASUAL, DECLINE: refuse or deflect the way you would with a friend"),
    "PI": ("professional", "identity", "PROFESSIONAL, IDENTITY: admit you are a model, in your own words"),
    "CI": ("casual", "identity", "CASUAL, IDENTITY: admit you are a model, casually"),
}
_ID = re.compile(r"^\[([A-Z]{1,2})(\d{3})\]\s*$")

HEADER = """\
another-me question sheet
=========================
Write your answer after "A:". It can run over several lines. Leave it empty to skip.
Do not change the [ID] or "Q:" lines. No private details: no employer names, money,
addresses, phone numbers, family or health information.
"""


def bank_questions(prefix: str) -> list[str]:
    register, kind, _ = SECTIONS[prefix]
    path = BANK / f"{register}.{kind}.txt"
    if not path.exists():
        return []
    # Bank files are generic: {name} and {name_lower} become the configured persona name.
    name = load_config()["persona"]["name"]
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line.replace("{name_lower}", name.lower()).replace("{name}", name)
            for line in lines if line and not line.startswith("#")]


def parse_sheet(text: str) -> list[dict]:
    """Return entries in file order: {id, prefix, prompt, reply}."""
    entries, current, field = [], None, None
    for line in text.splitlines():
        m = _ID.match(line)
        if m:
            current = {"id": m.group(1) + m.group(2), "prefix": m.group(1), "prompt": "", "reply": []}
            entries.append(current)
            field = None
        elif current is None or line.startswith("=== "):
            current = None if line.startswith("=== ") else current
        elif line.startswith("Q:") and field is None:
            current["prompt"] = line[2:].strip()
            field = "q"
        elif line.startswith("A:") and field == "q":
            current["reply"].append(line[2:].strip())
            field = "a"
        elif field == "a":
            current["reply"].append(line.rstrip())
    for e in entries:
        e["reply"] = "\n".join(e["reply"]).strip()
    return entries


def render(answers: dict[tuple[str, str], str]) -> str:
    """Build the sheet from the bank. `answers` maps (prefix, prompt) to a reply."""
    out = [HEADER]
    for prefix, (_, _, title) in SECTIONS.items():
        questions = bank_questions(prefix)
        out.append(f"\n=== {title} ({len(questions)}) ===\n")
        for n, prompt in enumerate(questions, 1):
            reply = answers.get((prefix, prompt), "")
            out.append(f"[{prefix}{n:03d}]\nQ: {prompt}\nA: {reply}".rstrip() + "\n")
    return "\n".join(out)


def load_entries() -> list[dict]:
    if not SHEET.exists():
        raise SystemExit(f"{display(SHEET)} not found. Run: python3 scripts/qa.py init")
    return parse_sheet(SHEET.read_text())


def cmd_init(_args) -> None:
    answers = {}
    if SHEET.exists():
        answers = {(e["prefix"], e["prompt"]): e["reply"] for e in parse_sheet(SHEET.read_text()) if e["reply"]}
    text = render(answers)
    kept = {(e["prefix"], e["prompt"]) for e in parse_sheet(text) if e["reply"]}
    lost = set(answers) - kept
    if lost:
        raise SystemExit("Refusing to rewrite: these answered questions are no longer in the bank:\n  "
                         + "\n  ".join(p for _, p in sorted(lost)))
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    SHEET.write_text(text)
    entries = parse_sheet(text)
    print(f"{display(SHEET)}: {len(entries)} questions, {len(kept)} answered")


def cmd_status(_args) -> None:
    entries = load_entries()
    for prefix, (register, kind, _) in SECTIONS.items():
        rows = [e for e in entries if e["prefix"] == prefix]
        done = sum(bool(e["reply"]) for e in rows)
        print(f"  {prefix:2s} {register:13s} {kind:9s} {done:3d} / {len(rows)}")
    print(f"  total answered: {sum(bool(e['reply']) for e in entries)} / {len(entries)}")


def cmd_set(args) -> None:
    reply = sys.stdin.read().strip()
    if not reply:
        raise SystemExit("No answer text on stdin.")
    entries = load_entries()
    target = next((e for e in entries if e["id"] == args.id.upper()), None)
    if target is None:
        raise SystemExit(f"No question with id {args.id}.")
    answers = {(e["prefix"], e["prompt"]): e["reply"] for e in entries if e["reply"]}
    answers[(target["prefix"], target["prompt"])] = reply
    SHEET.write_text(render(answers))
    print(f"{target['id']} saved ({len(answers)} answered)")


def cmd_sync(_args) -> None:
    out = resolve(load_config()["data"]["raw"])
    pairs = []
    for e in load_entries():
        if not e["reply"]:
            continue
        register, kind, _ = SECTIONS[e["prefix"]]
        pairs.append({"id": e["id"], "register": register, "kind": kind,
                      "prompt": e["prompt"], "reply": e["reply"], "source": "interview"})
    write_jsonl(out, pairs)
    print(f"{len(pairs)} pairs -> {display(out)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init").set_defaults(fn=cmd_init)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("set")
    p.add_argument("id")
    p.set_defaults(fn=cmd_set)
    sub.add_parser("sync").set_defaults(fn=cmd_sync)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
