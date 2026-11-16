#!/usr/bin/env python3
"""Validate the raw pairs and write train/valid/test chat files for training.

    python3 scripts/prepare.py                                   # datasets/raw/pairs.jsonl
    python3 scripts/prepare.py --input datasets/sample/pairs.sample.jsonl
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from another_me.config import display, load_config, load_profile, resolve  # noqa: E402
from another_me.schema import load_paraphrases, read_jsonl, to_chat, validate_pair, write_jsonl  # noqa: E402


def approx_tokens(chat: dict) -> int:
    """Rough estimate: 4 characters per token plus chat-template overhead."""
    return sum(len(m["content"]) for m in chat["messages"]) // 4 + 15


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input")
    ap.add_argument("--out")
    ap.add_argument("--no-paraphrases", action="store_true", help="train on the original questions only")
    args = ap.parse_args()

    cfg = load_config()
    name = cfg["persona"]["name"]
    profile = load_profile(cfg)
    src = resolve(args.input or cfg["data"]["raw"])
    out = resolve(args.out or cfg["data"]["processed"])
    if not src.exists():
        raise SystemExit(f"{src} not found. Run `scripts/qa.py sync` first, or pass --input.")

    exclude_file = resolve(cfg["data"]["exclude"])
    excluded = set()
    if exclude_file.exists():
        excluded = {line.split("#")[0].strip() for line in exclude_file.read_text().splitlines()} - {""}

    pairs, seen, problems, skipped = [], set(), 0, 0
    for n, pair in enumerate(read_jsonl(src), 1):
        if pair.get("id") in excluded:
            skipped += 1
            continue
        errors = validate_pair(pair)
        key = (pair.get("register"), str(pair.get("prompt", "")).strip().lower())
        if not errors and key in seen:
            errors = ["duplicate prompt"]
        if errors:
            problems += 1
            print(f"  line {n} dropped: {'; '.join(errors)}")
            continue
        seen.add(key)
        pairs.append(pair)
    if not pairs:
        raise SystemExit("No valid pairs.")

    random.Random(cfg["train"]["seed"]).shuffle(pairs)
    n = len(pairs)
    n_valid = max(1, round(n * cfg["data"]["valid_fraction"])) if n >= 5 else 0
    n_test = max(1, round(n * cfg["data"]["test_fraction"])) if n >= 5 else 0
    splits = {
        "valid": pairs[:n_valid],
        "test": pairs[n_valid : n_valid + n_test],
        "train": pairs[n_valid + n_test :],
    }

    # Reworded questions multiply the training pairs only. Validation and test keep the
    # original wording, and a held-out question never appears in training in any wording.
    extra = []
    if not args.no_paraphrases:
        reworded = load_paraphrases(resolve(cfg["data"]["paraphrases"]), name)
        for pair in splits["train"]:
            for text in reworded.get(pair["id"], []):
                extra.append({**pair, "prompt": text})
    train_rows = splits["train"] + extra
    random.Random(cfg["train"]["seed"] + 1).shuffle(train_rows)

    written = {"valid": splits["valid"], "test": splits["test"], "train": train_rows}
    for split, rows in written.items():
        write_jsonl(out / f"{split}.jsonl", [to_chat(p, name, profile) for p in rows])

    chats = [to_chat(p, name, profile) for p in train_rows]
    avg = sum(approx_tokens(c) for c in chats) // len(chats)
    by = Counter((p["register"], p["kind"]) for p in pairs)
    print(f"{n} valid pairs ({problems} dropped, {skipped} excluded) -> {display(out)}/")
    print(f"  train: {len(splits['train'])} originals + {len(extra)} reworded = {len(train_rows)}, "
          f"valid: {len(splits['valid'])}, test: {len(splits['test'])}")
    for (register, kind), c in sorted(by.items()):
        print(f"  {register:13s} {kind:9s} {c}")
    print(f"  profile notes: {'yes, ' + str(len(profile.split())) + ' words' if profile else 'none'}")
    print(f"  ~{avg} tokens per example, ~{avg * len(train_rows)} tokens per epoch")
    print(f"  budget: python3 scripts/budget.py --pairs {len(train_rows)} --avg-tokens {avg}")
    if n < 200:
        print("  note: under ~200 pairs the adapter mostly memorises. Aim for 300-500+.")


if __name__ == "__main__":
    main()
