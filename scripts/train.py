#!/usr/bin/env python3
"""Train, evaluate and publish the adapter on Modal.

    uv run scripts/train.py --action smoke                       # cheap check, no GPU
    uv run scripts/train.py --action download                    # base model onto the Modal volume
    uv run scripts/train.py --action train --run-name v1         # train the adapter
    uv run scripts/train.py --action compare --run-name v1       # tuned vs base on held-out questions
    uv run scripts/train.py --action deploy                      # publish the model service for the demo

    --max-steps 3 --limit 24                      a dry run of `train` that costs a few cents
    --split valid --temperatures 0.5,0.7,0.9      with `compare`: reply length per temperature

Requires a Modal token: MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env (see .env.example),
or a token saved by `modal token new`. Without one it refuses to run.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from another_me.config import ROOT, load_env  # noqa: E402

APP = "another_me/train_modal.py"


def modal_token_present() -> bool:
    in_env = bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))
    return in_env or (Path.home() / ".modal.toml").exists()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--action", choices=["smoke", "download", "train", "compare", "deploy"],
                    default="train", help="smoke and download use no GPU")
    ap.add_argument("--run-name", default="v1")
    ap.add_argument("--max-steps", type=int, default=0, help="train: stop early (dry run)")
    ap.add_argument("--limit", type=int, default=0, help="train, compare: use the first N rows")
    ap.add_argument("--split", choices=["valid", "test"], default="test", help="compare: which held-out split")
    ap.add_argument("--temperatures", default="", help="compare: comma-separated sweep, tuned replies only")
    ap.add_argument("--dry-run", action="store_true", help="print the command without running it")
    args = ap.parse_args()

    load_env()
    if not modal_token_present():
        raise SystemExit("No Modal token found. Fill MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env "
                         "(see .env.example) or run `modal token new`.")
    venv_modal = Path(sys.executable).parent / "modal"
    modal = str(venv_modal) if venv_modal.exists() else shutil.which("modal")
    if not modal:
        raise SystemExit("modal CLI not found. Run: uv sync --extra modal")

    if args.action == "deploy":
        cmd = [modal, "deploy", APP]
    else:
        cmd = [modal, "run", APP, "--action", args.action, "--run-name", args.run_name,
               "--max-steps", str(args.max_steps), "--limit", str(args.limit),
               "--split", args.split, "--temperatures", args.temperatures]
    print(f"GPU: {os.environ.get('ANOTHER_ME_GPU', 'L4')}\n$ {' '.join(cmd)}")
    if not args.dry_run:
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
