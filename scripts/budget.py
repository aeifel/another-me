#!/usr/bin/env python3
"""Estimate training time and cost on Modal before running anything.

    python3 scripts/budget.py --pairs 500
    python3 scripts/budget.py --pairs 500 --avg-tokens 170 --epochs 3
    python3 scripts/budget.py --pairs 500 --device L4 --size 4B --tps 900   # your measured tps

Only the A10 / 4B throughput is measured; the rest are estimates. Pass --tps to override.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from another_me.budget import EST_TPS, FREE_CREDIT_USD, MEASURED, estimate  # noqa: E402
from another_me.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pairs", type=int, default=500, help="training pairs")
    ap.add_argument("--avg-tokens", type=int, default=450, help="tokens per example (prepare.py prints yours)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--device", choices=list(EST_TPS), help="with --tps: the device measured")
    ap.add_argument("--size", choices=["1.7B", "4B", "8B"], help="with --tps: the model size")
    ap.add_argument("--tps", type=float, help="measured training tokens/second")
    ap.add_argument("--overhead-sec", type=float, default=180,
                    help="Modal start-up + model load per run (billed)")
    args = ap.parse_args()
    if args.tps and not (args.device and args.size):
        ap.error("--tps needs --device and --size")

    cfg = load_config()
    epochs = args.epochs or cfg["train"]["epochs"]
    m = cfg["modal"]
    total = args.pairs * args.avg_tokens * epochs
    print(f"{args.pairs} pairs x {args.avg_tokens} tokens x {epochs} epochs = {total:,} training tokens\n")
    print(f"{'device':10s} {'model':5s} {'tok/s':>7s} {'train':>9s} {'cost/run':>9s} {'runs per $' + str(int(FREE_CREDIT_USD)):>12s}")
    for device, sizes in EST_TPS.items():
        for size in sizes:
            measured = args.tps if (device, size) == (args.device, args.size) else None
            e = estimate(device, size, args.pairs, args.avg_tokens, epochs, measured,
                         args.overhead_sec, m["cpu_cores"], m["memory_gib"])
            if e.tps is None:
                print(f"{device:10s} {size:5s} {'-':>7s} {'no fit':>9s}")
                continue
            tag = "*" if measured or (device, size) in MEASURED else " "
            cost = f"${e.cost_usd:.2f}"
            runs = "-" if e.runs_per_credit is None else str(e.runs_per_credit)
            print(f"{device:10s} {size:5s} {e.tps:>6.0f}{tag} {e.train_minutes:>7.1f}m {cost:>9s} {runs:>12s}")
    print("\n* = measured throughput; all other tok/s values are estimates.")


if __name__ == "__main__":
    main()
