"""Training time and cost estimates.

Prices are Modal's published per-second rates (modal.com/pricing, November 2026).
The A10 throughput for the 4B model is measured (examples of about 450 tokens). The other
combinations are scaled from it by rough relative speed and are estimates. Pass a measured
value with --tps to override.
"""
from __future__ import annotations

from dataclasses import dataclass

GPU_PER_SEC = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
}
CPU_CORE_PER_SEC = 0.0000131
MEM_GIB_PER_SEC = 0.00000222
FREE_CREDIT_USD = 30.0

# Training tokens/second for LoRA. None = the model does not fit on that device.
EST_TPS = {
    "T4": {"1.7B": 1100, "4B": 520, "8B": None},
    "L4": {"1.7B": 1900, "4B": 890, "8B": 440},
    "A10": {"1.7B": 3200, "4B": 1480, "8B": 740},
    "L40S": {"1.7B": 8000, "4B": 3700, "8B": 1850},
}
MEASURED = {("A10", "4B")}


@dataclass
class Estimate:
    device: str
    size: str
    total_tokens: int
    tps: float | None
    train_minutes: float | None
    cost_usd: float | None
    runs_per_credit: int | None


def modal_rate_per_sec(gpu: str, cpu_cores: float = 4, memory_gib: float = 16) -> float:
    return GPU_PER_SEC[gpu] + cpu_cores * CPU_CORE_PER_SEC + memory_gib * MEM_GIB_PER_SEC


def estimate(device: str, size: str, pairs: int, avg_tokens: int, epochs: int,
             tps: float | None = None, overhead_sec: float = 180,
             cpu_cores: float = 4, memory_gib: float = 16) -> Estimate:
    total = pairs * avg_tokens * epochs
    tps = tps if tps is not None else EST_TPS[device][size]
    if tps is None:
        return Estimate(device, size, total, None, None, None, None)
    train_sec = total / tps
    cost = (train_sec + overhead_sec) * modal_rate_per_sec(device, cpu_cores, memory_gib)
    return Estimate(device, size, total, tps, train_sec / 60, cost, int(FREE_CREDIT_USD // cost))
