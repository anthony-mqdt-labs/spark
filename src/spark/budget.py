"""Resource budgeting: unified-memory footprint and disk headroom.

On a 16 GB Apple Silicon box these are real launch gates, not advisory. Estimates
are deliberately conservative; the operator can override with ``--force``.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .config.schema import DiskPolicy, MemoryPolicy, ModelEntry
from .probe.host import HostProfile

GIB = 2**30


@dataclass
class BudgetCheck:
    ok: bool
    estimate_bytes: int | None
    budget_bytes: int
    detail: str


def usable_memory_bytes(profile: HostProfile, policy: MemoryPolicy) -> int:
    return int(profile.total_memory_bytes * policy.usable_fraction)


def estimate_footprint_bytes(entry: ModelEntry, policy: MemoryPolicy) -> int | None:
    """Rough resident footprint = weights * overhead.

    Prefer a quant-aware parameter estimate; fall back to on-disk size."""
    if entry.params_billions and entry.quant:
        bpp = policy.bytes_per_param.get(entry.quant.lower())
        if bpp:
            weights = entry.params_billions * 1e9 * bpp
            return int(weights * policy.overhead_multiplier)
    if entry.size_bytes:
        return int(entry.size_bytes * policy.overhead_multiplier)
    return None  # unknown — caller decides whether to warn or proceed


def check_memory(
    entry: ModelEntry, profile: HostProfile, policy: MemoryPolicy
) -> BudgetCheck:
    budget = usable_memory_bytes(profile, policy)
    est = estimate_footprint_bytes(entry, policy)
    if est is None:
        return BudgetCheck(
            ok=True, estimate_bytes=None, budget_bytes=budget,
            detail="footprint unknown (no params/quant/size) — proceeding without a memory gate",
        )
    ok = est <= budget
    detail = (
        f"est {est / GIB:.1f} GiB vs budget {budget / GIB:.1f} GiB "
        f"({profile.total_memory_bytes / GIB:.0f} GiB total * {policy.usable_fraction:g})"
    )
    return BudgetCheck(ok=ok, estimate_bytes=est, budget_bytes=budget, detail=detail)


def free_disk_bytes(path: Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0


def check_disk(path: Path, need_bytes: int, policy: DiskPolicy) -> BudgetCheck:
    free = free_disk_bytes(path)
    floor = int(policy.min_free_gib * GIB)
    remaining_after = free - need_bytes
    ok = remaining_after >= floor
    detail = (
        f"free {free / GIB:.1f} GiB, need {need_bytes / GIB:.1f} GiB, "
        f"floor {policy.min_free_gib:g} GiB -> {remaining_after / GIB:.1f} GiB left"
    )
    return BudgetCheck(ok=ok, estimate_bytes=need_bytes, budget_bytes=floor, detail=detail)
