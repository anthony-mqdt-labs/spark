from __future__ import annotations

from spark.budget import (
    GIB,
    check_disk,
    check_memory,
    estimate_footprint_bytes,
    usable_memory_bytes,
)
from spark.config.schema import DiskPolicy, MemoryPolicy, ModelEntry
from spark.probe.host import HostProfile


def _profile(mem_gib=16):
    return HostProfile(
        os="Darwin", arch="arm64", chip="Apple M2",
        total_memory_bytes=mem_gib * GIB, cpu_count=8, probed_at=0.0,
    )


def test_usable_memory_fraction():
    pol = MemoryPolicy(usable_fraction=0.65)
    assert usable_memory_bytes(_profile(16), pol) == int(16 * GIB * 0.65)


def test_estimate_quant_aware():
    pol = MemoryPolicy()
    e = ModelEntry(id="m", params_billions=9.0, quant="q4")
    est = estimate_footprint_bytes(e, pol)
    # 9e9 * 0.5 * 1.2 = 5.4e9
    assert abs(est - 9e9 * 0.5 * 1.2) < 1e6


def test_estimate_unknown_returns_none():
    assert estimate_footprint_bytes(ModelEntry(id="m"), MemoryPolicy()) is None


def test_check_memory_small_model_ok():
    e = ModelEntry(id="m", params_billions=3.0, quant="q4")  # ~1.8 GiB
    chk = check_memory(e, _profile(16), MemoryPolicy())
    assert chk.ok


def test_check_memory_huge_model_fails():
    e = ModelEntry(id="m", params_billions=70.0, quant="q4")  # ~42 GB
    chk = check_memory(e, _profile(16), MemoryPolicy())
    assert not chk.ok


def test_check_memory_unknown_passes_without_gate():
    chk = check_memory(ModelEntry(id="m"), _profile(16), MemoryPolicy())
    assert chk.ok and chk.estimate_bytes is None


def test_check_disk_floor(tmp_path):
    pol = DiskPolicy(min_free_gib=10.0)
    # Asking for an absurd amount must fail.
    chk = check_disk(tmp_path, need_bytes=10_000 * GIB, policy=pol)
    assert not chk.ok
