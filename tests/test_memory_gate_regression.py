"""Regression tests: disk-first launches must not silently skip the memory gate."""

from __future__ import annotations

import pytest

from spark.budget import (
    check_memory,
    estimate_footprint_bytes,
    hydrate_budget_metadata,
)
from spark.config.loader import load_config
from spark.config.schema import DetectSpec, ModelEntry, RuntimeDef, ServerSpec
from spark.errors import InsufficientMemoryError
from spark.inventory import DiscoveredModel, infer_quant_and_params, synthesize_entry
from spark.probe.host import HostProfile, RuntimeStatus
from spark.registry.availability import HUB, Availability
from spark.runner import Supervisor
from spark.runtimes.base import Backend
from helpers import FakeStore


def _profile():
    return HostProfile(
        os="Darwin",
        arch="arm64",
        chip="Apple M2",
        total_memory_bytes=16 * 2**30,
        cpu_count=8,
        probed_at=0.0,
        runtimes={"boom": RuntimeStatus(name="boom", available=True)},
    )


def test_quant_and_param_inference_handles_bit_tags():
    assert infer_quant_and_params("org/model-8B-4bit") == ("q4", 8.0)
    assert infer_quant_and_params("ternary-bonsai-27b-mlx-2bit") == ("q2", 27.0)
    assert infer_quant_and_params("org/model-q6_K") == ("q6", None)


def test_synthesized_entry_carries_budget_metadata():
    disc = DiscoveredModel(
        display_id="org/huge-70B-4bit",
        source="hub",
        repo="org/huge-70B-4bit",
        location="/tmp/snapshot",
        bytes_present=40 * 2**30,
        kind="llm",
        model_format="mlx",
        quant="",  # a scan that could not infer a tag still gets one here
    )
    entry = synthesize_entry(disc)
    assert entry.quant == "q4"
    assert entry.params_billions == 70.0
    assert entry.size_bytes == 40 * 2**30
    assert estimate_footprint_bytes(entry, load_config().memory) is not None


def test_budget_refuses_the_synthesized_70b():
    entry = synthesize_entry(
        DiscoveredModel(
            display_id="org/huge-70B-4bit",
            source="hub",
            repo="org/huge-70B-4bit",
            location="/tmp/snapshot",
            bytes_present=40 * 2**30,
            kind="llm",
        )
    )
    cfg = load_config()
    result = check_memory(entry, _profile(), cfg.memory)
    assert not result.ok
    assert result.estimate_bytes is not None


def test_preflight_hydrates_metadata_from_availability(paths, monkeypatch):
    """The launch path builds inventory with with_bytes=False, so it cannot rely
    on a discovered size — it must use the availability result it already has."""
    cfg = load_config()
    rt = RuntimeDef(
        name="boom",
        detect=DetectSpec(binary="true"),
        server=ServerSpec(
            binary="true",
            args_template=[],
            health_path="/health",
            default_port=19033,
            ready_timeout_s=1.0,
        ),
    )
    cfg.runtimes["boom"] = rt
    entry = ModelEntry(id="huge", hf_repo="org/huge-70B-4bit")
    sup = Supervisor(Backend(rt, cfg), entry, cfg, _profile(), paths, FakeStore())
    monkeypatch.setattr(
        "spark.runner.supervisor.resolve_availability",
        lambda *a, **k: Availability(
            state=HUB,
            detail="in the HF hub cache",
            location="/tmp/snapshot",
            bytes_present=40 * 2**30,
        ),
    )
    with pytest.raises(InsufficientMemoryError):
        sup._preflight()


def test_hydrate_budget_metadata_is_a_reusable_seam():
    """Hydration belongs to the budget domain, not the launch loop."""
    cfg = load_config()
    entry = ModelEntry(id="huge", hf_repo="org/huge-70B-4bit")
    avail = Availability(
        state=HUB,
        detail="in the HF hub cache",
        location="/tmp/snapshot",
        bytes_present=40 * 2**30,
    )
    hydrated = hydrate_budget_metadata(entry, avail, cfg)
    assert hydrated is not entry
    assert hydrated.quant == "q4"
    assert hydrated.params_billions == 70.0
    assert hydrated.size_bytes == 40 * 2**30
    assert entry.quant == ""  # the registry entry is never mutated
    assert entry.size_bytes is None


def test_preflight_still_emits_the_hydration_event(paths, monkeypatch):
    cfg = load_config()
    rt = RuntimeDef(
        name="boom",
        detect=DetectSpec(binary="true"),
        server=ServerSpec(binary="true", default_port=19034),
    )
    cfg.runtimes["boom"] = rt
    sup = Supervisor(
        Backend(rt, cfg),
        ModelEntry(id="huge", hf_repo="org/huge-70B-4bit"),
        cfg,
        _profile(),
        paths,
        FakeStore(),
    )
    monkeypatch.setattr(
        "spark.runner.supervisor.resolve_availability",
        lambda *a, **k: Availability(
            state=HUB,
            detail="in the HF hub cache",
            location="/tmp/snapshot",
            bytes_present=40 * 2**30,
        ),
    )
    events: list[dict] = []

    class _Recorder:
        def info(self, component, event, **payload):
            events.append({"event": event, **payload})

    sup.log = _Recorder()
    with pytest.raises(InsufficientMemoryError):
        sup._preflight()
    hydrated = [e for e in events if e["event"] == "budget_metadata_hydrated"]
    assert len(hydrated) == 1
    assert hydrated[0]["fields"] == ["params_billions", "quant", "size_bytes"]
