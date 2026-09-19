"""Availability reconciliation: registry entries vs. weights actually on disk.

These tests pin the behaviour the 2026-09-18 incident demanded: an entry that
asserts weights nobody can find must be *reportable* (`spark list`), *nameable*
(`__complete`, in the administrative contexts where naming it is the point), and
*refused* at launch rather than turned into an invisible multi-GB fetch behind a
passing health check.
"""

from __future__ import annotations

import pytest

from spark.config.schema import ModelEntry
from spark.registry.availability import (
    EXTERNAL,
    HUB,
    LOCAL,
    MISSING,
    UNVERIFIABLE,
    availability_map,
    requires_weights_for,
    resolve_availability,
)


def _store_entry(paths, *, name="m-store", hf_repo="", payload=b"x" * 32):
    d = paths.model_store_dir / name
    d.mkdir(parents=True, exist_ok=True)
    if payload:
        (d / "model.safetensors").write_bytes(payload)
    return ModelEntry(id=name, path=str(d), hf_repo=hf_repo, model_format="mlx")


def _hub_entry(tmp_path, monkeypatch, repo="org/Model", files=None):
    """Seed a hub cache under a private HF_HUB_CACHE and point env at it."""
    root = tmp_path / "hfcache"
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    snap = root / ("models--" + repo.replace("/", "--")) / "snapshots" / ("a" * 40)
    if files is not None:
        snap.mkdir(parents=True, exist_ok=True)
        for f in files:
            (snap / f).write_bytes(b"y" * 16)
    return ModelEntry(id="hub-model", hf_repo=repo, model_format="mlx")


def test_store_weights_present_is_local(paths):
    a = resolve_availability(_store_entry(paths), paths)
    assert a.state == LOCAL
    assert a.bytes_present >= 32
    assert a.ok


def test_store_dir_deleted_reports_missing(paths):
    """The reported bug: the entry outlives the weights and is still listed."""
    entry = _store_entry(paths, hf_repo="org/Model")
    (paths.model_store_dir / "m-store" / "model.safetensors").unlink()
    a = resolve_availability(entry, paths)
    assert a.state == MISSING
    assert not a.ok
    assert a.downloadable  # hf_repo known -> `spark download` can restore it


def test_store_path_that_never_existed_is_missing(paths):
    entry = ModelEntry(id="ghost", path=str(paths.model_store_dir / "ghost"))
    assert resolve_availability(entry, paths).state == MISSING


def test_empty_store_dir_is_missing(paths):
    entry = _store_entry(paths, payload=b"")
    assert resolve_availability(entry, paths).state == MISSING


def test_hub_cache_hit_is_hub(paths, tmp_path, monkeypatch):
    entry = _hub_entry(tmp_path, monkeypatch, files=["config.json"])
    a = resolve_availability(entry, paths)
    assert a.state == HUB
    assert a.bytes_present >= 16


def test_hub_cache_miss_is_missing_and_downloadable(paths, tmp_path, monkeypatch):
    entry = _hub_entry(tmp_path, monkeypatch, files=None)  # repo dir absent
    a = resolve_availability(entry, paths)
    assert a.state == MISSING
    assert a.downloadable
    assert "hub cache" in a.detail


def test_runtime_that_owns_its_weights_is_external(paths):
    entry = ModelEntry(id="router", backend="darkcore", hf_repo="org/Whatever")
    a = resolve_availability(entry, paths, requires_weights=False)
    assert a.state == EXTERNAL
    assert a.ok


def test_entry_asserting_nothing_is_unverifiable(paths):
    assert resolve_availability(ModelEntry(id="bare"), paths).state == UNVERIFIABLE


def test_requires_weights_for_uses_runtime_def(config):
    """darkcore/relay/ollama own their weights; inference backends do not."""
    assert not requires_weights_for(ModelEntry(id="r", backend="darkcore"), config)
    assert not requires_weights_for(ModelEntry(id="r", backend="relay"), config)
    assert not requires_weights_for(ModelEntry(id="r", backend="ollama"), config)
    assert requires_weights_for(ModelEntry(id="m", backend="mlx_lm"), config)
    # No backend yet (freshly downloaded, research pending) -> verify anyway.
    assert requires_weights_for(ModelEntry(id="m"), config)


def test_availability_map_covers_registry(paths, tmp_path, monkeypatch):
    local = _store_entry(paths)
    gone = ModelEntry(id="gone", hf_repo="org/Gone", model_format="mlx")
    avail = availability_map([local, gone], paths)
    assert avail["m-store"].state == LOCAL
    assert avail["gone"].state == MISSING


def test_label_is_compact_and_flags_missing(paths):
    assert resolve_availability(_store_entry(paths), paths).label.startswith("local")
    gone = ModelEntry(id="g", hf_repo="org/G")
    assert resolve_availability(gone, paths).label == "MISSING"


def test_without_bytes_still_reports_state(paths):
    """The completion feed must stay fast but must not lose the MISSING signal."""
    a = resolve_availability(_store_entry(paths), paths, with_bytes=False)
    assert a.state == LOCAL
    gone = ModelEntry(id="g", hf_repo="org/G")
    assert resolve_availability(gone, paths, with_bytes=False).state == MISSING


@pytest.mark.parametrize("state", [LOCAL, HUB, EXTERNAL, UNVERIFIABLE])
def test_ok_states(paths, tmp_path, monkeypatch, state):
    if state == LOCAL:
        a = resolve_availability(_store_entry(paths), paths)
    elif state == HUB:
        a = resolve_availability(
            _hub_entry(tmp_path, monkeypatch, files=["config.json"]), paths
        )
    elif state == EXTERNAL:
        a = resolve_availability(
            ModelEntry(id="r", backend="darkcore"), paths, requires_weights=False
        )
    else:
        a = resolve_availability(ModelEntry(id="bare"), paths)
    assert a.state == state and a.ok
