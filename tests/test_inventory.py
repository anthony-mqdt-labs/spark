"""Disk-first inventory: the CLI offers what is on disk, not what is registered.

Pins the production-UX contract behind the user's complaint: an unregistered but
complete model in the HF hub cache (or the spark store) is runnable, offered in
completion, and resolvable by `spark run` — while a registered entry whose
weights are gone is reported as a defect, never offered as runnable. Cached
repos that are not chat models (embedders, speech) are never offered anywhere.
"""

from __future__ import annotations

import json

import pytest

from spark.config.schema import ModelEntry
from spark.inventory import (
    build_inventory,
    classify_snapshot,
    resolve_for_run,
    scan_hub_cache,
    scan_store,
    synthesize_entry,
)
from spark.registry import save_model


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A fake HF hub cache with one LLM, one embedder, one TTS, one partial."""
    root = tmp_path / "hfcache"
    monkeypatch.setenv("HF_HUB_CACHE", str(root))

    def seed(repo: str, files: dict, config: dict | None):
        snap = root / ("models--" + repo.replace("/", "--")) / "snapshots" / ("a" * 40)
        snap.mkdir(parents=True, exist_ok=True)
        if config is not None:
            files = {**files, "config.json": json.dumps(config)}
        for name, payload in files.items():
            (snap / name).write_bytes(payload if isinstance(payload, bytes) else payload.encode())
        # Pin refs/main like a real `hf download` would.
        ref = snap.parent.parent / "refs"
        ref.mkdir(parents=True, exist_ok=True)
        (ref / "main").write_text("a" * 40)
        return snap

    seed(
        "mlx-community/Small-LLM-8bit",
        {"model.safetensors": b"w" * 64, "tokenizer.json": b"{}",
         "chat_template.jinja": b"{% for m in messages %}{% endfor %}"},
        {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]},
    )
    seed(
        "BAAI/bge-small",
        {"model.safetensors": b"w" * 64, "tokenizer.json": b"{}",
         "modules.json": b"{}", "sentence_bert_config.json": b"{}"},
        {"model_type": "bert", "architectures": ["BertModel"]},
    )
    seed(
        "mlx-community/Some-TTS-8bit",
        {"model.safetensors": b"w" * 64, "tokenizer_config.json": b"{}"},
        {"model_type": "qwen3_tts", "architectures": ["Qwen3TTSForConditionalGeneration"]},
    )
    # Partial: no config.json -> unresolvable, must be skipped silently.
    seed("org/HalfFetched", {"model.safetensors": b"w" * 64}, None)
    return root


def _llm_snap(hub):
    return hub / "models--mlx-community--Small-LLM-8bit" / "snapshots" / ("a" * 40)


# --- classification -----------------------------------------------------------
def test_llm_snapshot_is_servable(hub):
    assert classify_snapshot(_llm_snap(hub)) == ("llm", "")


def test_gguf_snapshot_is_llm(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model-q4.gguf").write_bytes(b"g" * 16)
    assert classify_snapshot(snap)[0] == "llm"


def test_embedding_snapshot_is_not_servable(hub):
    kind, reason = classify_snapshot(
        hub / "models--BAAI--bge-small" / "snapshots" / ("a" * 40)
    )
    assert kind == "embedding" and reason


def test_tts_snapshot_is_not_servable(hub):
    kind, _ = classify_snapshot(
        hub / "models--mlx-community--Some-TTS-8bit" / "snapshots" / ("a" * 40)
    )
    assert kind == "tts"


def test_prism_pack_is_servable_with_prism_format(tmp_path):
    """Hadamard packs route to prism_ml, not stock mlx_lm (which rejects them)."""
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text(
        json.dumps({"model_type": "prism_hadamard_qwen35"})
    )
    (snap / "model.safetensors").write_bytes(b"w" * 16)
    (snap / "tokenizer.json").write_bytes(b"{}")
    kind, reason = classify_snapshot(snap)
    assert (kind, reason) == ("llm", "")
    from spark.inventory import _guess_format

    assert _guess_format("prism-ml/Pack-mlx-2bit", snap) == "prism"


def test_unreadable_snapshot_is_unknown_not_an_error(tmp_path):
    assert classify_snapshot(tmp_path / "nope")[0] == "unknown"


# --- hub scan -----------------------------------------------------------------
def test_hub_scan_finds_llm_and_skips_partial(hub):
    found = {m.display_id: m for m in scan_hub_cache()}
    assert "mlx-community/Small-LLM-8bit" in found
    assert "org/HalfFetched" not in found
    llm = found["mlx-community/Small-LLM-8bit"]
    assert llm.servable and llm.source == "hub" and llm.bytes_present >= 64
    assert llm.model_format == "mlx"  # "mlx" in the repo name


def test_hub_scan_reports_non_servable_with_reasons(hub):
    found = {m.display_id: m for m in scan_hub_cache()}
    assert found["BAAI/bge-small"].kind == "embedding"
    assert found["mlx-community/Some-TTS-8bit"].kind == "tts"
    assert not found["BAAI/bge-small"].servable


def test_hub_scan_without_bytes_stays_fast_but_exact(hub):
    found = {m.display_id: m for m in scan_hub_cache(with_bytes=False)}
    assert found["mlx-community/Small-LLM-8bit"].servable


# --- store scan ---------------------------------------------------------------
def test_store_scan_marks_partial_downloads(paths):
    d = paths.model_store_dir / "half"
    dl = d / ".cache" / "huggingface" / "download"
    dl.mkdir(parents=True)
    (dl / "blob.incomplete").write_bytes(b"z" * 8)
    (d / "config.json").write_bytes(b"{}")
    (found,) = scan_store(paths)
    assert found.kind == "partial" and not found.servable


def test_store_scan_lists_complete_dirs_as_servable(paths):
    d = paths.model_store_dir / "whole"
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"w" * 32)
    (found,) = scan_store(paths)
    assert found.servable and found.source == "store"


# --- inventory join -----------------------------------------------------------
def test_inventory_offers_unregistered_disk_models(paths, hub):
    inv = build_inventory(paths, config=None)
    assert "mlx-community/Small-LLM-8bit" in inv.runnable_ids
    assert {m.display_id for m in inv.runnable_discovered} == {
        "mlx-community/Small-LLM-8bit"
    }
    assert {m.display_id for m in inv.non_servable} == {
        "BAAI/bge-small", "mlx-community/Some-TTS-8bit",
    }


def test_registered_entry_matching_hub_repo_is_not_double_listed(paths, hub):
    save_model(
        ModelEntry(id="small", hf_repo="mlx-community/Small-LLM-8bit",
                   model_format="mlx"),
        paths,
    )
    inv = build_inventory(paths, config=None)
    assert inv.runnable_discovered == []
    assert [e.id for e, _ in inv.runnable_registered] == ["small"]


def test_missing_entries_are_defects_not_runnable(paths, hub):
    save_model(
        ModelEntry(id="gone", hf_repo="org/Nope", model_format="mlx"), paths
    )
    inv = build_inventory(paths, config=None)
    assert "gone" not in inv.runnable_ids
    assert [e.id for e, _ in inv.missing] == ["gone"]


# --- run resolution -----------------------------------------------------------
def test_run_resolves_registry_exact(paths, hub):
    d = paths.model_store_dir / "live"
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"w" * 16)
    save_model(ModelEntry(id="live", path=str(d), model_format="mlx"), paths)
    hit = resolve_for_run("live", paths)
    assert hit is not None and not hit.ephemeral and hit.via == "registry"


def test_run_resolves_unregistered_hub_repo(paths, hub):
    hit = resolve_for_run("mlx-community/Small-LLM-8bit", paths)
    assert hit is not None and hit.ephemeral and hit.via == "hub"
    assert hit.entry.hf_repo == "mlx-community/Small-LLM-8bit"
    assert hit.entry.path == ""  # runtime reads the hub cache via the repo id


def test_run_resolves_hub_tail_when_unique(paths, hub):
    hit = resolve_for_run("small-llm-8bit", paths)
    assert hit is not None and hit.ephemeral


def test_run_never_resolves_embeddings(paths, hub):
    assert resolve_for_run("bge-small", paths) is None
    assert resolve_for_run("BAAI/bge-small", paths) is None


def test_run_miss_returns_none_for_suggestions(paths, hub):
    assert resolve_for_run("definitely-not-here", paths) is None


def test_synthesized_entry_prefers_repo_ref(paths, hub):
    disc = next(
        m for m in scan_hub_cache() if m.display_id == "mlx-community/Small-LLM-8bit"
    )
    entry = synthesize_entry(disc)
    assert entry.hf_repo == "mlx-community/Small-LLM-8bit"
    assert entry.path == ""  # runtime reads the hub cache via the repo id
    assert entry.model_format == "mlx"
