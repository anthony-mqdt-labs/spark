"""`spark adopt` misses must report what the query actually is.

The complaint: `spark adopt ternary-…` on an already-registered model said
"Nothing adoptable matches", even though `spark list` showed the row with
weights present. The `pending` column is research state, not download state —
the error must say already-registered (with backend/research/weights) instead.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from spark.cli.app import _adopt_miss_error, adopt_command
from spark.config.schema import ModelEntry
from spark.errors import (
    AlreadyRegisteredError,
    ModelNotFoundError,
)
from spark.inventory import build_inventory
from spark.registry import save_model


class _QuietTelemetry:
    def info(self, *a, **k):
        pass

    def warn(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


def _ctx(paths, config=None):
    return SimpleNamespace(paths=paths, config=config, telemetry=_QuietTelemetry())


def _seed_hub_llm(monkeypatch, tmp_path, repo, model_type="qwen3"):
    root = tmp_path / "hfcache"
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    snap = root / ("models--" + repo.replace("/", "--")) / "snapshots" / ("a" * 40)
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_text(
        json.dumps({"model_type": model_type, "architectures": ["Qwen3ForCausalLM"]})
    )
    (snap / "model.safetensors").write_bytes(b"w" * 64)
    (snap / "tokenizer.json").write_bytes(b"{}")
    ref = snap.parent.parent / "refs"
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "main").write_text("a" * 40)
    return snap


def _seed_hub_embedder(monkeypatch, tmp_path, repo="BAAI/bge-small"):
    root = tmp_path / "hfcache"
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    snap = root / ("models--" + repo.replace("/", "--")) / "snapshots" / ("b" * 40)
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_text(
        json.dumps({"model_type": "bert", "architectures": ["BertModel"]})
    )
    (snap / "model.safetensors").write_bytes(b"w" * 64)
    (snap / "tokenizer.json").write_bytes(b"{}")
    (snap / "modules.json").write_bytes(b"{}")
    ref = snap.parent.parent / "refs"
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "main").write_text("b" * 40)
    return snap


@pytest.fixture(autouse=True)
def _isolated_hub(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hfcache-empty"))


# --- already registered -----------------------------------------------------
def test_adopt_miss_on_registered_short_id_reports_state(paths, tmp_path, monkeypatch):
    _seed_hub_llm(monkeypatch, tmp_path, "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
                   model_type="prism_hadamard_qwen35")
    save_model(
        ModelEntry(
            id="ternary-bonsai-2-27b-mlx-2bit",
            hf_repo="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
            backend="prism_ml",
            model_format="prism",
            research_status="pending",
        ),
        paths,
    )
    inv = build_inventory(paths, config=None, with_bytes=False)
    assert inv.runnable_discovered == []  # registered rows are never adoptable
    err = _adopt_miss_error("ternary-bonsai-2-27b-mlx-2bit", inv, _ctx(paths))
    assert isinstance(err, AlreadyRegisteredError)
    assert err.code == "MODEL_ALREADY_REGISTERED"
    assert "already registered as 'ternary-bonsai-2-27b-mlx-2bit'" in err.message
    assert "research: pending" in err.message
    assert "weights present" in err.message
    assert any("spark research ternary-bonsai-2-27b-mlx-2bit" in s for s in err.remediation)


def test_adopt_miss_matches_hf_repo_case_insensitively(paths, tmp_path, monkeypatch):
    _seed_hub_llm(monkeypatch, tmp_path, "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
                   model_type="prism_hadamard_qwen35")
    save_model(
        ModelEntry(
            id="ternary-bonsai-2-27b-mlx-2bit",
            hf_repo="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
            backend="prism_ml",
            model_format="prism",
        ),
        paths,
    )
    inv = build_inventory(paths, config=None, with_bytes=False)
    err = _adopt_miss_error("prism-ml/ternary-bonsai-2-27b-mlx-2bit", inv, _ctx(paths))
    assert isinstance(err, AlreadyRegisteredError)
    assert "already registered" in err.message


def test_adopt_miss_on_missing_weights_suggests_download(paths):
    save_model(
        ModelEntry(id="gone", hf_repo="org/Gone-7B", backend="mlx_lm", model_format="mlx"),
        paths,
    )
    inv = build_inventory(paths, config=None, with_bytes=False)
    assert [e.id for e, _ in inv.missing] == ["gone"]
    err = _adopt_miss_error("gone", inv, _ctx(paths))
    assert isinstance(err, AlreadyRegisteredError)
    assert "weights missing" in err.message
    assert any("spark download org/Gone-7B" in s for s in err.remediation)


# --- cached but not servable -------------------------------------------------
def test_adopt_miss_on_embedder_says_not_servable(paths, tmp_path, monkeypatch):
    _seed_hub_embedder(monkeypatch, tmp_path)
    inv = build_inventory(paths, config=None, with_bytes=False)
    assert [m.display_id for m in inv.non_servable] == ["BAAI/bge-small"]
    err = _adopt_miss_error("BAAI/bge-small", inv, _ctx(paths))
    assert isinstance(err, ModelNotFoundError)
    assert "cannot serve it" in err.message
    assert "embedding" in err.message


# --- genuinely unknown -------------------------------------------------------
def test_adopt_miss_unknown_still_suggests_download(paths):
    inv = build_inventory(paths, config=None, with_bytes=False)
    err = _adopt_miss_error("org/Definitely-Not-Here", inv, _ctx(paths))
    assert isinstance(err, ModelNotFoundError)
    assert "Nothing adoptable matches" in err.message
    assert any("spark download org/Definitely-Not-Here" in s for s in err.remediation)


# --- end to end ----------------------------------------------------------------
def test_adopt_command_success_registers_discovered(paths, tmp_path, monkeypatch):
    _seed_hub_llm(monkeypatch, tmp_path, "mlx-community/Small-LLM-8bit")
    monkeypatch.setattr("spark.cli.app.build_context", lambda: _ctx(paths))
    result = CliRunner().invoke(adopt_command, ["mlx-community/Small-LLM-8bit"])
    assert result.exit_code == 0, result.output
    assert "adopted" in result.output
    from spark.registry import resolve_model

    assert resolve_model("small-llm-8bit", paths).hf_repo == "mlx-community/Small-LLM-8bit"


def test_adopt_command_on_registered_reports_already(paths, tmp_path, monkeypatch):
    _seed_hub_llm(monkeypatch, tmp_path, "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
                   model_type="prism_hadamard_qwen35")
    save_model(
        ModelEntry(
            id="ternary-bonsai-2-27b-mlx-2bit",
            hf_repo="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
            backend="prism_ml",
            model_format="prism",
        ),
        paths,
    )
    monkeypatch.setattr("spark.cli.app.build_context", lambda: _ctx(paths))
    result = CliRunner().invoke(adopt_command, ["ternary-bonsai-2-27b-mlx-2bit"])
    assert result.exit_code != 0
    assert isinstance(result.exception, AlreadyRegisteredError)
    assert "already registered" in str(result.exception.message)
