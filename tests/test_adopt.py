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
    AmbiguousModelError,
    ModelNotFoundError,
)
from spark.inventory import DiscoveredModel, Inventory, build_inventory
from spark.registry import adopt_errors as adopt_err
from spark.registry import save_model
from spark.registry.store import StoreIssue


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
    _seed_hub_llm(
        monkeypatch,
        tmp_path,
        "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
        model_type="prism_hadamard_qwen35",
    )
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
    _seed_hub_llm(
        monkeypatch,
        tmp_path,
        "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
        model_type="prism_hadamard_qwen35",
    )
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
    _seed_hub_llm(
        monkeypatch,
        tmp_path,
        "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
        model_type="prism_hadamard_qwen35",
    )
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


# --- diagnostic branches (fake inventory objects, no disk/Keychain/network) -----
def _avail(state="local", detail="", location=""):
    return SimpleNamespace(state=state, detail=detail, location=location)


def _disc(display_id, kind="llm", reason=""):
    return DiscoveredModel(
        display_id=display_id,
        source="hub",
        location="/cache/" + display_id,
        kind=kind,
        reason=reason,
    )


def _no_cache(monkeypatch):
    """Neutralize every disk probe used by the classifier."""
    monkeypatch.setattr(adopt_err, "scan_store", lambda _paths: [])
    monkeypatch.setattr(adopt_err, "repo_dir", lambda _repo: SimpleNamespace(is_dir=lambda: False))
    monkeypatch.setattr(adopt_err, "hf_cache_root", lambda: SimpleNamespace(is_dir=lambda: False))


def test_adopt_registry_exact_matches_alias_location_and_path():
    pairs = [
        (ModelEntry(id="alias-id", aliases=["Alt-Name"]), _avail()),
        (ModelEntry(id="loc-id"), _avail(location="/models/loc")),
        (ModelEntry(id="path-id", path="/models/path"), _avail()),
    ]
    assert adopt_err.adopt_registry_exact("ALT-NAME", pairs)[0].id == "alias-id"
    assert adopt_err.adopt_registry_exact("/models/LOC", pairs)[0].id == "loc-id"
    assert adopt_err.adopt_registry_exact("/models/path", pairs)[0].id == "path-id"
    assert adopt_err.adopt_registry_exact("not-there", pairs) is None


@pytest.mark.parametrize(
    ("state", "detail", "location", "expected"),
    [
        ("local", "", "/models/weights", "weights present (local: /models/weights)"),
        ("hub", "", "", "weights present (hub)"),
        ("external", "", "", "weights owned by 'mlx_lm' runtime"),
        ("unverifiable", "", "", "entry declares neither path nor hf_repo"),
        ("mystery", "", "", "weights state: mystery"),
        ("missing", "OSError", "", "weights missing (OSError)"),
    ],
)
def test_adopt_registered_error_reports_every_weight_state(state, detail, location, expected):
    entry = ModelEntry(id="m", hf_repo="org/M", backend="mlx_lm", research_status="registered")
    err = adopt_err.adopt_registered_error("org/M", entry, _avail(state, detail, location))
    assert isinstance(err, AlreadyRegisteredError)
    assert expected in err.message
    assert err.context["availability"] == state
    if state == "missing":
        assert err.remediation == [
            "Fetch weights: spark download org/M",
            "Drop the entry: spark forget m",
        ]
    else:
        assert err.remediation == ["Run it: spark run m"]


def test_adopt_registered_missing_without_repo_only_offers_drop():
    err = adopt_err.adopt_registered_error(
        "m", ModelEntry(id="m", backend="ollama"), _avail("missing")
    )
    assert "weights missing" in err.message
    assert err.remediation == ["Drop the entry: spark forget m"]


def test_adopt_miss_ambiguous_registry_query(paths, monkeypatch):
    _no_cache(monkeypatch)
    inv = Inventory(
        runnable_registered=[
            (ModelEntry(id="long-model-a"), _avail()),
            (ModelEntry(id="long-model-b"), _avail()),
        ]
    )
    err = _adopt_miss_error("long-model", inv, _ctx(paths))
    assert isinstance(err, AmbiguousModelError)
    assert err.context["matches"] == ["long-model-a", "long-model-b"]
    assert any("long-model-a, long-model-b" in s for s in err.remediation)


def test_adopt_non_servable_hits_tail_prefix_and_substring(paths, monkeypatch):
    _no_cache(monkeypatch)
    inv = Inventory(
        non_servable=[
            _disc(
                "org/LongEmbedder",
                kind="embedding",
                reason="sentence-transformers embedding model",
            ),
            _disc("prefix-hit"),
            _disc("has-substring-here"),
        ]
    )
    tail = _adopt_miss_error("longembedder", inv, _ctx(paths))
    assert "cached as 'org/LongEmbedder'" in tail.message
    assert "embedding" in tail.message
    prefix = _adopt_miss_error("prefix", inv, _ctx(paths))
    assert "cached as 'prefix-hit'" in prefix.message
    substring = _adopt_miss_error("substring", inv, _ctx(paths))
    assert "cached as 'has-substring-here'" in substring.message
    # (+N more) when several cached repos share the query
    many = _adopt_miss_error(
        "shared",
        Inventory(non_servable=[_disc("shared-a"), _disc("shared-b")]),
        _ctx(paths),
    )
    assert "(+1 more)" in many.message


@pytest.mark.parametrize("query", ["partial-id", "PARTIAL-ID", "<path>"])
def test_incomplete_store_download_says_resume(paths, monkeypatch, tmp_path, query):
    _no_cache(monkeypatch)
    if query == "<path>":
        query = str(tmp_path / "partial")
    monkeypatch.setattr(
        adopt_err,
        "scan_store",
        lambda _paths: [
            StoreIssue(
                path=tmp_path / "partial",
                model_id="partial-id",
                registered=False,
                hf_repo="org/Partial",
                incomplete_files=2,
                bytes_present=0,
            )
        ],
    )
    err = _adopt_miss_error(query, Inventory(), _ctx(paths))
    assert "incomplete download" in err.message
    assert "2 resume fragment(s)" in err.message
    assert any("spark download org/Partial" in step for step in err.remediation)


def test_incomplete_store_download_without_repo_offers_placeholder(paths, monkeypatch, tmp_path):
    _no_cache(monkeypatch)
    monkeypatch.setattr(
        adopt_err,
        "scan_store",
        lambda _paths: [
            StoreIssue(
                path=tmp_path / "p",
                model_id="local-id",
                registered=False,
                hf_repo="",
                incomplete_files=1,
                bytes_present=0,
            )
        ],
    )
    err = _adopt_miss_error("local-id", Inventory(), _ctx(paths))
    assert any("spark download <hf-repo> --id local-id" in s for s in err.remediation)


def test_orphaned_store_dir_says_no_snapshot(paths, monkeypatch, tmp_path):
    _no_cache(monkeypatch)
    monkeypatch.setattr(
        adopt_err,
        "scan_store",
        lambda _paths: [
            StoreIssue(
                path=tmp_path / "orphan",
                model_id="orphan-id",
                registered=False,
                hf_repo="",
                incomplete_files=0,
                bytes_present=0,
            )
        ],
    )
    err = _adopt_miss_error("orphan-id", Inventory(), _ctx(paths))
    assert "has no servable snapshot" in err.message
    assert "rm -rf" in " ".join(err.remediation)


def test_scan_store_error_falls_through_to_generic_miss(paths, monkeypatch):
    _no_cache(monkeypatch)

    def boom(_paths):
        raise OSError("store unreadable")

    monkeypatch.setattr(adopt_err, "scan_store", boom)
    err = _adopt_miss_error("org/Unknown", Inventory(), _ctx(paths))
    assert "Nothing adoptable matches" in err.message


def test_hub_cache_without_snapshot_says_resume(paths, monkeypatch):
    _no_cache(monkeypatch)
    monkeypatch.setattr(adopt_err, "repo_dir", lambda _repo: SimpleNamespace(is_dir=lambda: True))
    monkeypatch.setattr(adopt_err, "snapshot_dir", lambda _repo: None)
    err = _adopt_miss_error("org/Needs-Resume", Inventory(), _ctx(paths))
    assert "HF hub cache but has no complete snapshot" in err.message
    assert any("spark download org/Needs-Resume" in s for s in err.remediation)


def test_hub_cache_probe_error_falls_through(paths, monkeypatch):
    _no_cache(monkeypatch)

    def boom(_repo):
        raise OSError("cache unreadable")

    monkeypatch.setattr(adopt_err, "repo_dir", boom)
    err = _adopt_miss_error("org/Needs-Resume", Inventory(), _ctx(paths))
    assert "Nothing adoptable matches" in err.message


def test_case_insensitive_hub_hit_names_the_cached_row(paths, monkeypatch):
    _no_cache(monkeypatch)

    class FakeDir:
        def __init__(self, name):
            self.name = name

    root = SimpleNamespace(
        is_dir=lambda: True,
        iterdir=lambda: iter([FakeDir("models--Org--Mixed-Case"), FakeDir("x--y")]),
    )
    monkeypatch.setattr(adopt_err, "hf_cache_root", lambda: root)
    err = _adopt_miss_error("org/mixed-case", Inventory(), _ctx(paths))
    assert "case differs from cached 'Org/Mixed-Case'" in err.message
    assert err.context["cached"] == "Org/Mixed-Case"


def test_hub_cache_iterdir_error_falls_through(paths, monkeypatch):
    _no_cache(monkeypatch)
    root = SimpleNamespace(
        is_dir=lambda: True,
        iterdir=lambda: (_ for _ in ()).throw(OSError("gone")),
    )
    monkeypatch.setattr(adopt_err, "hf_cache_root", lambda: root)
    err = _adopt_miss_error("org/Unknown", Inventory(), _ctx(paths))
    assert "Nothing adoptable matches" in err.message


def test_unknown_query_lists_up_to_five_adoptable_rows(paths, monkeypatch):
    _no_cache(monkeypatch)
    inv = Inventory(runnable_discovered=[_disc(f"org/m{i}") for i in range(7)])
    err = _adopt_miss_error("org/Whatever", inv, _ctx(paths))
    assert any("org/m0" in step for step in err.remediation)
    assert "(+2 more)" in " ".join(err.remediation)
