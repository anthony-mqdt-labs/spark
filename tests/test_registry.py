from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from spark.config.schema import DFlashProfile, ModelEntry
from spark.errors import AmbiguousModelError, ModelNotFoundError
from spark.registry import list_models, resolve_model, save_model


def _seed(paths, *ids):
    for i in ids:
        save_model(ModelEntry(id=i, model_format="mlx", quant="q4"), paths)


def test_save_and_list_roundtrip(paths):
    _seed(paths, "alpha-9b", "beta-7b")
    ids = {m.id for m in list_models(paths)}
    assert ids == {"alpha-9b", "beta-7b"}


def test_resolve_exact(paths):
    _seed(paths, "alpha-9b", "beta-7b")
    assert resolve_model("alpha-9b", paths).id == "alpha-9b"


def test_resolve_alias(paths):
    save_model(ModelEntry(id="alpha-9b", aliases=["ornith9b"]), paths)
    assert resolve_model("ornith9b", paths).id == "alpha-9b"


def test_resolve_unique_prefix(paths):
    _seed(paths, "alpha-9b", "beta-7b")
    assert resolve_model("alph", paths).id == "alpha-9b"


def test_resolve_ambiguous_raises(paths):
    _seed(paths, "qwen-7b", "qwen-14b")
    with pytest.raises(AmbiguousModelError):
        resolve_model("qwen", paths)


def test_resolve_miss_raises(paths):
    _seed(paths, "alpha-9b")
    with pytest.raises(ModelNotFoundError):
        resolve_model("zzz", paths)


def test_resolve_empty_registry_raises(paths):
    with pytest.raises(ModelNotFoundError):
        resolve_model("anything", paths)


def test_entry_persists_fields(paths):
    save_model(
        ModelEntry(
            id="m",
            hf_repo="org/M",
            quant="q4",
            params_billions=9.0,
            launch_overrides={"--ctx-size": "8192"},
        ),
        paths,
    )
    m = resolve_model("m", paths)
    assert m.hf_repo == "org/M"
    assert m.params_billions == 9.0
    assert m.launch_overrides == {"--ctx-size": "8192"}


def test_dflash_profile_roundtrips(paths):
    """A [dflash] profile survives save -> load through the custom TOML emitter."""
    save_model(
        ModelEntry(
            id="ornith",
            backend="mlx_lm",
            model_format="mlx",
            dflash=DFlashProfile(
                backend="mlx_vlm",
                launch_overrides={"--draft-model": "org/Draft", "--draft-kind": "dflash"},
            ),
        ),
        paths,
    )
    m = resolve_model("ornith", paths)
    assert m.backend == "mlx_lm"  # default path unchanged
    assert m.dflash is not None
    assert m.dflash.backend == "mlx_vlm"
    assert m.dflash.launch_overrides["--draft-model"] == "org/Draft"


def test_with_dflash_swaps_backend_and_merges_overrides():
    entry = ModelEntry(
        id="ornith",
        backend="mlx_lm",
        launch_overrides={"--max-tokens": "2048"},
        dflash=DFlashProfile(
            backend="mlx_vlm",
            launch_overrides={"--draft-model": "org/Draft", "--draft-kind": "dflash"},
        ),
    )
    d = entry.with_dflash()
    assert d.backend == "mlx_vlm"  # profile backend wins
    assert d.launch_overrides["--draft-model"] == "org/Draft"
    assert d.launch_overrides["--max-tokens"] == "2048"  # base overrides preserved
    assert entry.backend == "mlx_lm"  # original untouched (copy)


def test_no_dflash_profile_by_default():
    assert ModelEntry(id="m").dflash is None


# --- invalid entries are skipped loudly ----------------------------------------
_BAD_ENTRY = 'id = "legacy"\n[launch_overrides]\nhost = "127.0.0.1"\n'


def test_list_models_logs_full_validation_message_and_notifies_callback(paths, tmp_path):
    from spark.telemetry import init_telemetry

    init_telemetry(tmp_path / "logs", "invalid-registry")
    bad = paths.models_dir / "legacy.toml"
    bad.write_text(_BAD_ENTRY, encoding="utf-8")

    seen: list[tuple[object, str]] = []
    assert list_models(paths, on_invalid=lambda p, msg: seen.append((p, msg))) == []

    assert len(seen) == 1 and seen[0][0] == bad
    assert "reserved" in seen[0][1] and "host" in seen[0][1]

    records = [
        json.loads(line)
        for f in (tmp_path / "logs" / "registry").glob("*.jsonl")
        for line in f.read_text(encoding="utf-8").splitlines()
    ]
    warnings = [r for r in records if r.get("event") == "entry_unreadable"]
    assert len(warnings) == 1
    assert warnings[0]["file"] == "legacy.toml"
    assert "reserved" in warnings[0]["message"]
    assert "host" in warnings[0]["message"]


def test_list_command_surfaces_invalid_registry_entry(paths, config, monkeypatch):
    from click.testing import CliRunner

    from spark.cli.app import list_command

    (paths.models_dir / "legacy.toml").write_text(_BAD_ENTRY, encoding="utf-8")

    class _QuietTelemetry:
        def info(self, *a, **k):
            pass

        def warn(self, *a, **k):
            pass

        def error(self, *a, **k):
            pass

    ctx = SimpleNamespace(paths=paths, config=config, telemetry=_QuietTelemetry())
    monkeypatch.setattr("spark.cli.app.build_context", lambda: ctx)
    monkeypatch.setattr("spark.cli.context.refresh_catalog", lambda *a, **k: None)

    result = CliRunner().invoke(list_command)
    assert result.exit_code == 0, result.output
    assert "legacy.toml" in result.output
    assert "reserved" in result.output and "host" in result.output
