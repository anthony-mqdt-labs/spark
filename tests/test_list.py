"""List-view UX: scannable table, actionable pointers, progressive disclosure.

CliRunner drives the real command against an isolated SPARK_HOME and HF cache —
no server, network or Keychain access (the secret store is stubbed out).
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from spark.cli.app import cli, list_command
from spark.config.paths import resolve_paths
from spark.config.schema import ModelEntry
from spark.inventory import DiscoveredModel
from spark.registry import save_model


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    monkeypatch.setenv("SPARK_HOME", str(h))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hfcache"))
    # Listing never reads secrets; keep the Keychain out of the test entirely.
    monkeypatch.setattr("spark.cli.context.get_secret_store", lambda paths: object())
    monkeypatch.setattr("spark.cli.context.refresh_catalog", lambda *a, **k: None)
    return resolve_paths().ensure()


def _seed_store_model(home, model_id: str, size: int):
    d = home.model_store_dir / model_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.safetensors").write_bytes(b"w" * size)
    return d


def test_list_table_uses_research_header_and_compact_labels(home):
    for mid, status in (("tuned-one", "registered"), ("pending-one", "pending")):
        d = _seed_store_model(home, mid, 2**16)
        save_model(
            ModelEntry(
                id=mid,
                path=str(d),
                backend="mlx_lm",
                model_format="mlx",
                research_status=status,
            ),
            home,
        )
    result = CliRunner().invoke(list_command)
    assert result.exit_code == 0, result.output
    assert "research" in result.output
    assert "status" not in result.output
    assert "tuned" in result.output and "pending" in result.output


def test_list_uses_kib_and_never_reports_zero_mib(home):
    d = _seed_store_model(home, "small", 2**16)
    save_model(ModelEntry(id="small", path=str(d), backend="mlx_lm", model_format="mlx"), home)
    result = CliRunner().invoke(list_command)
    assert result.exit_code == 0, result.output
    assert "64 KiB" in result.output
    assert "0 MiB" not in result.output


def test_empty_state_offers_download_and_adopt(home):
    result = CliRunner().invoke(list_command)
    assert result.exit_code == 0, result.output
    assert "nothing runnable on disk" in result.output
    assert "spark download <repo>" in result.output
    assert "spark adopt <path>" in result.output
    assert result.output.count("→") == 2


def test_bare_spark_lands_on_the_list_view(home):
    assert CliRunner().invoke(cli, []).output == CliRunner().invoke(list_command).output


def test_all_shows_missing_detail_and_truncates_long_reasons(home, monkeypatch):
    save_model(
        ModelEntry(id="gone", hf_repo="org/Gone-7B", backend="mlx_lm", model_format="mlx"),
        home,
    )
    long_reason = "embedding/classifier model, not a chat model " * 6
    from spark.inventory import build_inventory as real_build

    def _fake_build(paths, config, **kw):
        return _with_non_servable_reason(real_build(paths, config, **kw), long_reason)

    monkeypatch.setattr("spark.inventory.build_inventory", _fake_build)

    detailed = CliRunner().invoke(list_command, ["--all"])
    assert detailed.exit_code == 0, detailed.output
    assert "spark download org/Gone-7B" in detailed.output
    assert "spark forget gone" in detailed.output
    assert "org/BigEmbedder" in detailed.output
    assert "…" in detailed.output
    assert long_reason not in detailed.output

    brief = CliRunner().invoke(list_command)
    assert "hidden, see --all" in brief.output
    assert "org/BigEmbedder" not in brief.output
    assert "spark download org/Gone-7B" not in brief.output


def _with_non_servable_reason(inv, reason):
    inv.non_servable = [
        DiscoveredModel(
            display_id="org/BigEmbedder",
            source="hub",
            location="/cache/org/BigEmbedder",
            kind="embedding",
            reason=reason,
        )
    ]
    return inv


def test_store_issues_are_scannable(home):
    d = home.model_store_dir / "partial-id"
    dl = d / ".cache" / "huggingface" / "download"
    dl.mkdir(parents=True)
    (dl / "blob.incomplete").write_bytes(b"z" * 8)
    (d / "model.safetensors").write_bytes(b"w" * (2**16))

    result = CliRunner().invoke(list_command)
    assert result.exit_code == 0, result.output
    bullets = [ln for ln in result.output.splitlines() if "•" in ln]
    assert len(bullets) == 1
    assert "partial (1 resume fragment(s))" in result.output
    assert "64 KiB on disk" in result.output
    assert "→ fix:" in result.output
    flat = " ".join(result.output.split())
    assert "spark download <hf-repo> --id partial-id" in flat
    assert "rm -rf" in flat
