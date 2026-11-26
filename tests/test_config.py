from __future__ import annotations

import pytest

from spark.config.loader import _deep_merge, load_config
from spark.config.schema import SparkConfig
from spark.errors import ConfigError


def test_bundled_config_loads_all_runtimes():
    cfg = load_config()
    assert isinstance(cfg, SparkConfig)
    for name in ("mlx_lm", "mlx_vlm", "llama_cpp", "omlx", "ollama", "vllm"):
        assert name in cfg.runtimes


def test_preference_order_mlx_first():
    cfg = load_config()
    assert cfg.general.preference[0] == "mlx_lm"


def test_default_bind_is_loopback():
    cfg = load_config()
    assert cfg.general.host == "127.0.0.1"


def test_vllm_unavailable_metadata_present():
    cfg = load_config()
    assert cfg.runtimes["vllm"].requires_os == ["Linux"]
    assert cfg.runtimes["vllm"].unavailable_reason


def test_secret_env_mapping_for_mlx():
    cfg = load_config()
    assert cfg.runtimes["mlx_lm"].server.secret_env["HF_TOKEN"] == "hf_token"


def test_deep_merge_precedence():
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    over = {"a": {"y": 9, "z": 3}, "c": 4}
    out = _deep_merge(base, over)
    assert out == {"a": {"x": 1, "y": 9, "z": 3}, "b": 1, "c": 4}


def test_user_override_merges(tmp_path, monkeypatch):
    home = tmp_path / "root"
    monkeypatch.setenv("SPARK_HOME", str(home))
    from spark.config.paths import resolve_paths

    p = resolve_paths().ensure()
    p.user_config_file.write_text(
        '[general]\nhost = "0.0.0.0"\nlog_retention_days = 30\n',
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.general.host == "0.0.0.0"
    assert cfg.general.log_retention_days == 30
    # untouched defaults still present
    assert "mlx_lm" in cfg.runtimes


def test_malformed_user_config_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARK_HOME", str(tmp_path / "root"))
    from spark.config.paths import resolve_paths

    p = resolve_paths().ensure()
    p.user_config_file.write_text("this is = = not toml", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_config_validate_surfaces_reserved_override_in_registry(paths, config, monkeypatch):
    # An old entry with a bare `host` key is silently skipped by list_models;
    # `spark config validate` must name the key and point at the file to edit.
    from click.testing import CliRunner

    from spark.cli.config_cmd import config_validate

    monkeypatch.setattr("spark.cli.config_cmd.load_config", lambda: config)
    monkeypatch.setattr("spark.cli.config_cmd.resolve_paths", lambda: paths)
    bad = paths.models_dir / "legacy.toml"
    bad.write_text('id = "legacy"\n[launch_overrides]\nhost = "127.0.0.1"\n', encoding="utf-8")
    result = CliRunner().invoke(config_validate)
    err = result.exception
    assert result.exit_code != 0
    assert isinstance(err, ConfigError)
    assert "host" in err.message and "launch override" in err.message
    assert any("legacy.toml" in step for step in err.remediation)


def test_config_validate_accepts_valid_registry(paths, config, monkeypatch):
    from click.testing import CliRunner

    from spark.cli.config_cmd import config_validate
    from spark.registry import save_model
    from spark.config.schema import ModelEntry

    monkeypatch.setattr("spark.cli.config_cmd.load_config", lambda: config)
    monkeypatch.setattr("spark.cli.config_cmd.resolve_paths", lambda: paths)
    save_model(ModelEntry(id="ok", launch_overrides={"--max-tokens": "2048"}), paths)
    result = CliRunner().invoke(config_validate)
    assert result.exit_code == 0, result.output


# --- CLI presentation (compact summary, alignment, actionable hints) ----------
def test_config_show_default_is_compact_and_json_flag_dumps_everything(config, monkeypatch):
    from click.testing import CliRunner

    from spark.cli.config_cmd import config_show

    monkeypatch.setattr("spark.cli.config_cmd.load_config", lambda: config)
    runner = CliRunner()
    summary = runner.invoke(config_show, []).output
    for token in ("host", "127.0.0.1", "8095", "memory", "disk", "research", "runtimes", "probe"):
        assert token in summary, token
    # A runtime-def field must not leak into the default view.
    assert "ready_timeout_s" not in summary

    for flag in ("--json", "-j"):
        full = runner.invoke(config_show, [flag]).output
        assert "ready_timeout_s" in full
        assert "runtimes" in full


def test_config_path_aligns_values_in_one_column_and_hints(paths, monkeypatch):
    from click.testing import CliRunner

    from spark.cli import config_cmd
    from spark.config.paths import bundled_config_dir

    monkeypatch.setattr("spark.cli.config_cmd.resolve_paths", lambda: paths)
    # Long tmp paths must not wrap, or there is no single value column to assert.
    monkeypatch.setattr(config_cmd.console, "width", 240)
    out = CliRunner().invoke(config_cmd.config_path).output
    values = [
        str(bundled_config_dir()),
        str(paths.user_config_file),
        str(paths.data_dir),
        str(paths.models_dir),
        str(paths.log_dir),
        str(paths.host_profile_file),
    ]
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == len(values) + 1  # rows plus the hint
    for line, value in zip(lines[: len(values)], values, strict=True):
        assert line[2:22].strip()  # label occupies the first fixed-width column
        assert line[22:] == value  # every value starts at the same column
    assert lines[-1].startswith("→") and "spark config validate" in lines[-1]


def test_config_validate_success_is_followed_by_a_hint(paths, config, monkeypatch):
    from click.testing import CliRunner

    from spark.cli.config_cmd import config_validate

    monkeypatch.setattr("spark.cli.config_cmd.load_config", lambda: config)
    monkeypatch.setattr("spark.cli.config_cmd.resolve_paths", lambda: paths)
    out = CliRunner().invoke(config_validate).output
    assert "✓" in out and "→" in out and "spark config show" in out


def test_config_review_keeps_json_and_adds_heading_and_hints(paths, config, monkeypatch):
    from click.testing import CliRunner

    from spark.cli.config_cmd import config_review
    from spark.cli.context import SparkCtx
    from spark.config.schema import ModelEntry
    from spark.registry import save_model
    from spark.research import stage
    from spark.research.types import ResearchOutput
    from spark.telemetry import Telemetry

    save_model(ModelEntry(id="m", hf_repo="org/m"), paths)
    stage(
        paths,
        "m",
        ResearchOutput(recommended_backend="mlx_lm"),
        provider="claude",
        hf_repo="org/m",
        host_summary={"chip": "M2"},
    )
    ctx = SparkCtx(
        paths=paths,
        config=config,
        secrets=None,
        telemetry=Telemetry(paths.log_dir, session_id="test"),
        session_id="test",
    )
    monkeypatch.setattr("spark.cli.context.build_context", lambda: ctx)
    out = CliRunner().invoke(config_review, ["m"]).output
    assert "Staged research" in out
    assert "recommended_backend" in out  # full staged JSON is preserved
    assert "→" in out
    assert "spark config review m --accept" in out
    assert "spark config review m --reject" in out
