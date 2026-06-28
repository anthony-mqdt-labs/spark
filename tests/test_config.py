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
