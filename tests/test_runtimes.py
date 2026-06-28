from __future__ import annotations

from pathlib import Path

import pytest

from spark.config.loader import load_config
from spark.config.schema import ModelEntry
from spark.errors import BackendUnavailableError, ConfigError
from spark.runtimes import get_backend, select_backend_for


def test_mlx_launch_cmd_fills_placeholders():
    cfg = load_config()
    b = get_backend("mlx_lm", cfg)
    entry = ModelEntry(id="m", path="/models/m", model_format="mlx")
    cmd = b.build_launch_cmd(entry, "127.0.0.1", 8080)
    assert cmd[0] == "mlx_lm.server"
    assert "--model" in cmd and "/models/m" in cmd
    assert "127.0.0.1" in cmd and "8080" in cmd


def test_llama_cmd_includes_metal_offload_local_gguf(tmp_path):
    cfg = load_config()
    b = get_backend("llama_cpp", cfg)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"")
    entry = ModelEntry(id="g", path=str(gguf), model_format="gguf")
    cmd = b.build_launch_cmd(entry, "127.0.0.1", 8082)
    assert "-ngl" in cmd and "999" in cmd
    assert "--model" in cmd and str(gguf) in cmd
    assert "g" in cmd  # alias = model_id


def test_llama_cmd_uses_hf_for_repo_without_local_file():
    cfg = load_config()
    b = get_backend("llama_cpp", cfg)
    entry = ModelEntry(id="g2", hf_repo="org/Repo-GGUF", model_format="gguf")
    cmd = b.build_launch_cmd(entry, "127.0.0.1", 8082)
    assert "-hf" in cmd and "org/Repo-GGUF" in cmd
    assert "--model" not in cmd


def test_llama_raises_without_file_or_repo():
    from spark.errors import ModelNotFoundError

    cfg = load_config()
    b = get_backend("llama_cpp", cfg)
    with pytest.raises(ModelNotFoundError):
        b.build_launch_cmd(ModelEntry(id="g3", model_format="gguf"), "127.0.0.1", 8082)


def test_launch_overrides_appended_as_flags():
    cfg = load_config()
    b = get_backend("mlx_lm", cfg)
    entry = ModelEntry(id="m", path="/m",
                       launch_overrides={"--max-tokens": "4096", "--trust-remote-code": ""})
    cmd = b.build_launch_cmd(entry, "127.0.0.1", 8080)
    assert cmd[-3:] == ["--max-tokens", "4096", "--trust-remote-code"] or (
        "--max-tokens" in cmd and "4096" in cmd and "--trust-remote-code" in cmd
    )


def test_llama_overrides_appended_as_flags():
    cfg = load_config()
    b = get_backend("llama_cpp", cfg)
    entry = ModelEntry(id="g", hf_repo="org/R-GGUF", model_format="gguf",
                       launch_overrides={"-c": "8192"})
    cmd = b.build_launch_cmd(entry, "127.0.0.1", 8082)
    assert cmd[-2:] == ["-c", "8192"]


def test_unknown_placeholder_raises_config_error():
    cfg = load_config()
    cfg.runtimes["mlx_lm"].server.args_template.append("{nope}")
    b = get_backend("mlx_lm", cfg)
    with pytest.raises(ConfigError):
        b.build_launch_cmd(ModelEntry(id="m", path="/m"), "h", 1)


def test_health_url():
    cfg = load_config()
    b = get_backend("llama_cpp", cfg)
    assert b.health_url("127.0.0.1", 8082) == "http://127.0.0.1:8082/health"


def test_select_respects_registered_backend_when_available():
    cfg = load_config()
    entry = ModelEntry(id="m", backend="llama_cpp", model_format="gguf")
    chosen = select_backend_for(entry, cfg, ["mlx_lm", "llama_cpp"])
    assert chosen == "llama_cpp"


def test_select_falls_back_to_preference_order():
    cfg = load_config()
    entry = ModelEntry(id="m", model_format="mlx")
    chosen = select_backend_for(entry, cfg, ["llama_cpp", "mlx_lm", "omlx"])
    # mlx_lm beats llama_cpp/omlx in preference and is format-compatible
    assert chosen == "mlx_lm"


def test_omlx_cmd_uses_staged_model_dir(monkeypatch, tmp_path):
    # oMLX serves a directory of models, not a single positional model.
    monkeypatch.setenv("SPARK_HOME", str(tmp_path))
    cfg = load_config()
    b = get_backend("omlx", cfg)
    entry = ModelEntry(id="q05", model_format="mlx", path="/models/q05")
    cmd = b.build_launch_cmd(entry, "127.0.0.1", 8083)
    assert cmd[0] == "omlx" and cmd[1] == "serve"
    i = cmd.index("--model-dir")
    assert cmd[i + 1].endswith("/run/omlx/q05")  # private per-model staging dir
    assert "--memory-guard" in cmd and "safe" in cmd
    assert "127.0.0.1" in cmd and "8083" in cmd
    assert "/models/q05" not in cmd  # the old positional form is gone


def test_omlx_prepare_stages_one_model_symlink(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARK_HOME", str(tmp_path))
    src = tmp_path / "weights"
    src.mkdir()
    (src / "config.json").write_text("{}")  # oMLX's model-dir marker
    cfg = load_config()
    b = get_backend("omlx", cfg)
    entry = ModelEntry(id="q05", model_format="mlx", path=str(src))
    b.prepare(entry)
    link = Path(b.resolve_model_dir(entry)) / "q05"
    assert link.is_symlink() and link.resolve() == src.resolve()


def test_omlx_prepare_aborts_without_local_weights(monkeypatch, tmp_path):
    from spark.errors import RuntimeBackendError

    monkeypatch.setenv("SPARK_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))  # isolate from real cache
    cfg = load_config()
    b = get_backend("omlx", cfg)
    entry = ModelEntry(id="nope", model_format="mlx", hf_repo="org/not-cached-xyz")
    with pytest.raises(RuntimeBackendError):
        b.prepare(entry)


def test_select_raises_when_nothing_available():
    cfg = load_config()
    entry = ModelEntry(id="m", model_format="gguf")
    with pytest.raises(BackendUnavailableError):
        select_backend_for(entry, cfg, [])


def test_vlm_model_routes_to_mlx_vlm():
    cfg = load_config()
    entry = ModelEntry(id="v", model_format="mlx-vlm")
    # mlx_lm (format 'mlx') is higher preference but incompatible -> mlx_vlm wins
    chosen = select_backend_for(entry, cfg, ["mlx_lm", "mlx_vlm"])
    assert chosen == "mlx_vlm"


def test_openai_base_url():
    cfg = load_config()
    assert get_backend("mlx_lm", cfg).openai_base_url("127.0.0.1", 8080) == \
        "http://127.0.0.1:8080/v1"


def test_ollama_attaches():
    cfg = load_config()
    assert get_backend("ollama", cfg).attach_if_running is True
