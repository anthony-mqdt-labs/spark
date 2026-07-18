"""darkcore runtime def — the patchwork router as a spark-supervised child.

Pure-TOML runtime (generic Backend): these tests pin the launch contract so
config drift breaks loudly, not at 2am mid-route."""
from __future__ import annotations

from spark.config.loader import load_config
from spark.config.schema import ModelEntry
from spark.runtimes import get_backend

MLXPY = "$(uv tool dir)/mlx-lm/bin/python"
DARKCORE_DIR = "${DARKCORE_ROUTER_DIR}"


def _entry() -> ModelEntry:
    return ModelEntry(id="darkcore-router", model_format="any", backend="darkcore")


def test_darkcore_runtime_loads():
    cfg = load_config()
    assert "darkcore" in cfg.runtimes
    rt = cfg.runtimes["darkcore"]
    assert rt.model_format == "any"
    assert rt.requires_os == ["Darwin"] and rt.requires_arch == ["arm64"]


def test_darkcore_launch_cmd_is_module_invocation():
    cfg = load_config()
    b = get_backend("darkcore", cfg)
    cmd = b.build_launch_cmd(_entry(), "127.0.0.1", 8000)
    assert cmd[0] == MLXPY
    assert cmd[1:3] == ["-m", "darkcore.server"]
    assert "--host" in cmd and "127.0.0.1" in cmd
    assert "--port" in cmd and "8000" in cmd


def test_darkcore_env_carries_pythonpath():
    """cwd-independence: the darkcore package resolves via PYTHONPATH, and the
    router's own paths are __file__-derived."""
    cfg = load_config()
    b = get_backend("darkcore", cfg)
    assert b.extra_env().get("PYTHONPATH") == DARKCORE_DIR


def test_darkcore_health_and_base_urls():
    cfg = load_config()
    b = get_backend("darkcore", cfg)
    assert b.health_url("127.0.0.1", 8000) == "http://127.0.0.1:8000/health"
    assert b.openai_base_url("127.0.0.1", 8000) == "http://127.0.0.1:8000/v1"


def test_darkcore_is_spawned_not_attached():
    """The whole point of seam 5: spark OWNS the process (restart on crash),
    unlike the attach-only relay backend."""
    cfg = load_config()
    b = get_backend("darkcore", cfg)
    assert b.attach_if_running is False
    assert b.ready_timeout_s() == 300.0
