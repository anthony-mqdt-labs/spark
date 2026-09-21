"""prism_ml backend: Hadamard packs resolve to snapshot dirs and launch the shim.

Pins the 2026-09-21 work: stock mlx_lm answers "Model type
prism_hadamard_qwen35 not supported" for these packs, so they route to the
vendor loader + stdlib-HTTP shim instead. The shim itself (stdlib + mlx, run by
another interpreter) is verified live, not here.
"""

from __future__ import annotations

import json

import pytest

from spark.config.schema import ModelEntry
from spark.errors import RuntimeBackendError
from spark.runtimes import get_backend, select_backend_for
from spark.runtimes.prism_ml import PrismMlBackend, shim_path


def _pack(tmp_path, *, complete=True) -> object:
    d = tmp_path / "pack"
    (d / "runtime").mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"model_type": "prism_hadamard_qwen35"}))
    (d / "model.safetensors").write_bytes(b"w" * 64)
    (d / "tokenizer.json").write_bytes(b"{}")
    (d / "chat_template.jinja").write_text("{{ messages }}")
    (d / "runtime" / "artifact.py").write_text("# vendor loader\n")
    if not complete:
        (d / "tokenizer.json").unlink()
    return d


def test_shim_is_shipped_package_data():
    p = shim_path()
    assert p.is_file() and p.name == "prism_shim.py"


def test_pack_dir_prefers_entry_path(paths, tmp_path, config):
    d = _pack(tmp_path)
    entry = ModelEntry(id="p", path=str(d), model_format="prism")
    assert PrismMlBackend(config.runtimes["prism_ml"], config).pack_dir(entry) == d


def test_pack_dir_missing_weights_is_actionable(paths, tmp_path, config):
    entry = ModelEntry(id="p", hf_repo="org/Nope", model_format="prism")
    with pytest.raises(RuntimeBackendError):
        PrismMlBackend(config.runtimes["prism_ml"], config).pack_dir(entry)


def test_prepare_rejects_incomplete_packs(paths, tmp_path, config):
    d = _pack(tmp_path, complete=False)
    entry = ModelEntry(id="p", path=str(d), model_format="prism")
    with pytest.raises(RuntimeBackendError) as exc:
        PrismMlBackend(config.runtimes["prism_ml"], config).prepare(entry)
    assert "tokenizer.json" in str(exc.value.message)


def test_launch_cmd_runs_shim_on_configured_interpreter(paths, tmp_path, config):
    d = _pack(tmp_path)
    entry = ModelEntry(id="p", path=str(d), model_format="prism")
    cmd = get_backend("prism_ml", config).build_launch_cmd(entry, "127.0.0.1", 8098)
    assert cmd[0] == config.runtimes["prism_ml"].server.binary
    assert cmd[1] == str(shim_path())
    assert "--pack" in cmd and str(d) in cmd
    assert "--model-id" in cmd and "p" in cmd


def test_prism_format_routes_to_prism_backend(paths, config):
    entry = ModelEntry(id="p", model_format="prism")
    assert select_backend_for(entry, config, ["mlx_lm", "prism_ml"]) == "prism_ml"


def test_mlx_format_never_routes_to_prism(paths, config):
    entry = ModelEntry(id="m", model_format="mlx")
    assert select_backend_for(entry, config, ["mlx_lm", "prism_ml"]) == "mlx_lm"


def test_prism_runtime_is_registered_in_config(config):
    assert config.runtimes["prism_ml"].model_format == "prism"
    assert config.runtimes["prism_ml"].server.default_port == 8098


def test_prism_publishes_registry_id_not_snapshot_path(paths, tmp_path, config):
    """The shim ignores the request `model` field, so the published API id is
    the registry id — never a snapshot path consumers would cargo-cult."""
    d = _pack(tmp_path)
    entry = ModelEntry(id="p", path=str(d), model_format="prism")
    backend = get_backend("prism_ml", config)
    assert backend.api_model_id(entry) == "p"
    assert backend.resolve_model_ref(entry) == str(d)


def test_default_api_id_is_the_launch_ref(paths, config):
    entry = ModelEntry(id="m", hf_repo="org/M", model_format="mlx")
    backend = get_backend("mlx_lm", config)
    assert backend.api_model_id(entry) == "org/M"


def test_package_import_registers_prism_adapter():
    """A backend missing from runtimes/__init__ silently degrades to the
    generic launcher (observed: argv collapsed to [binary], instant code-0
    exit, 4 wasted restarts). This fails in a fresh interpreter where no test
    import has triggered registration as a side effect."""
    import subprocess
    import sys

    code = (
        "from spark.runtimes import get_backend\n"
        "from spark.runtimes.base import _REGISTRY\n"
        "assert 'prism_ml' in _REGISTRY, sorted(_REGISTRY)\n"
        "print('registered:', sorted(_REGISTRY))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "prism_ml" in proc.stdout
