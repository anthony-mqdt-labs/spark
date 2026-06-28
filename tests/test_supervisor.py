from __future__ import annotations

import shutil

import pytest

from spark.config.loader import load_config
from spark.config.schema import DetectSpec, ModelEntry, RuntimeDef, ServerSpec
from spark.errors import RecoveryExhaustedError
from spark.probe.host import HostProfile, RuntimeStatus
from spark.runner import Supervisor
from spark.runtimes.base import Backend
from helpers import FakeStore


def test_pick_port_returns_free():
    from spark.runner.supervisor import _pick_port, _port_free

    p = _pick_port("127.0.0.1", 8080, 8080, 8099)
    assert _port_free("127.0.0.1", p)


def _profile():
    return HostProfile(
        os="Darwin", arch="arm64", chip="Apple M2",
        total_memory_bytes=16 * 2**30, cpu_count=8, probed_at=0.0,
        runtimes={"boom": RuntimeStatus(name="boom", available=True)},
    )


def test_supervisor_exhausts_restarts_on_immediate_exit(paths):
    false_bin = shutil.which("false") or "/usr/bin/false"
    cfg = load_config()
    cfg.runtimes["boom"] = RuntimeDef(
        name="boom",
        detect=DetectSpec(binary=false_bin),
        server=ServerSpec(
            binary=false_bin, args_template=[], health_path="/nope",
            default_port=8137, ready_timeout_s=2.0,
        ),
    )
    cfg.supervisor.max_restarts = 1
    cfg.supervisor.backoff_base_s = 0.01
    cfg.supervisor.backoff_max_s = 0.01

    backend = Backend(cfg.runtimes["boom"], cfg)
    entry = ModelEntry(id="boom-model")  # no path -> no fs check; no params -> no mem gate

    sup = Supervisor(backend, entry, cfg, _profile(), paths, FakeStore())
    with pytest.raises(RecoveryExhaustedError):
        sup.run()


def test_supervisor_attach_mode_does_not_spawn(paths, monkeypatch):
    """A daemon-style backend that is already healthy is attached, not spawned."""
    cfg = load_config()
    rt = RuntimeDef(
        name="daemon",
        detect=DetectSpec(binary="true"),
        server=ServerSpec(binary="true", attach_if_running=True, default_port=18999),
    )
    cfg.runtimes["daemon"] = rt
    backend = Backend(rt, cfg)
    sup = Supervisor(backend, ModelEntry(id="m"), cfg, _profile(), paths, FakeStore())

    monkeypatch.setattr(sup, "_is_healthy", lambda url: True)
    spawned = {"v": False}
    monkeypatch.setattr(sup, "_spawn", lambda *a, **k: spawned.__setitem__("v", True))
    sup._stop.set()  # make the attached monitor return immediately

    result = sup.run()
    assert not spawned["v"]
    assert result.endpoint.endswith("/v1")
    assert result.port == 18999


def test_supervisor_prepare_hook_is_called(paths, monkeypatch):
    cfg = load_config()
    rt = RuntimeDef(
        name="prep", detect=DetectSpec(binary="true"),
        server=ServerSpec(binary="true", attach_if_running=True, default_port=18998),
    )
    cfg.runtimes["prep"] = rt
    backend = Backend(rt, cfg)
    called = {"v": False}
    monkeypatch.setattr(backend, "prepare",
                        lambda entry, secrets=None: called.__setitem__("v", True))
    sup = Supervisor(backend, ModelEntry(id="m"), cfg, _profile(), paths, FakeStore())
    monkeypatch.setattr(sup, "_is_healthy", lambda url: True)
    sup._stop.set()
    sup.run()
    assert called["v"]


def test_supervisor_secret_env_only_logs_var_names(paths):
    """resolve_env must inject the value but the log event records only the name."""
    false_bin = shutil.which("false") or "/usr/bin/false"
    cfg = load_config()
    rt = RuntimeDef(
        name="boom2",
        detect=DetectSpec(binary=false_bin),
        server=ServerSpec(
            binary=false_bin, secret_env={"HF_TOKEN": "hf_token"},
            default_port=8138, ready_timeout_s=1.0,
        ),
    )
    cfg.runtimes["boom2"] = rt
    store = FakeStore()
    store.set("hf_token", "secret-abc-123")
    backend = Backend(rt, cfg)
    sup = Supervisor(backend, ModelEntry(id="m"), cfg, _profile(), paths, store)
    env = sup._child_env()
    assert env["HF_TOKEN"] == "secret-abc-123"
