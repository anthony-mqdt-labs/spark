from __future__ import annotations

import os
import shutil
import socket

import pytest

from spark.config.loader import load_config
from spark.config.schema import DetectSpec, ModelEntry, RuntimeDef, ServerSpec
from spark.errors import (
    InsufficientDiskError,
    RecoveryExhaustedError,
    WeightsMissingError,
)
from spark.probe.host import HostProfile, RuntimeStatus
from spark.runner import Supervisor
from spark.runtimes.base import Backend
from helpers import FakeStore


def test_pick_port_returns_free():
    from spark.runner.supervisor import _pick_port, _port_free

    p = _pick_port("127.0.0.1", 8080, 8080, 8099)
    assert _port_free("127.0.0.1", p)


def test_port_helpers_are_owned_by_ports_module():
    """The supervisor re-exports the seam; both import paths name one function."""
    from spark.runner.ports import _pick_port, _port_free
    from spark.runner.supervisor import _pick_port as sup_pick
    from spark.runner.supervisor import _port_free as sup_free

    assert sup_pick is _pick_port
    assert sup_free is _port_free


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_reserve_port_holds_the_port_until_closed():
    """reserve_port binds + listens and returns the OPEN socket: a same-user
    process cannot take the port between selection and child exec."""
    from spark.runner.ports import reserve_port

    port = _free_port()
    sock = reserve_port("127.0.0.1", port)
    try:
        with pytest.raises(OSError):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as other:
                other.bind(("127.0.0.1", port))
                other.listen(1)
    finally:
        sock.close()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as other:
        other.bind(("127.0.0.1", port))
        other.listen(1)  # released again


class _FdBackend(Backend):
    """A backend whose launched child can adopt a pre-bound listen fd."""

    @property
    def accepts_reserved_fd(self) -> bool:
        return True


def _fd_supervisor(paths, cfg=None):
    """Supervisor with a reserved-fd backend and a pinned, known-free port."""
    cfg = cfg or load_config()
    port = _free_port()
    cfg.general.port_range = (port, port)
    rt = RuntimeDef(
        name="fd",
        detect=DetectSpec(binary="/bin/true"),
        server=ServerSpec(
            binary="/bin/true",
            args_template=[],
            health_path="/health",
            default_port=port,
            ready_timeout_s=1.0,
        ),
    )
    cfg.runtimes["fd"] = rt
    backend = _FdBackend(rt, cfg)
    return Supervisor(backend, ModelEntry(id="m"), cfg, _profile(), paths, FakeStore()), port


class _FakePump:
    def write(self, _text):
        pass

    def start_pipe(self):
        read_fd, write_fd = os.pipe()
        os.close(read_fd)
        return write_fd


def _capture_popen(monkeypatch, sup, seen):
    """Replace Popen with a recorder that also notes whether the fd is still open."""
    sup._log_pump = _FakePump()  # noqa: SLF001
    monkeypatch.setattr(sup, "_open_server_log", lambda: object())  # noqa: SLF001

    def _popen(cmd, **kwargs):
        seen["pass_fds"] = kwargs.get("pass_fds")
        seen["env"] = dict(kwargs.get("env") or {})
        seen["fd_open_at_popen"] = (
            sup._reserved_sock is not None and sup._reserved_sock.fileno() >= 0  # noqa: SLF001
        )
        if sup._reserved_sock is not None:  # noqa: SLF001
            seen["fd"] = sup._reserved_sock.fileno()  # noqa: SLF001
        return object()

    monkeypatch.setattr("spark.runner.supervisor.subprocess.Popen", _popen)


def test_preflight_reserves_port_held_until_popen(paths, monkeypatch):
    """Preflight holds the port through exec and hands the fd via pass_fds/env."""
    sup, port = _fd_supervisor(paths)
    assert sup._preflight() == port  # noqa: SLF001
    sock = sup._reserved_sock  # noqa: SLF001
    assert sock is not None
    with pytest.raises(OSError):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as other:
            other.bind(("127.0.0.1", port))
            other.listen(1)

    seen: dict = {}
    _capture_popen(monkeypatch, sup, seen)
    sup._spawn(["true"], {})  # noqa: SLF001
    assert seen["fd_open_at_popen"] is True
    assert seen["pass_fds"] == (seen["fd"],)
    assert seen["env"]["SPARK_LISTEN_FD"] == str(seen["fd"])
    assert sock.fileno() == -1  # parent dropped its copy after Popen
    assert sup._reserved_sock is None  # noqa: SLF001


def test_restart_re_reserves_the_port(paths, monkeypatch):
    """Each spawn gets a live reservation, including spawns after a restart."""
    sup, port = _fd_supervisor(paths)
    assert sup._preflight() == port  # noqa: SLF001
    events: list[dict] = []
    sup._log_pump = _FakePump()  # noqa: SLF001
    monkeypatch.setattr(sup, "_open_server_log", lambda: object())  # noqa: SLF001

    def _popen(cmd, **kwargs):
        events.append({"pass_fds": kwargs.get("pass_fds"), "env": dict(kwargs.get("env") or {})})
        return object()

    monkeypatch.setattr("spark.runner.supervisor.subprocess.Popen", _popen)
    env = {}
    sup._spawn(["true"], env)  # noqa: SLF001
    assert sup._reserved_sock is None  # noqa: SLF001  (first copy dropped)
    sup._spawn(["true"], env)  # noqa: SLF001

    assert len(events) == 2
    for event in events:
        assert len(event["pass_fds"]) == 1
        assert event["env"]["SPARK_LISTEN_FD"] == str(event["pass_fds"][0])
    assert sup._reserved_sock is None  # noqa: SLF001
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as other:
        other.bind(("127.0.0.1", port))
        other.listen(1)


def test_non_proxy_backend_gets_no_pass_fds(paths, monkeypatch):
    """Backends that bind their own port get no fd and no stale env var.

    They keep the small bind-then-release window the runtime alone can close.
    """
    cfg = load_config()
    port = _free_port()
    cfg.general.port_range = (port, port)
    rt = RuntimeDef(
        name="plain",
        detect=DetectSpec(binary="/bin/true"),
        server=ServerSpec(binary="/bin/true", default_port=port),
    )
    cfg.runtimes["plain"] = rt
    sup = Supervisor(Backend(rt, cfg), ModelEntry(id="m"), cfg, _profile(), paths, FakeStore())
    assert sup._preflight() == port  # noqa: SLF001
    assert sup._reserved_sock is None  # noqa: SLF001
    # Not held by spark: a same-user process may still bind it.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as other:
        other.bind(("127.0.0.1", port))
        other.listen(1)

    seen: dict = {}
    _capture_popen(monkeypatch, sup, seen)
    sup._spawn(["true"], {"SPARK_LISTEN_FD": "999"})  # noqa: SLF001
    assert seen["pass_fds"] == ()
    assert "SPARK_LISTEN_FD" not in seen["env"]
    assert seen["fd_open_at_popen"] is False


def test_reservation_released_on_early_exception(paths, monkeypatch):
    """Preflight succeeded but launch aborted: run()'s finally must free the port."""
    sup, port = _fd_supervisor(paths)

    def boom(*_a, **_k):
        raise RuntimeError("pre-launch abort")

    monkeypatch.setattr(sup.backend, "prepare", boom)
    with pytest.raises(RuntimeError, match="pre-launch abort"):
        sup.run()
    assert sup._reserved_sock is None  # noqa: SLF001
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as other:
        other.bind(("127.0.0.1", port))
        other.listen(1)  # the reservation did not leak past the failed launch


def _profile():
    return HostProfile(
        os="Darwin",
        arch="arm64",
        chip="Apple M2",
        total_memory_bytes=16 * 2**30,
        cpu_count=8,
        probed_at=0.0,
        runtimes={"boom": RuntimeStatus(name="boom", available=True)},
    )


def test_supervisor_exhausts_restarts_on_immediate_exit(paths):
    false_bin = shutil.which("false") or "/usr/bin/false"
    cfg = load_config()
    cfg.runtimes["boom"] = RuntimeDef(
        name="boom",
        detect=DetectSpec(binary=false_bin),
        server=ServerSpec(
            binary=false_bin,
            args_template=[],
            health_path="/nope",
            default_port=8137,
            ready_timeout_s=2.0,
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
        name="prep",
        detect=DetectSpec(binary="true"),
        server=ServerSpec(binary="true", attach_if_running=True, default_port=18998),
    )
    cfg.runtimes["prep"] = rt
    backend = Backend(rt, cfg)
    called = {"v": False}
    monkeypatch.setattr(
        backend, "prepare", lambda entry, secrets=None: called.__setitem__("v", True)
    )
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
            binary=false_bin,
            secret_env={"HF_TOKEN": "hf_token"},
            default_port=8138,
            ready_timeout_s=1.0,
        ),
    )
    cfg.runtimes["boom2"] = rt
    store = FakeStore()
    store.set("hf_token", "secret-abc-123")
    backend = Backend(rt, cfg)
    sup = Supervisor(backend, ModelEntry(id="m"), cfg, _profile(), paths, store)
    env = sup._child_env()
    assert env["HF_TOKEN"] == "secret-abc-123"
    # The event must name only the env var, never its value. This is the actual
    # leak this test claims to guard (the old version only checked injection).
    window = sup.log.context_window()
    assert "secret-abc-123" not in repr(window)
    events = [e for e in window if e.get("event") == "secret_env_resolved"]
    assert events and events[-1].get("vars_set") == ["HF_TOKEN"]


# --- weight availability gate ---------------------------------------------------
def _mlx_backend(cfg):
    """A real mlx_lm RuntimeDef (no binary needed — preflight touches no process)."""
    rt = cfg.runtimes.get("mlx_lm") or RuntimeDef(
        name="mlx_lm",
        detect=DetectSpec(binary="/bin/true"),
        server=ServerSpec(binary="/bin/true", default_port=8092),
    )
    cfg.runtimes["mlx_lm"] = rt
    return Backend(rt, cfg)


def _free_gib(monkeypatch, gib: float):
    """Pin the free-space reading the disk gate sees."""
    import spark.budget as budget

    monkeypatch.setattr(budget, "free_disk_bytes", lambda path: int(gib * 2**30))


def test_preflight_refuses_when_said_weights_are_absent(paths, monkeypatch, tmp_path):
    """The incident: entry claims a repo, cache is empty -> refuse, don't fetch.

    Launching anyway hands the runtime a repo id; the fetch happens on the first
    request, after spark has already reported the server ready.
    """
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hub"))
    cfg = load_config()
    entry = ModelEntry(id="ghost", hf_repo="org/Ghost", model_format="mlx")
    sup = Supervisor(_mlx_backend(cfg), entry, cfg, _profile(), paths, FakeStore())
    with pytest.raises(WeightsMissingError) as exc:
        sup._preflight()
    assert "org/Ghost" in " ".join(exc.value.remediation)


def test_preflight_refuses_to_fetch_onto_a_full_disk(paths, monkeypatch, tmp_path):
    """--force allows the fetch, but the disk floor still gates it."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hub"))
    _free_gib(monkeypatch, 1.0)  # below disk.min_free_gib = 10
    cfg = load_config()
    entry = ModelEntry(id="ghost", hf_repo="org/Ghost", size_bytes=8 * 2**30)
    sup = Supervisor(_mlx_backend(cfg), entry, cfg, _profile(), paths, FakeStore(), force=True)
    with pytest.raises(InsufficientDiskError):
        sup._preflight()


def test_preflight_allows_forced_fetch_when_disk_is_roomy(paths, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hub"))
    _free_gib(monkeypatch, 100.0)
    cfg = load_config()
    entry = ModelEntry(id="ghost", hf_repo="org/Ghost")
    sup = Supervisor(_mlx_backend(cfg), entry, cfg, _profile(), paths, FakeStore(), force=True)
    assert sup._preflight() > 0  # reached port selection: gates passed


def test_low_disk_does_not_block_a_local_model(paths, monkeypatch):
    """Nothing is written when the weights are on disk, so the floor is advisory."""
    _free_gib(monkeypatch, 1.0)
    cfg = load_config()
    store = paths.model_store_dir / "local-model"
    store.mkdir(parents=True, exist_ok=True)
    (store / "model.safetensors").write_bytes(b"w" * 64)
    entry = ModelEntry(id="local-model", path=str(store), model_format="mlx")
    sup = Supervisor(_mlx_backend(cfg), entry, cfg, _profile(), paths, FakeStore())
    assert sup._preflight() > 0


def test_runtime_owning_its_weights_is_never_asked_for_them(paths, monkeypatch, tmp_path):
    """A router child declares requires_local_weights=false; no gate, no fetch."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-hub"))
    cfg = load_config()
    rt = RuntimeDef(
        name="router",
        detect=DetectSpec(binary="/bin/true"),
        server=ServerSpec(binary="/bin/true", default_port=8093),
        requires_local_weights=False,
    )
    cfg.runtimes["router"] = rt
    entry = ModelEntry(id="router", backend="router", hf_repo="org/Tiers")
    sup = Supervisor(Backend(rt, cfg), entry, cfg, _profile(), paths, FakeStore())
    assert sup._preflight() > 0


# --- generation-verified readiness ----------------------------------------------
class _LazyBackend(Backend):
    """Stands in for mlx_lm/mlx_vlm: HTTP up before the model is resident."""

    lazy_loads_model = True

    def __init__(self, rt, cfg, warm):
        super().__init__(rt, cfg)
        self._warm = warm
        self.warmed = 0

    def warmup(self, entry, host, port, timeout_s):
        self.warmed += 1
        return self._warm


def _lazy_sup(paths, cfg, warm, *, verify=True):
    rt = RuntimeDef(
        name="lazy",
        detect=DetectSpec(binary="/bin/true"),
        server=ServerSpec(binary="/bin/true", default_port=8094, ready_timeout_s=1.0),
    )
    cfg.runtimes["lazy"] = rt
    cfg.supervisor.max_restarts = 1
    cfg.supervisor.backoff_base_s = 0.01
    cfg.supervisor.backoff_max_s = 0.01
    cfg.supervisor.verify_generation = verify
    backend = _LazyBackend(rt, cfg, warm)
    sup = Supervisor(backend, ModelEntry(id="m"), cfg, _profile(), paths, FakeStore())
    monkeypatch_noop(sup)
    return sup, backend


def monkeypatch_noop(sup):
    """Do not spawn or signal anything real; the gates are what is under test."""
    sup._spawn = lambda cmd, env: None  # noqa: SLF001
    sup._terminate = lambda: None  # noqa: SLF001
    sup._is_healthy = lambda url: True  # noqa: SLF001


def test_warmup_failure_is_a_launch_failure_not_a_silent_zombie(paths):
    """A server that answers /health but produces no tokens must not be 'ready'.

    This is the 2026-09-18 failure: the model load crashed inside the runtime,
    the process stayed up, and every request hung while spark reported success.
    """
    cfg = load_config()
    sup, backend = _lazy_sup(paths, cfg, warm=(False, "ConnectionReset: boom"))
    sup._stop.clear()
    with pytest.raises(RecoveryExhaustedError) as exc:
        sup.run()
    assert backend.warmed >= 1
    assert "produced no tokens" in str(exc.value) or "failed to start" in str(exc.value)


def test_warmup_success_reports_ready(paths, monkeypatch):
    cfg = load_config()
    sup, backend = _lazy_sup(paths, cfg, warm=(True, "generated"))
    monkeypatch.setattr(sup, "_monitor", lambda url: "stop_requested")
    seen = []
    result = sup.run(on_ready=lambda info: seen.append(info))
    assert backend.warmed == 1
    assert len(seen) == 1
    assert result.endpoint.endswith("/v1")


def test_warmup_skipped_when_verification_disabled(paths, monkeypatch):
    cfg = load_config()
    sup, backend = _lazy_sup(paths, cfg, warm=(False, "would fail"), verify=False)
    monkeypatch.setattr(sup, "_monitor", lambda url: "stop_requested")
    sup.run()  # no exception: the check was switched off
    assert backend.warmed == 0


def test_eager_backend_needs_no_warmup(paths, monkeypatch):
    """llama.cpp/omlx load before serving; their health probe is already honest."""
    cfg = load_config()
    rt = RuntimeDef(
        name="eager",
        detect=DetectSpec(binary="/bin/true"),
        server=ServerSpec(binary="/bin/true", default_port=8095, ready_timeout_s=1.0),
    )
    cfg.runtimes["eager"] = rt
    backend = Backend(rt, cfg)  # lazy_loads_model is False by default
    sup = Supervisor(backend, ModelEntry(id="m"), cfg, _profile(), paths, FakeStore())
    monkeypatch_noop(sup)
    monkeypatch.setattr(sup, "_monitor", lambda url: "stop_requested")
    sup.run()  # no warmup call exists to fail
