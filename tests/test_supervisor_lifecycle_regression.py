"""Regression tests for supervisor lifecycle contracts (D16/D18)."""

from __future__ import annotations

import os
import signal
import subprocess

import pytest

from spark.catalog import catalog_path, read_instances
from spark.config.loader import load_config
from spark.config.schema import DetectSpec, ModelEntry, RuntimeDef, ServerSpec
from spark.probe.host import HostProfile, RuntimeStatus
from spark.runner import Supervisor
from spark.runtimes.base import Backend
from helpers import FakeStore


def _profile():
    return HostProfile(
        os="Darwin",
        arch="arm64",
        chip="Apple M2",
        total_memory_bytes=16 * 2**30,
        cpu_count=8,
        probed_at=0.0,
        runtimes={"daemon": RuntimeStatus(name="daemon", available=True)},
    )


def _attached_supervisor(paths):
    cfg = load_config()
    rt = RuntimeDef(
        name="daemon",
        detect=DetectSpec(binary="true"),
        server=ServerSpec(binary="true", attach_if_running=True, default_port=19021),
    )
    cfg.runtimes["daemon"] = rt
    return Supervisor(
        Backend(rt, cfg),
        ModelEntry(id="daemon-model"),
        cfg,
        _profile(),
        paths,
        FakeStore(),
    )


class _NoLog:
    """A logger that fails loudly if a signal handler tries to write to it."""

    def info(self, *args, **kwargs):
        raise AssertionError("signal handler must not log (deadlock risk)")

    def warn(self, *args, **kwargs):
        raise AssertionError("signal handler must not log (deadlock risk)")

    def error(self, *args, **kwargs):
        raise AssertionError("signal handler must not log (deadlock risk)")


def test_signal_handler_only_sets_stop(paths, monkeypatch):
    """Handlers run on the main thread: logging inside one can re-enter the
    telemetry lock and deadlock if the main thread was mid-write."""
    sup = _attached_supervisor(paths)
    sup.log = _NoLog()

    captured: dict[int, object] = {}
    monkeypatch.setattr(signal, "signal", lambda sig, handler: captured.setdefault(sig, handler))
    sup._install_signals()
    captured[signal.SIGINT](signal.SIGINT, None)
    assert sup._stop.is_set()


def test_stop_signal_is_logged_from_the_monitor_loop(paths, monkeypatch):
    """The event still needs to exist — just not from inside the handler."""
    sup = _attached_supervisor(paths)
    events: list[str] = []

    class _Recorder:
        def info(self, component, event, **kwargs):
            events.append(event)

        def warn(self, component, event, **kwargs):
            events.append(event)

        def context_window(self):
            return []

    sup.log = _Recorder()
    sup._stop.set()
    assert sup._monitor("http://127.0.0.1:19021/health") == "stop_requested"
    assert "signal_received" in events


def test_polling_uses_the_configured_interval(paths, monkeypatch):
    sup = _attached_supervisor(paths)
    sup.config.supervisor.health_interval_s = 0.02
    sleeps: list[float] = []
    monkeypatch.setattr("spark.runner.supervisor.time.sleep", sleeps.append)
    healthy = iter([False, True])
    monkeypatch.setattr(sup, "_is_healthy", lambda url: next(healthy))
    sup._wait_ready("http://127.0.0.1:19021/health", 1.0)
    assert sleeps == [0.02]


def test_attached_monitor_uses_the_configured_interval(paths, monkeypatch):
    sup = _attached_supervisor(paths)
    sup.config.supervisor.health_interval_s = 0.03
    waits: list[float] = []
    monkeypatch.setattr(sup._stop, "wait", lambda value: waits.append(value) or True)
    assert sup._monitor_attached("http://127.0.0.1:19021/health") == "stop_requested"
    assert waits == [0.03]


def test_rss_sampling_is_throttled(paths, monkeypatch):
    sup = _attached_supervisor(paths)

    class _Proc:
        def poll(self):
            return None

        pid = 1

    sup._proc = _Proc()
    calls: list[int] = []
    monkeypatch.setattr("spark.runner.supervisor._rss_bytes", lambda pid: calls.append(pid) or 1024)
    ticks = iter(range(12))

    def _wait(_interval):
        try:
            next(ticks)
            return False
        except StopIteration:
            sup._stop.set()
            return True

    monkeypatch.setattr(sup._stop, "wait", _wait)
    sup._monitor("http://127.0.0.1:19021/health")
    # One sample every six ticks, not a fork+exec of `ps` on every tick.
    assert len(calls) == 2


def test_terminate_escalates_to_sigkill(paths):
    sup = _attached_supervisor(paths)
    killed: list[bool] = []

    class _StubbornProc:
        pid = 4242

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("child", timeout or 0)

        def kill(self):
            killed.append(True)

    sup._proc = _StubbornProc()
    sup._terminate()
    assert killed == [True]


def test_secret_resolve_failure_logs_var_name_not_value(paths, monkeypatch):
    class _FailingStore(FakeStore):
        def exists(self, name):
            return True  # the vault says it is there, but reading it fails

        def get(self, name):
            raise RuntimeError("hf_token-value-must-not-leak")

    cfg = load_config()
    rt = RuntimeDef(
        name="daemon",
        detect=DetectSpec(binary="true"),
        server=ServerSpec(
            binary="true",
            attach_if_running=True,
            default_port=19022,
            secret_env={"HF_TOKEN": "hf_token"},
        ),
    )
    cfg.runtimes["daemon"] = rt
    sup = Supervisor(Backend(rt, cfg), ModelEntry(id="m"), cfg, _profile(), paths, _FailingStore())
    events: list[dict] = []

    class _Recorder:
        def warn(self, component, event, **payload):
            events.append({"event": event, **payload})

    monkeypatch.setattr("spark.secrets.store.get_telemetry", lambda: _Recorder())

    env = sup._child_env()
    assert "HF_TOKEN" not in env
    assert [e["event"] for e in events] == ["secret_resolve_failed"]
    assert events[0]["env_var"] == "HF_TOKEN"
    assert "hf_token-value-must-not-leak" not in repr(events)


def test_attached_daemon_publishes_and_removes_instance_record(paths, monkeypatch):
    """README §9.8 promises one record per live server, including attached ones."""
    sup = _attached_supervisor(paths)
    monkeypatch.setattr(sup, "_is_healthy", lambda url: True)
    seen: dict[str, list[dict]] = {}

    def _fake_monitor(url):
        seen["during"] = read_instances(paths)
        return "stop_requested"

    monkeypatch.setattr(sup, "_monitor_attached", _fake_monitor)

    sup.run()

    assert any(r.get("model_alias") == "daemon-model" for r in seen["during"])
    assert not [r for r in read_instances(paths) if r.get("model_alias") == "daemon-model"]


class _Events:
    def __init__(self):
        self.events: list[str] = []

    def info(self, component, event, **kwargs):
        self.events.append(event)

    def warn(self, component, event, **kwargs):
        self.events.append(event)

    def error(self, component, event, **kwargs):
        self.events.append(event)

    def context_window(self):
        return []


def _spawn_supervisor(paths, cfg=None):
    cfg = cfg or load_config()
    rt = RuntimeDef(
        name="boom",
        detect=DetectSpec(binary="true"),
        server=ServerSpec(
            binary="true",
            args_template=[],
            health_path="/health",
            default_port=19031,
            ready_timeout_s=1.0,
        ),
    )
    cfg.runtimes["boom"] = rt
    return Supervisor(
        Backend(rt, cfg),
        ModelEntry(id="stop-model"),
        cfg,
        _profile(),
        paths,
        FakeStore(),
    )


def test_stop_signal_during_startup_still_logs_signal_received(paths, monkeypatch):
    """The `break` after `_wait_ready` used to skip the required event."""
    sup = _spawn_supervisor(paths)
    events = _Events()
    sup.log = events
    monkeypatch.setattr(sup, "_spawn", lambda cmd, env: None)
    monkeypatch.setattr(sup, "_wait_ready", lambda url, timeout: sup._stop.set())
    monkeypatch.setattr(sup, "_monitor", lambda url: "stop_requested")
    sup.run()
    assert "signal_received" in events.events


def test_missing_process_is_an_explicit_runtime_error(paths):
    """An invariant assert disappears under `python -O`; fail explicitly."""
    sup = _attached_supervisor(paths)
    sup._proc = None
    with pytest.raises(RuntimeError, match="process"):
        sup._monitor("http://127.0.0.1:19021/health")


def test_missing_log_pump_is_an_explicit_runtime_error(paths, monkeypatch):
    sup = _attached_supervisor(paths)
    monkeypatch.setattr(sup, "_open_server_log", lambda: None)
    with pytest.raises(RuntimeError, match="log pump"):
        sup._spawn(["true"], {})


def test_signal_handlers_are_restored_to_the_originals(paths):
    sup = _attached_supervisor(paths)
    sigs = (signal.SIGINT, signal.SIGTERM)
    originals = {sig: signal.getsignal(sig) for sig in sigs}

    def custom(_signum, _frame):
        pass

    for sig in sigs:
        signal.signal(sig, custom)
    try:
        sup._install_signals()
        assert all(signal.getsignal(sig) is not custom for sig in sigs)
        sup._restore_signals()
        assert all(signal.getsignal(sig) is custom for sig in sigs)
        assert sup._prev_signals == {}
    finally:
        for sig, handler in originals.items():
            signal.signal(sig, handler)


def test_configured_heartbeat_reaches_the_instance_publisher(paths, monkeypatch):
    sup = _attached_supervisor(paths)
    sup.config.supervisor.instance_heartbeat_s = 7.5
    monkeypatch.setattr(sup, "_is_healthy", lambda url: True)
    sup._stop.set()  # attached monitor returns immediately

    captured: dict = {}

    class _Publisher:
        def __init__(self, _paths, **kwargs):
            captured.update(kwargs)

        def publish(self):
            from pathlib import Path

            return Path("instance.json")

        def maybe_heartbeat(self):
            pass

        def remove(self):
            pass

    monkeypatch.setattr("spark.runner.supervisor.InstancePublisher", _Publisher)
    sup.run()
    assert captured["heartbeat_s"] == 7.5


def test_health_probe_timeout_comes_from_config(paths, monkeypatch):
    sup = _attached_supervisor(paths)
    sup.config.supervisor.health_probe_timeout_s = 0.42
    seen: dict[str, float] = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def urlopen(_req, timeout=None):
        seen["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("spark.runner.supervisor.urllib.request.urlopen", urlopen)
    assert sup._is_healthy("http://127.0.0.1/health") is True
    assert seen["timeout"] == 0.42


def test_rss_sampling_interval_comes_from_config(paths, monkeypatch):
    sup = _attached_supervisor(paths)
    sup.config.supervisor.rss_sample_ticks = 3

    class _Proc:
        pid = 1

        def poll(self):
            return None

    sup._proc = _Proc()
    calls: list[int] = []
    monkeypatch.setattr("spark.runner.supervisor._rss_bytes", lambda pid: calls.append(pid) or 1024)
    ticks = iter(range(10))

    def _wait(_interval):
        try:
            next(ticks)
            return False
        except StopIteration:
            sup._stop.set()
            return True

    monkeypatch.setattr(sup._stop, "wait", _wait)
    sup._monitor("http://127.0.0.1:19021/health")
    assert len(calls) == 3  # ticks 3, 6, 9 — not the old hard-coded every 6th


def test_unexpected_exception_cleans_up_the_whole_lifecycle(paths, monkeypatch):
    """An unanticipated exception in attached mode must still restore handlers,
    drop the instance record, join the pump, close the log and refresh catalog."""
    sup = _attached_supervisor(paths)
    monkeypatch.setattr(sup, "_is_healthy", lambda url: True)
    write_fd = sup._server_log.start_pipe()
    os.close(write_fd)

    sigs = (signal.SIGINT, signal.SIGTERM)
    originals = {sig: signal.getsignal(sig) for sig in sigs}

    def boom(_url):
        raise RuntimeError("unanticipated")

    monkeypatch.setattr(sup, "_monitor_attached", boom)
    catalog = catalog_path(paths)
    before = catalog.stat().st_mtime_ns if catalog.exists() else 0
    try:
        with pytest.raises(RuntimeError, match="unanticipated"):
            sup.run()
        assert all(signal.getsignal(sig) is originals[sig] for sig in sigs)
        assert not [r for r in read_instances(paths) if r.get("model_alias") == "daemon-model"]
        assert all(not thread.is_alive() for thread in sup._server_log.threads)
        assert sup._log_fh is None
        assert sup._log_pump is None
        assert catalog.exists() and catalog.stat().st_mtime_ns >= before
    finally:
        for sig, handler in originals.items():
            signal.signal(sig, handler)
