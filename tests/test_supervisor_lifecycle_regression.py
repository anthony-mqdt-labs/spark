"""Regression tests for supervisor lifecycle contracts (D16/D18)."""

from __future__ import annotations

import signal
import subprocess

import pytest

from spark.catalog import read_instances
from spark.config.loader import load_config
from spark.config.schema import DetectSpec, ModelEntry, RuntimeDef, ServerSpec
from spark.probe.host import HostProfile, RuntimeStatus
from spark.runner import Supervisor
from spark.runtimes.base import Backend
from helpers import FakeStore


def _profile():
    return HostProfile(
        os="Darwin", arch="arm64", chip="Apple M2",
        total_memory_bytes=16 * 2**30, cpu_count=8, probed_at=0.0,
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
        Backend(rt, cfg), ModelEntry(id="daemon-model"), cfg,
        _profile(), paths, FakeStore(),
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
    monkeypatch.setattr(
        signal, "signal", lambda sig, handler: captured.setdefault(sig, handler)
    )
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
    monkeypatch.setattr(
        "spark.runner.supervisor._rss_bytes", lambda pid: calls.append(pid) or 1024
    )
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
            binary="true", attach_if_running=True, default_port=19022,
            secret_env={"HF_TOKEN": "hf_token"},
        ),
    )
    cfg.runtimes["daemon"] = rt
    sup = Supervisor(
        Backend(rt, cfg), ModelEntry(id="m"), cfg, _profile(), paths, _FailingStore()
    )
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
