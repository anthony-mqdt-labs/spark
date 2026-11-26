"""Regression tests: the per-server child log must not become a secret sink."""

from __future__ import annotations

import os
import stat

import pytest

from spark.config.loader import load_config
from spark.config.schema import DetectSpec, ModelEntry, RuntimeDef, ServerSpec
from spark.probe.host import HostProfile, RuntimeStatus
from spark.redaction import REDACTED, register_secret_value, reset_secret_values
from spark.runner import Supervisor
from spark.runtimes.base import Backend
from helpers import FakeStore


class _WritingPopen:
    """Fake child that writes secret-bearing lines straight to its stdout fd."""

    def __init__(self, cmd, stdout=None, stderr=None, env=None, **kwargs):
        self.pid = 999_999
        self.returncode = None
        if stdout is not None:
            os.write(stdout, b"error: HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz\n")
            os.write(stdout, b"token value: secret-value-1234\n")

    def poll(self):
        return None  # alive; the patched monitor decides when to stop

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def terminate(self):
        self.returncode = 0

    def kill(self):
        self.returncode = -9


def _profile():
    return HostProfile(
        os="Darwin", arch="arm64", chip="Apple M2",
        total_memory_bytes=16 * 2**30, cpu_count=8, probed_at=0.0,
        runtimes={"boom": RuntimeStatus(name="boom", available=True)},
    )


@pytest.fixture
def sup(paths, monkeypatch):
    reset_secret_values()
    register_secret_value("secret-value-1234")
    cfg = load_config()
    rt = RuntimeDef(
        name="boom",
        detect=DetectSpec(binary="true"),
        server=ServerSpec(
            binary="true", args_template=[], health_path="/health",
            default_port=19011, ready_timeout_s=1.0,
        ),
    )
    cfg.runtimes["boom"] = rt
    cfg.supervisor.verify_generation = False
    s = Supervisor(
        Backend(rt, cfg), ModelEntry(id="log-model"), cfg,
        _profile(), paths, FakeStore(),
    )
    monkeypatch.setattr(s, "_is_healthy", lambda url: True)
    monkeypatch.setattr(s, "_monitor", lambda url: "stop_requested")
    from spark.runner import supervisor as mod

    monkeypatch.setattr(mod.subprocess, "Popen", _WritingPopen)
    yield s
    reset_secret_values()


def test_child_log_file_is_private(sup):
    sup.run()
    mode = stat.S_IMODE(sup.log_path.stat().st_mode)
    assert mode == 0o600


def test_child_output_is_scrubbed_before_it_reaches_disk(sup):
    sup.run()
    content = sup.log_path.read_text()
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in content
    assert "secret-value-1234" not in content
    assert REDACTED in content


def test_log_tail_scrubs_legacy_unscrubbed_content(sup):
    # A log written by an older spark (or by a child that bypassed the pump)
    # must still be safe when rendered to the operator's terminal.
    sup._log_path.parent.mkdir(parents=True, exist_ok=True)
    sup._log_path.write_text("legacy: secret-value-1234\n")
    assert "secret-value-1234" not in sup.log_tail()
    assert REDACTED in sup.log_tail()
