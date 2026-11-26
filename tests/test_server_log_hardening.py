"""Regression tests: the per-server child log must not become a secret sink."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from spark.config.loader import load_config
from spark.config.schema import DetectSpec, ModelEntry, RuntimeDef, ServerSpec
from spark.probe.host import HostProfile, RuntimeStatus
from spark.redaction import REDACTED, register_secret_value, reset_secret_values
from spark.runner import Supervisor
from spark.runner import server_log as server_log_mod
from spark.runner.server_log import ServerLog
from spark.runtimes.base import Backend
from helpers import FakeStore


def _write_all(fd, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view[:65536]) :]


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
        os="Darwin",
        arch="arm64",
        chip="Apple M2",
        total_memory_bytes=16 * 2**30,
        cpu_count=8,
        probed_at=0.0,
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
            binary="true",
            args_template=[],
            health_path="/health",
            default_port=19011,
            ready_timeout_s=1.0,
        ),
    )
    cfg.runtimes["boom"] = rt
    cfg.supervisor.verify_generation = False
    s = Supervisor(
        Backend(rt, cfg),
        ModelEntry(id="log-model"),
        cfg,
        _profile(),
        paths,
        FakeStore(),
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


def test_two_spawns_share_one_scrubbing_pump(tmp_path):
    """Restarts reuse the log: each spawn gets its own pipe, all lines land."""
    reset_secret_values()
    register_secret_value("secret-value-1234")
    path = tmp_path / "servers" / "m.log"
    log = ServerLog(path, roll_bytes=1 << 20)
    write_fds = [log.start_pipe(), log.start_pipe()]
    os.write(write_fds[0], b"first secret-value-1234\n")
    os.close(write_fds[0])
    os.write(write_fds[1], b"second HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz\n")
    os.close(write_fds[1])
    log.join()
    text = path.read_text()
    assert "first" in text and "second" in text
    assert "secret-value-1234" not in text
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in text
    assert REDACTED in text
    assert len(log.threads) == 2
    assert all(not thread.is_alive() for thread in log.threads)
    log.close()
    reset_secret_values()


def test_close_joins_pump_and_closes_file(tmp_path):
    """An early exception may call close() without an explicit join()."""
    path = tmp_path / "servers" / "early.log"
    log = ServerLog(path, roll_bytes=1 << 20)
    write_fd = log.start_pipe()
    os.write(write_fd, b"only line\n")
    os.close(write_fd)
    fh = log.fh
    log.close()
    assert all(not thread.is_alive() for thread in log.threads)
    assert fh.closed
    assert "only line" in path.read_text()


def test_log_rolls_at_the_configured_size(tmp_path):
    path = tmp_path / "servers" / "roll.log"
    log = ServerLog(path, roll_bytes=16)
    log.write("x" * 100)
    assert path.stat().st_size > 16
    write_fd = log.start_pipe()  # a new spawn triggers the size check
    os.close(write_fd)
    assert (tmp_path / "servers" / "roll.log.1").exists()
    assert path.stat().st_size < 100
    log.close()


def test_early_exception_joins_the_supervisor_pump(sup, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("early failure")

    monkeypatch.setattr(sup, "_wait_ready", boom)
    with pytest.raises(RuntimeError, match="early failure"):
        sup.run()
    assert sup._log_fh is None
    assert sup._log_pump is None
    assert all(not thread.is_alive() for thread in sup._server_log.threads)


# --- bounded newline-free output ------------------------------------------------
def test_newline_free_output_is_flushed_in_bounded_pieces(tmp_path, monkeypatch):
    """A child emitting a newline-free progress bar must not grow the buffer forever."""
    monkeypatch.setattr(server_log_mod, "_MAX_PENDING_BYTES", 32)
    path = tmp_path / "progress.log"
    log = ServerLog(path, roll_bytes=1 << 20)
    log.open()
    writes = []
    real_write = log._write

    def recording(raw):
        writes.append(raw)
        real_write(raw)

    monkeypatch.setattr(log, "_write", recording)
    chunks = iter([b"A" * 40, b"B" * 40, b"tail\n", b""])
    monkeypatch.setattr(server_log_mod.os, "read", lambda _fd, _n: next(chunks))
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    log._drain(read_fd)
    assert writes == [b"A" * 40, b"B" * 40, b"tail\n"]
    assert path.read_bytes() == b"A" * 40 + b"B" * 40 + b"tail\n"
    log.close()


def test_large_newline_free_payload_survives_the_default_cap(tmp_path):
    """Bytes with no newline at all are still logged intact at EOF."""
    path = tmp_path / "big-progress.log"
    log = ServerLog(path, roll_bytes=10 << 20)
    payload = b"\r".join(b"step-%04d" % i for i in range(2000))
    write_fd = log.start_pipe()
    _write_all(write_fd, payload)
    os.close(write_fd)
    log.join(timeout=5.0)
    log.close()
    assert path.read_bytes() == payload


# --- failure tolerance (logging must never kill the supervised child) -----------
def test_roll_survives_a_file_that_cannot_flush_or_close(tmp_path):
    path = tmp_path / "servers" / "broken.log"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x" * 100)

    class FailingFH:
        def flush(self):
            raise OSError("flush failed")

        def close(self):
            raise ValueError("close failed")

    log = ServerLog(path, roll_bytes=16)
    log._fh = FailingFH()
    fh = log.open()  # must roll anyway and reopen a usable file
    assert fh is not None and fh is log.fh
    assert (tmp_path / "servers" / "broken.log.1").read_bytes() == b"x" * 100
    assert path.read_bytes() == b""
    log.close()


def test_failed_roll_rename_still_appends(tmp_path, monkeypatch):
    path = tmp_path / "servers" / "roll.log"
    path.parent.mkdir(parents=True)
    path.write_text("keep me")

    def boom_replace(self, target):
        raise OSError("rename failed")

    monkeypatch.setattr(Path, "replace", boom_replace)
    log = ServerLog(path, roll_bytes=1)
    log.write("more\n")
    assert path.read_text() == "keep memore\n"
    log.close()


def test_open_closes_the_fd_when_fdopen_fails(tmp_path, monkeypatch):
    log = ServerLog(tmp_path / "fd.log", roll_bytes=1 << 20)
    real_close = os.close
    closed = []

    def fake_close(fd):
        closed.append(fd)
        real_close(fd)

    def boom(*_a, **_k):
        raise RuntimeError("fdopen failed")

    monkeypatch.setattr(server_log_mod.os, "fdopen", boom)
    monkeypatch.setattr(server_log_mod.os, "close", fake_close)
    with pytest.raises(RuntimeError, match="fdopen failed"):
        log.open()
    assert closed  # the raw fd was not leaked
    assert log.fh is None
    log.close()


def test_write_and_pump_ignore_file_errors(tmp_path):
    class BrokenFH:
        def write(self, text):
            raise OSError("disk full")

        def flush(self):
            raise ValueError("already closed")

    log = ServerLog(tmp_path / "err.log", roll_bytes=1 << 20)
    log._fh = BrokenFH()
    log.write("line\n")
    log._write(b"raw\n")


def test_pump_write_before_open_is_a_noop(tmp_path):
    log = ServerLog(tmp_path / "none.log", roll_bytes=1 << 20)
    log._write(b"ignored")
    assert log.fh is None
    assert not (tmp_path / "none.log").exists()


def test_close_swallows_flush_and_close_errors(tmp_path):
    class BrokenFH:
        def flush(self):
            raise ValueError("closed")

        def close(self):
            raise OSError("close failed")

    log = ServerLog(tmp_path / "close.log", roll_bytes=1 << 20)
    log._fh = BrokenFH()
    log.close()
    assert log.fh is None


def test_tail_returns_empty_when_unreadable(tmp_path):
    log = ServerLog(tmp_path, roll_bytes=1 << 20)  # a directory, not a file
    assert log.tail() == ""


def test_drain_swallows_read_and_close_errors(tmp_path, monkeypatch):
    log = ServerLog(tmp_path / "pipe.log", roll_bytes=1 << 20)
    log.open()
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    real_close = os.close
    closed = []

    def boom_read(_fd, _n):
        raise OSError("pipe closed")

    def fake_close(fd):
        closed.append(fd)
        raise OSError("close failed")

    with monkeypatch.context() as m:
        m.setattr(server_log_mod.os, "read", boom_read)
        m.setattr(server_log_mod.os, "close", fake_close)
        log._drain(read_fd)  # neither error may escape
    assert closed == [read_fd]
    real_close(read_fd)
    log.close()
