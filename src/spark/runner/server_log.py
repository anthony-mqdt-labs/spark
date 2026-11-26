"""The supervised server's stdout/stderr log.

A child writes to a file *descriptor*, so wrapping a Python file object in a
redactor would not intercept a single byte. This module owns the read end of
one pipe per spawn and scrubs complete lines before they reach disk. One
instance serves several spawns (a restart does not lose the log or its pump).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TextIO

from ..redaction import scrub_string

#: A child can emit newline-free output forever (``\r`` progress bars). Cap the
#: pending buffer so such a child cannot grow it without bound: once it fills,
#: flush what we have and keep reading. One extra read (64 KiB) is the most that
#: can accumulate past the cap between checks.
_MAX_PENDING_BYTES = 1 << 20


class ServerLog:
    """Owner of one server's log file and its scrubbing pipe pumps."""

    def __init__(self, path: Path, roll_bytes: int) -> None:
        self._path = path
        self._roll_bytes = roll_bytes
        self._fh: TextIO | None = None
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    @property
    def path(self) -> Path:
        return self._path

    @property
    def fh(self) -> TextIO | None:
        return self._fh

    @property
    def threads(self) -> tuple[threading.Thread, ...]:
        return tuple(self._threads)

    def _roll_if_needed(self) -> None:
        """Roll once past the configured size; keep one previous copy.

        No lock is held here: a pump thread may still be finishing its last
        write, and joining while holding ``_lock`` would deadlock with it.
        """
        try:
            size = self._path.stat().st_size
        except OSError:
            size = 0
        if size <= self._roll_bytes:
            return
        self.join()
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.flush()
                except (OSError, ValueError):
                    pass
                try:
                    self._fh.close()
                except (OSError, ValueError):
                    pass
                self._fh = None
            try:
                self._path.replace(Path(str(self._path) + ".1"))
            except OSError:
                pass  # absent or unrollable: append regardless

    def open(self) -> TextIO:
        """Open (or reopen after a roll) in append mode, owner-only."""
        self._roll_if_needed()
        with self._lock:
            if self._fh is None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    os.fchmod(fd, 0o600)  # tighten a log created by older spark
                    self._fh = os.fdopen(fd, "a", buffering=1)
                except BaseException:
                    os.close(fd)
                    raise
            return self._fh

    def write(self, text: str) -> None:
        """Write an already-formatted line, scrubbing it first."""
        fh = self.open()
        with self._lock:
            try:
                fh.write(scrub_string(text))
                fh.flush()
            except (OSError, ValueError):
                pass

    def start_pipe(self) -> int:
        """Return the write fd to hand to ``Popen`` (parent closes it thereafter)."""
        self.open()
        read_fd, write_fd = os.pipe()
        thread = threading.Thread(
            target=self._drain,
            args=(read_fd,),
            daemon=True,
            name="spark-server-log",
        )
        thread.start()
        self._threads.append(thread)
        return write_fd

    def _drain(self, read_fd: int) -> None:
        pending = b""
        try:
            while True:
                chunk = os.read(read_fd, 65536)
                if not chunk:
                    break
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    self._write(line + b"\n")
                if len(pending) >= _MAX_PENDING_BYTES:
                    # No newline for a long time: flush safely and start fresh.
                    # Never drop the file or kill the pump over a chatty child.
                    self._write(pending)
                    pending = b""
            if pending:
                self._write(pending)
        except OSError:
            pass  # child/parent closed the pipe mid-read; nothing left to log
        finally:
            try:
                os.close(read_fd)
            except OSError:
                pass

    def _write(self, raw: bytes) -> None:
        text = scrub_string(raw.decode("utf-8", errors="replace"))
        with self._lock:
            fh = self._fh
            if fh is None:
                return
            try:
                fh.write(text)
                fh.flush()
            except (OSError, ValueError):
                pass  # logging must never kill the supervised child

    def tail(self, n: int = 30) -> str:
        """Last ``n`` lines, scrubbed even if an older spark wrote them."""
        try:
            text = self._path.read_text(errors="replace")
        except OSError:
            return ""
        return scrub_string("\n".join(text.splitlines()[-n:]))

    def join(self, timeout: float = 1.0) -> None:
        for thread in tuple(self._threads):
            thread.join(timeout)

    def close(self) -> None:
        """Drain every pipe, then close the file. Safe to call repeatedly."""
        self.join()
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.flush()
                except (OSError, ValueError):
                    pass
                try:
                    self._fh.close()
                except (OSError, ValueError):
                    pass
                self._fh = None
