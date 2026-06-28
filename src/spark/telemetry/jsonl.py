"""JSONL structured logger with rotation, retention, and secret redaction.

Usage::

    from spark.telemetry import init_telemetry, get_telemetry
    init_telemetry(log_dir, session_id)
    log = get_telemetry()
    log.info("supervisor", "process_start", exe="mlx_lm.server", pid=123)

Design notes:
- One file per component per UTC day: ``<log_dir>/<component>/<YYYY-MM-DD>.jsonl``.
- Retention prunes that component's old files on first write each run.
- Redaction is defense-in-depth: secrets should *never* reach a log call, but if
  one slips through (by literal value or by a sensitive key name) it is masked.
"""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque

REQUIRED_LEVELS = ("debug", "info", "warn", "error", "critical")
_REDACTED = "***REDACTED***"

# Keys whose values are always masked regardless of content.
_SENSITIVE_KEY_HINTS = (
    "token",
    "secret",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "authorization",
    "auth_header",
    "bearer",
    "credential",
)

# Registry of literal secret values to scrub from any logged string. Populated by
# the secrets layer at the moment a value is read, so even an accidental log of a
# wrapping object gets cleaned.
_secret_values: set[str] = set()
_secret_lock = threading.Lock()


def register_secret_value(value: str | None) -> None:
    """Register a literal secret so the logger masks it everywhere. No-op for
    short/empty values (avoids masking innocuous substrings)."""
    if value and len(value) >= 4:
        with _secret_lock:
            _secret_values.add(value)


def _scrub_string(s: str) -> str:
    if not _secret_values:
        return s
    with _secret_lock:
        values = tuple(_secret_values)
    for v in values:
        if v in s:
            s = s.replace(v, _REDACTED)
    return s


def _is_sensitive_key(key: str) -> bool:
    k = key.lower()
    return any(hint in k for hint in _SENSITIVE_KEY_HINTS)


def _redact(obj: Any, *, key: str | None = None) -> Any:
    """Recursively redact: mask sensitive keys; scrub literal secrets from strings."""
    if key is not None and _is_sensitive_key(key):
        return _REDACTED
    if isinstance(obj, dict):
        return {k: _redact(v, key=str(k)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_redact(v) for v in obj]
    if isinstance(obj, str):
        return _scrub_string(obj)
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    # Fallback: stringify unknown types and scrub.
    return _scrub_string(repr(obj))


class Telemetry:
    """Per-session structured logger.

    Keeps a small ring buffer of recent events so a crash handler can attach a
    ``context_window`` for post-hoc correlation (per the logging spec)."""

    def __init__(
        self,
        log_dir: Path,
        session_id: str,
        *,
        retention_days: int = 7,
        context_window: int = 50,
        echo_errors: bool = False,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.session_id = session_id
        self.retention_days = retention_days
        self.pid = os.getpid()
        self._ring: Deque[dict[str, Any]] = deque(maxlen=context_window)
        self._pruned: set[str] = set()
        self._lock = threading.Lock()
        self._echo_errors = echo_errors

    # -- public level helpers ---------------------------------------------------
    def debug(self, component: str, event: str, **payload: Any) -> None:
        self.emit("debug", component, event, **payload)

    def info(self, component: str, event: str, **payload: Any) -> None:
        self.emit("info", component, event, **payload)

    def warn(self, component: str, event: str, **payload: Any) -> None:
        self.emit("warn", component, event, **payload)

    def error(self, component: str, event: str, **payload: Any) -> None:
        self.emit("error", component, event, **payload)

    def critical(self, component: str, event: str, **payload: Any) -> None:
        self.emit("critical", component, event, **payload)

    # -- core -------------------------------------------------------------------
    def emit(self, level: str, component: str, event: str, **payload: Any) -> None:
        if level not in REQUIRED_LEVELS:
            level = "info"
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": level,
            "component": component,
            "event": event,
            "session_id": self.session_id,
            "pid": self.pid,
        }
        if payload:
            record.update(_redact(payload))

        self._ring.append(record)
        line = json.dumps(record, ensure_ascii=False, default=str)
        self._write(component, line)
        if self._echo_errors and level in ("error", "critical"):
            # stderr mirror for interactive sessions; already redacted.
            import sys

            print(line, file=sys.stderr)

    def context_window(self) -> list[dict[str, Any]]:
        """Return recent events for crash correlation."""
        return list(self._ring)

    # -- io ---------------------------------------------------------------------
    def _path_for(self, component: str) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / component / f"{day}.jsonl"

    def _write(self, component: str, line: str) -> None:
        path = self._path_for(component)
        with self._lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                if component not in self._pruned:
                    self._prune(path.parent)
                    self._pruned.add(component)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                # Logging must never crash the program. Swallow IO errors.
                pass

    def _prune(self, component_dir: Path) -> None:
        """Delete component log files older than the retention window."""
        cutoff = datetime.now(timezone.utc).timestamp() - self.retention_days * 86400
        try:
            for f in component_dir.glob("*.jsonl"):
                if f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
        except OSError:
            pass


# -- module-level singleton -----------------------------------------------------
_active: Telemetry | None = None


def init_telemetry(
    log_dir: Path,
    session_id: str,
    *,
    retention_days: int = 7,
    echo_errors: bool = False,
) -> Telemetry:
    global _active
    _active = Telemetry(
        log_dir,
        session_id,
        retention_days=retention_days,
        echo_errors=echo_errors,
    )
    return _active


def get_telemetry() -> Telemetry:
    """Return the active telemetry, or a no-op-ish default writing to ./logs."""
    global _active
    if _active is None:
        _active = Telemetry(Path("logs"), session_id="unbound")
    return _active
