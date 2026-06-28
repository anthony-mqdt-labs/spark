"""Structured JSONL telemetry for spark.

Implements the mandatory logging contract from the repo CLAUDE.md: one JSON
object per line to ``logs/<component>/<YYYY-MM-DD>.jsonl`` with the required
fields (``ts, level, component, event, session_id, pid``), daily rotation, 7-day
retention, and a hard secret-redaction filter.
"""

from .jsonl import (
    Telemetry,
    get_telemetry,
    init_telemetry,
    register_secret_value,
)

__all__ = [
    "Telemetry",
    "get_telemetry",
    "init_telemetry",
    "register_secret_value",
]
