"""Regression tests: rendered errors must be redacted, not just logged."""

from __future__ import annotations

import spark.cli.app as app
from spark.errors import SparkError
from spark.redaction import REDACTED, register_secret_value, reset_secret_values


class _FakeTelemetry:
    def critical(self, *args, **kwargs):
        pass

    def context_window(self):
        return []


def test_unexpected_error_message_is_scrubbed_before_rendering(monkeypatch):
    reset_secret_values()
    register_secret_value("very-secret-value-1234")
    seen: dict[str, SparkError] = {}
    monkeypatch.setattr(app, "render_error", lambda err: seen.setdefault("err", err))
    monkeypatch.setattr("spark.telemetry.get_telemetry", lambda: _FakeTelemetry())
    monkeypatch.delenv("SPARK_DEBUG", raising=False)

    app._render_unexpected(RuntimeError("boom: very-secret-value-1234"))

    err = seen["err"]
    assert "very-secret-value-1234" not in str(err)
    assert REDACTED in str(err)
    reset_secret_values()
