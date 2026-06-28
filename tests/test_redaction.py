from __future__ import annotations

import json

from spark.telemetry import init_telemetry, register_secret_value


def test_secret_value_is_redacted_in_logs(tmp_path):
    log = init_telemetry(tmp_path / "logs", "sess-test")
    register_secret_value("super-secret-token-value")
    log.info("test", "leak_attempt", token_blob="prefix super-secret-token-value suffix")

    files = list((tmp_path / "logs" / "test").glob("*.jsonl"))
    assert files, "no log file written"
    content = files[0].read_text()
    assert "super-secret-token-value" not in content
    assert "***REDACTED***" in content


def test_sensitive_key_name_is_masked(tmp_path):
    log = init_telemetry(tmp_path / "logs", "sess-test2")
    log.info("test", "event", api_key="abc123", authorization="Bearer xyz", safe="ok")
    rec = json.loads(
        list((tmp_path / "logs" / "test").glob("*.jsonl"))[0].read_text().splitlines()[0]
    )
    assert rec["api_key"] == "***REDACTED***"
    assert rec["authorization"] == "***REDACTED***"
    assert rec["safe"] == "ok"


def test_required_log_fields_present(tmp_path):
    log = init_telemetry(tmp_path / "logs", "sess-fields")
    log.info("comp", "evt", extra=1)
    rec = json.loads(
        list((tmp_path / "logs" / "comp").glob("*.jsonl"))[0].read_text().splitlines()[0]
    )
    for field in ("ts", "level", "component", "event", "session_id", "pid"):
        assert field in rec
