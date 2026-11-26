from __future__ import annotations

import pytest

from spark.config.loader import load_config
from spark.config.paths import SparkPaths


@pytest.fixture
def paths(tmp_path) -> SparkPaths:
    p = SparkPaths(
        config_dir=tmp_path / "config",
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
    )
    p.ensure()
    return p


@pytest.fixture
def config():
    return load_config()


@pytest.fixture(autouse=True)
def _isolate_telemetry(tmp_path):
    """No test inherits — or leaks — the process-wide telemetry singleton.

    Seed a tmp-dir logger for every test: an unbound ``get_telemetry()``
    otherwise falls back to ``Path("logs")`` and writes into the repo.
    Tests that call ``init_telemetry(tmp_path / ...)`` simply replace it.
    """
    from spark.telemetry import init_telemetry, jsonl as telemetry_jsonl

    telemetry_jsonl._active = None
    init_telemetry(tmp_path / "logs", session_id="test")
    try:
        yield
    finally:
        telemetry_jsonl._active = None


@pytest.fixture(autouse=True)
def _isolate_secret_registry():
    """Run each test with a clean secret registry, restoring it afterwards."""
    from spark import redaction

    previous = redaction.secret_values()
    redaction.reset_secret_values()
    try:
        yield
    finally:
        redaction.reset_secret_values()
        for value in previous:
            redaction.register_secret_value(value)
