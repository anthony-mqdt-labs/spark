"""Hardening regressions for the published catalog.

Four failure modes are pinned here:

* a fixed ``catalog.json.tmp`` path that two writers race on, briefly exposing
  world-readable content before ``chmod``;
* an inventory scan failure silently publishing an empty ``discovered`` roster;
* a recycled PID making a stale instance manifest look live;
* duplicated "publishing is a courtesy" error handling at call sites.

Written before the fixes; each test fails on the unhardened module.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

import spark.catalog as catalog
from spark.config.schema import ModelEntry
from spark.errors import SparkError
from spark.registry import save_model


class _Recorder:
    """Minimal telemetry stand-in: records warn() calls in order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def warn(self, component: str, event: str, /, **payload: Any) -> None:
        self.events.append((component, event, payload))


@pytest.fixture
def warnings(monkeypatch) -> _Recorder:
    """Capture telemetry warnings regardless of where catalog imports it from."""
    rec = _Recorder()
    monkeypatch.setattr(
        "spark.telemetry.jsonl.get_telemetry", lambda: rec, raising=False
    )
    monkeypatch.setattr("spark.telemetry.get_telemetry", lambda: rec, raising=False)
    monkeypatch.setattr(catalog, "get_telemetry", lambda: rec, raising=False)
    return rec


def _record(**kw):
    base = dict(
        alias="m-live",
        api_model_id="org/Model",
        backend="mlx_lm",
        base_url="http://127.0.0.1:8095/v1",
        port=8095,
        health_url="http://127.0.0.1:8095/v1/models",
        pid=os.getpid(),
        session_id="s1",
        started_at="2026-09-18T19:00:00+00:00",
    )
    base.update(kw)
    return catalog.instance_record(**base)


def _seed(paths, *ids):
    for i in ids:
        save_model(ModelEntry(id=i, model_format="mlx"), paths)


# --- 1. atomic write ------------------------------------------------------------
def test_atomic_write_uses_a_unique_temp_per_call_and_cleans_up(paths, monkeypatch):
    """Concurrent writers must not share a temp path; failures leave no debris."""
    seen: list[Path] = []

    def stop(src, dst):
        seen.append(Path(src))
        raise RuntimeError("replace refused")

    monkeypatch.setattr(catalog.os, "replace", stop)

    for n in (1, 2):
        with pytest.raises(RuntimeError, match="replace refused"):
            catalog._atomic_write_json(paths.data_dir / "catalog.json", {"n": n})

    assert len(seen) == 2
    assert seen[0] != seen[1], "two calls reused the same temp path"
    assert not list(paths.data_dir.glob("*.tmp"))


def test_atomic_write_temp_is_owner_only_with_content_before_replace(paths, monkeypatch):
    seen: dict[str, Any] = {}
    real_replace = catalog.os.replace

    def inspect(src, dst):
        src = Path(src)
        seen["mode"] = src.stat().st_mode & 0o777
        seen["text"] = src.read_text(encoding="utf-8")
        seen["tmp_exists"] = src.exists()
        return real_replace(src, dst)

    monkeypatch.setattr(catalog.os, "replace", inspect)
    target = paths.data_dir / "catalog.json"
    catalog._atomic_write_json(target, {"answer": 42})

    assert seen["mode"] == 0o600
    assert json.loads(seen["text"]) == {"answer": 42}
    assert seen["tmp_exists"] is True
    assert (target.stat().st_mode & 0o777) == 0o600
    assert json.loads(target.read_text(encoding="utf-8")) == {"answer": 42}


def test_atomic_write_leaves_no_temp_when_serialization_fails(paths):
    target = paths.data_dir / "catalog.json"
    with pytest.raises(TypeError):
        catalog._atomic_write_json(target, {"bad": object()})
    assert not list(paths.data_dir.glob("*.tmp"))
    assert not target.exists()


def test_atomic_write_failure_does_not_touch_existing_file(paths, monkeypatch):
    target = paths.data_dir / "catalog.json"
    catalog._atomic_write_json(target, {"version": "good"})
    before = target.read_bytes()

    def stop(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(catalog.os, "replace", stop)
    with pytest.raises(OSError):
        catalog._atomic_write_json(target, {"version": "bad"})

    assert target.read_bytes() == before
    assert not list(paths.data_dir.glob("*.tmp"))


# --- 2. inventory failure never publishes a partial roster -----------------------
def _break_inventory(monkeypatch, exc=OSError):
    def boom(*args, **kwargs):
        raise exc("scan failed")

    monkeypatch.setattr("spark.inventory.build_inventory", boom)


def test_inventory_failure_keeps_last_known_good_and_warns(paths, warnings, monkeypatch):
    _seed(paths, "kept")
    path = catalog.write_catalog(paths)
    before = path.read_bytes()
    assert json.loads(before)["discovered"] == []

    _break_inventory(monkeypatch)
    got = catalog.write_catalog(paths)

    assert got == path
    assert path.read_bytes() == before, "last-known-good catalog was overwritten"
    assert warnings.events
    component, event, payload = warnings.events[-1]
    assert (component, event) == ("catalog", "inventory_failed")
    assert payload["error_type"] == "OSError"


def test_inventory_failure_without_prior_catalog_raises_typed_error(
    paths, warnings, monkeypatch
):
    _seed(paths, "m")
    _break_inventory(monkeypatch)

    with pytest.raises(SparkError) as caught:
        catalog.write_catalog(paths)

    assert caught.value.remediation, "typed error needs do-this-next steps"
    assert not catalog.catalog_path(paths).exists()
    assert warnings.events[-1][2]["error_type"] == "OSError"


def test_inventory_failure_does_not_publish_empty_roster(paths, warnings, monkeypatch):
    """Even a direct build() must not hand back discovered=[], count=0."""
    _seed(paths, "m")
    _break_inventory(monkeypatch)
    with pytest.raises(SparkError):
        catalog.build_catalog(paths)
    assert not catalog.catalog_path(paths).exists()


# --- 3. PID recycling ------------------------------------------------------------
def test_recycled_pid_is_filtered_out(paths, monkeypatch):
    now = time.time()
    monkeypatch.setattr(catalog, "_process_start_epoch", lambda pid: now)
    # Same PID is alive, but this record belongs to a process that started an
    # hour ago — the PID has been reused since the manifest was written.
    catalog.publish_instance(paths, _record(pid_started_at=now - 3600))
    assert [r["model_alias"] for r in catalog.read_instances(paths)] == []


def test_matching_pid_start_is_kept(paths, monkeypatch):
    now = time.time()
    monkeypatch.setattr(catalog, "_process_start_epoch", lambda pid: now)
    catalog.publish_instance(paths, _record(pid_started_at=now - 0.5))
    assert [r["model_alias"] for r in catalog.read_instances(paths)] == ["m-live"]


def test_legacy_record_without_pid_started_at_is_kept(paths, monkeypatch):
    monkeypatch.setattr(
        catalog, "_process_start_epoch", lambda pid: pytest.fail("should not be called")
    )
    catalog.publish_instance(paths, _record())  # no optional field
    assert [r["model_alias"] for r in catalog.read_instances(paths)] == ["m-live"]


def test_start_lookup_failure_never_raises_from_read_instances(paths, monkeypatch):
    def broken(pid):
        raise OSError("ps unavailable")

    monkeypatch.setattr(catalog, "_process_start_epoch", broken)
    catalog.publish_instance(paths, _record(pid_started_at=time.time()))
    assert [r["model_alias"] for r in catalog.read_instances(paths)] == ["m-live"]


def test_pid_started_at_is_an_optional_additive_field():
    assert "pid_started_at" not in _record()  # legacy / consumer field set intact
    rec = _record(pid_started_at=1234.5)
    assert rec["pid_started_at"] == 1234.5
    assert rec["schema_version"] == catalog.SCHEMA_VERSION


def test_publisher_records_the_process_start_time(paths, monkeypatch):
    monkeypatch.setattr(catalog, "_process_start_epoch", lambda pid: 4321.0)
    pub = catalog.InstancePublisher(
        paths,
        alias="m-live",
        api_model_id="org/Model",
        backend="mlx_lm",
        base_url="http://127.0.0.1:8095/v1",
        port=8095,
        health_url="http://127.0.0.1:8095/v1/models",
        session_id="s1",
    )
    path = pub.publish()
    assert json.loads(path.read_text(encoding="utf-8"))["pid_started_at"] == 4321.0


def test_process_start_helper_parses_and_caches(monkeypatch):
    """The helper is monkeypatchable and uses no dependency beyond ps."""
    assert callable(catalog._process_start_epoch)
    called: list[int] = []

    class Result:
        returncode = 0
        stdout = "Wed Nov 25 10:04:05 2026\n"

    def fake_run(cmd, **kwargs):
        called.append(1)
        return Result()

    monkeypatch.setattr(catalog.subprocess, "run", fake_run)
    catalog._PID_START_CACHE.clear()
    first = catalog._process_start_epoch(os.getpid())
    second = catalog._process_start_epoch(os.getpid())
    assert first is not None
    assert second == first
    assert len(called) == 1, "short in-module cache should avoid a second ps"


# --- 4. shared safe writer -------------------------------------------------------
def test_safe_write_catalog_returns_the_path_on_success(paths):
    _seed(paths, "m")
    got = catalog.safe_write_catalog(paths, None)
    assert got == catalog.catalog_path(paths)
    assert json.loads(got.read_text(encoding="utf-8"))["models"]


def test_safe_write_catalog_warns_and_returns_none_on_failure(
    paths, warnings, monkeypatch
):
    def boom(*args, **kwargs):
        raise RuntimeError("write failed")

    monkeypatch.setattr(catalog, "build_catalog", boom)
    got = catalog.safe_write_catalog(paths, None, model_id="m")
    assert got is None
    component, event, payload = warnings.events[-1]
    assert (component, event) == ("catalog", "write_failed")
    assert payload["error_type"] == "RuntimeError"
    assert payload["model_id"] == "m"
