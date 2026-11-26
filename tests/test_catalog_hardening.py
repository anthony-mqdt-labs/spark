"""Hardening regressions for the published catalog.

Four failure modes are pinned here:

* a fixed ``catalog.json.tmp`` path that two writers race on, briefly exposing
  world-readable content before ``chmod``;
* an inventory scan failure silently publishing an empty ``discovered`` roster;
* a SIGKILLed publisher leaving a stale instance manifest that looks live;
* duplicated "publishing is a courtesy" error handling at call sites;
* two availability maps (and two hub scans) per catalog build.

Written before the fixes; each test fails on the unhardened module.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any

import pytest

import spark.catalog as catalog
import spark.inventory as inventory
import spark.registry.availability as availability
from spark.config.schema import ModelEntry
from spark.errors import CatalogError, SparkError
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
    monkeypatch.setattr("spark.telemetry.jsonl.get_telemetry", lambda: rec, raising=False)
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


def test_inventory_failure_without_prior_catalog_raises_typed_error(paths, warnings, monkeypatch):
    _seed(paths, "m")
    _break_inventory(monkeypatch)

    with pytest.raises(CatalogError) as caught:
        catalog.write_catalog(paths)

    assert isinstance(caught.value, SparkError)
    assert caught.value.code == "RUN_CATALOG"
    assert caught.value.remediation, "typed error needs do-this-next steps"
    assert caught.value.context["data_dir"] == str(paths.data_dir)
    assert not catalog.catalog_path(paths).exists()
    assert warnings.events[-1][2]["error_type"] == "OSError"


def test_inventory_failure_does_not_publish_empty_roster(paths, warnings, monkeypatch):
    """Even a direct build() must not hand back discovered=[], count=0."""
    _seed(paths, "m")
    _break_inventory(monkeypatch)
    with pytest.raises(SparkError):
        catalog.build_catalog(paths)
    assert not catalog.catalog_path(paths).exists()


# --- 3. kernel flock liveness ----------------------------------------------------
def _publisher(paths):
    return catalog.InstancePublisher(
        paths,
        alias="m-live",
        api_model_id="org/Model",
        backend="mlx_lm",
        base_url="http://127.0.0.1:8095/v1",
        port=8095,
        health_url="http://127.0.0.1:8095/v1/models",
        session_id="s1",
    )


def _sigkill(pub):
    """The kernel drops flock on SIGKILL, but the lock file itself survives."""
    if pub._lock_fd is not None:  # noqa: SLF001
        os.close(pub._lock_fd)  # noqa: SLF001
        pub._lock_fd = None  # noqa: SLF001


def test_live_publisher_holds_an_exclusive_lock(paths):
    pub = _publisher(paths)
    pub.publish()
    lock = catalog.instance_lock_path(paths, "m-live")
    assert lock.exists()
    assert (lock.stat().st_mode & 0o777) == 0o600
    fd = os.open(lock, os.O_RDWR)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(fd)
    assert [r["model_alias"] for r in catalog.read_instances(paths)] == ["m-live"]
    pub.remove()


def test_sigkilled_publisher_record_is_filtered_out(paths):
    """A dead owner's lock is free for the taking: the manifest must be skipped."""
    pub = _publisher(paths)
    pub.publish()
    lock = catalog.instance_lock_path(paths, "m-live")
    _sigkill(pub)  # no remove(): exactly what SIGKILL leaves behind
    assert lock.exists(), "a SIGKILL does not unlink the lock file"
    assert catalog.read_instances(paths) == []


def test_remove_releases_and_unlinks_the_lock(paths):
    pub = _publisher(paths)
    pub.publish()
    lock = catalog.instance_lock_path(paths, "m-live")
    pub.remove()
    assert not lock.exists()
    assert not catalog.instances_dir(paths).joinpath("m-live.json").exists()
    # A fresh publisher can own the alias again after a clean exit.
    again = _publisher(paths)
    again.publish()
    assert [r["model_alias"] for r in catalog.read_instances(paths)] == ["m-live"]
    again.remove()


def test_legacy_record_without_lock_falls_back_to_pid_check(paths, monkeypatch):
    catalog.publish_instance(paths, _record())  # hand-written: no lock file
    assert not catalog.instance_lock_path(paths, "m-live").exists()
    assert [r["model_alias"] for r in catalog.read_instances(paths)] == ["m-live"]

    def dead(pid):
        raise ProcessLookupError

    monkeypatch.setattr(catalog.os, "kill", dead)
    assert catalog.read_instances(paths) == []


def test_lock_probe_failure_never_raises_from_read_instances(paths, monkeypatch):
    pub = _publisher(paths)
    pub.publish()

    def broken(*args, **kwargs):
        raise OSError("flock unavailable")

    monkeypatch.setattr(catalog.fcntl, "flock", broken)
    assert [r["model_alias"] for r in catalog.read_instances(paths)] == ["m-live"]


def test_pid_start_fields_are_gone(paths):
    """The published shape is back to the consumer's v1 field set."""
    publisher = catalog.InstancePublisher(
        paths,
        alias="m-live",
        api_model_id="org/Model",
        backend="mlx_lm",
        base_url="http://127.0.0.1:8095/v1",
        port=8095,
        health_url="http://127.0.0.1:8095/v1/models",
        session_id="s1",
    )
    publisher.publish()
    try:
        published = json.loads(
            (catalog.instances_dir(paths) / "m-live.json").read_text(encoding="utf-8")
        )
        assert "pid_started_at" not in published
        assert set(published) == set(_record())
    finally:
        publisher.remove()
    assert not hasattr(catalog, "_process_start_epoch")
    assert not hasattr(catalog, "PID_START_TOLERANCE_S")
    assert not hasattr(catalog, "subprocess")  # no more ps fork on read/heartbeat
    with pytest.raises(TypeError):
        _record(pid_started_at=1234.5)


# --- 4. shared safe writer -------------------------------------------------------
def test_safe_write_catalog_returns_the_path_on_success(paths):
    _seed(paths, "m")
    got = catalog.safe_write_catalog(paths, None)
    assert got == catalog.catalog_path(paths)
    assert json.loads(got.read_text(encoding="utf-8"))["models"]


def test_safe_write_catalog_warns_and_returns_none_on_failure(paths, warnings, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("write failed")

    monkeypatch.setattr(catalog, "build_catalog", boom)
    got = catalog.safe_write_catalog(paths, None, model_id="m")
    assert got is None
    component, event, payload = warnings.events[-1]
    assert (component, event) == ("catalog", "write_failed")
    assert payload["error_type"] == "RuntimeError"
    assert payload["model_id"] == "m"


# --- 5. one scan per catalog build -----------------------------------------------
def test_catalog_build_shares_one_availability_map_and_one_hub_scan(paths, monkeypatch):
    """build_catalog computes availability once and hands it to build_inventory."""
    calls = {"avail": 0, "hub": 0}
    real_avail = catalog.availability_map
    real_scan = inventory.scan_hub_cache

    def counted_avail(*args, **kwargs):
        calls["avail"] += 1
        return real_avail(*args, **kwargs)

    def no_second_map(*args, **kwargs):
        raise AssertionError("build_inventory recomputed availability")

    def counted_scan(*args, **kwargs):
        calls["hub"] += 1
        return real_scan(*args, **kwargs)

    monkeypatch.setattr(catalog, "availability_map", counted_avail)
    # inventory imports availability_map at call time; if it builds its own map
    # this raises instead of silently doing the work twice.
    monkeypatch.setattr(availability, "availability_map", no_second_map)
    monkeypatch.setattr(inventory, "scan_hub_cache", counted_scan)

    payload = catalog.build_catalog(paths)
    assert payload["schema_version"] == catalog.SCHEMA_VERSION
    assert calls == {"avail": 1, "hub": 1}
