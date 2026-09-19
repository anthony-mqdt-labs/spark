"""The live record: catalog.json and run/instances/<model>.json.

The instance record is a *published contract* — banter's `backend-spark`
validates a fixed field set, requires schema_version 1, requires
`api_contract == "openai.chat-completions.v1"`, rejects symlinks and files over
64 KiB, and checks the recorded pid against the process table. These tests pin
that contract so a rename or a dropped field cannot pass review silently.
"""

from __future__ import annotations

import json
import os

import pytest

from spark.catalog import (
    API_CONTRACT,
    MAX_RECORD_BYTES,
    SCHEMA_VERSION,
    InstancePublisher,
    build_catalog,
    catalog_path,
    instance_record,
    instances_dir,
    publish_instance,
    read_instances,
    remove_instance,
    write_catalog,
)
from spark.config.schema import ModelEntry
from spark.registry import save_model

#: Exactly the fields a consumer's discovery record declares. Kept literal
#: (not derived from the implementation) so the test fails if the publisher
#: drifts from the consumer.
CONSUMER_FIELDS = {
    "schema_version",
    "session_id",
    "pid",
    "state",
    "base_url",
    "port",
    "api_contract",
    "model_id",
    "model_alias",
    "backend",
    "health_url",
    "started_at",
    "updated_at",
}


def _record(**kw):
    base = dict(
        alias="m-1b",
        api_model_id="org/Model-1B",
        backend="mlx_lm",
        base_url="http://127.0.0.1:8095/v1",
        port=8095,
        health_url="http://127.0.0.1:8095/v1/models",
        pid=os.getpid(),
        session_id="abc123",
        started_at="2026-09-18T19:00:00+00:00",
    )
    base.update(kw)
    return instance_record(**base)


def _seed(paths, *ids):
    for i in ids:
        save_model(ModelEntry(id=i, model_format="mlx", quant="q4"), paths)


# --- the published contract ----------------------------------------------------
def test_instance_record_matches_the_consumer_field_set():
    rec = _record()
    assert set(rec) == CONSUMER_FIELDS


def test_instance_record_declares_schema_and_api_contract():
    rec = _record()
    assert rec["schema_version"] == SCHEMA_VERSION == 1
    assert rec["api_contract"] == API_CONTRACT == "openai.chat-completions.v1"


def test_instance_record_carries_the_api_model_id_not_just_the_alias():
    """A store-backed model is addressed by absolute path; consumers need it."""
    rec = _record(
        alias="minicpm5-2b-8bit",
        api_model_id="/Users/x/.local/share/spark/store/minicpm5-2b-8bit",
    )
    assert rec["model_alias"] == "minicpm5-2b-8bit"
    assert rec["model_id"].startswith("/")


def test_instance_record_requires_an_alias():
    with pytest.raises(ValueError):
        publish_instance(None, {"model_id": "x"})  # type: ignore[arg-type]


# --- publishing / retracting ----------------------------------------------------
def test_publish_writes_one_file_per_alias(paths):
    publish_instance(paths, _record(alias="alpha"))
    publish_instance(paths, _record(alias="beta"))
    d = instances_dir(paths)
    assert sorted(p.name for p in d.glob("*.json")) == ["alpha.json", "beta.json"]
    assert (d.stat().st_mode & 0o777) in (0o700, 0o750)  # parent stays private


def test_published_record_is_owner_only(paths):
    path = publish_instance(paths, _record())
    assert (path.stat().st_mode & 0o777) == 0o600


def test_remove_instance_is_idempotent(paths):
    publish_instance(paths, _record(alias="gone"))
    remove_instance(paths, "gone")
    remove_instance(paths, "gone")  # no raise
    assert read_instances(paths) == []


def test_heartbeat_rewrites_updated_at(paths):
    pub = InstancePublisher(
        paths, alias="hb", api_model_id="org/M", backend="mlx_lm",
        base_url="http://127.0.0.1:8095/v1", port=8095,
        health_url="http://127.0.0.1:8095/v1/models", session_id="s",
        heartbeat_s=0.0,
    )
    path = pub.publish()
    first = json.loads(path.read_text())["updated_at"]
    pub.maybe_heartbeat()
    second = json.loads(path.read_text())["updated_at"]
    assert first <= second
    pub.remove()
    assert not path.exists()


# --- reading defensively --------------------------------------------------------
def test_read_skips_malformed_oversize_symlink_and_foreign_schema(paths, tmp_path):
    d = instances_dir(paths)
    d.mkdir(parents=True, exist_ok=True)
    publish_instance(paths, _record(alias="good"))
    (d / "broken.json").write_text("{not json")
    (d / "huge.json").write_text("x" * (MAX_RECORD_BYTES + 10))
    (d / "foreign.json").write_text(json.dumps({"schema_version": 99}))
    (d / "notafile.json").mkdir()
    # A symlink is refused even when it points at a perfectly good record, and
    # the target lives outside the directory so it is not read on its own merit.
    target = tmp_path / "outside.json"
    target.write_text(json.dumps(_record(alias="outside")))
    (d / "link.json").symlink_to(target)

    got = [r["model_alias"] for r in read_instances(paths)]
    assert got == ["good"]  # symlink + junk all skipped, good record survives


def test_read_of_absent_dir_is_empty(paths):
    assert read_instances(paths) == []


# --- catalog --------------------------------------------------------------------
def test_catalog_joins_registry_availability_and_instances(paths):
    store = paths.model_store_dir / "present"
    store.mkdir(parents=True, exist_ok=True)
    (store / "w.safetensors").write_bytes(b"w" * 32)
    save_model(ModelEntry(id="present", path=str(store), model_format="mlx"), paths)
    save_model(ModelEntry(id="ghost", hf_repo="org/Ghost", model_format="mlx"), paths)
    publish_instance(paths, _record(alias="present"))

    cat = build_catalog(paths)
    assert cat["schema_version"] == SCHEMA_VERSION
    assert cat["api_contract"] == API_CONTRACT
    by_id = {m["id"]: m for m in cat["models"]}
    assert by_id["present"]["availability"]["state"] == "local"
    assert by_id["present"]["availability"]["ok"] is True
    assert by_id["ghost"]["availability"]["state"] == "missing"
    assert by_id["ghost"]["availability"]["ok"] is False
    # availability is timestamped: the catalog is a *live* record, not a dump
    assert by_id["present"]["availability"]["checked_at"]
    assert [i["model_alias"] for i in cat["instances"]] == ["present"]


def test_write_catalog_is_atomic_and_parseable(paths):
    _seed(paths, "a")
    path = write_catalog(paths)
    assert path == catalog_path(paths)
    assert path.parent == paths.data_dir
    json.loads(path.read_text())
    assert not list(paths.data_dir.glob("*.tmp"))


def test_catalog_reflects_a_removed_model(paths):
    _seed(paths, "keep", "drop")
    write_catalog(paths)
    from spark.registry import delete_model

    delete_model("drop", paths)
    ids = [m["id"] for m in build_catalog(paths)["models"]]
    assert ids == ["keep"]


# --- supervisor integration -----------------------------------------------------
def test_supervisor_publishes_then_retracts_the_instance(paths, monkeypatch):
    """End-to-end through the supervisor: published while serving, gone after."""
    from spark.config.loader import load_config
    from spark.config.schema import DetectSpec, RuntimeDef, ServerSpec
    from spark.probe.host import HostProfile, RuntimeStatus
    from spark.runner import Supervisor
    from helpers import FakeStore

    cfg = load_config()
    rt = RuntimeDef(
        name="fake", detect=DetectSpec(binary="/bin/true"),
        server=ServerSpec(binary="/bin/true", default_port=8095, ready_timeout_s=1.0),
        requires_local_weights=False,
    )
    cfg.runtimes["fake"] = rt
    from spark.runtimes.base import Backend

    backend = Backend(rt, cfg)
    profile = HostProfile(
        os="Darwin", arch="arm64", chip="Apple M2", total_memory_bytes=16 * 2**30,
        cpu_count=8, probed_at=0.0,
        runtimes={"fake": RuntimeStatus(name="fake", available=True)},
    )
    sup = Supervisor(backend, ModelEntry(id="m-live"), cfg, profile, paths, FakeStore())
    save_model(ModelEntry(id="m-live", model_format="mlx"), paths)  # catalog lists the registry
    sup._spawn = lambda cmd, env: None        # noqa: SLF001
    sup._terminate = lambda: None             # noqa: SLF001
    sup._is_healthy = lambda url: True        # noqa: SLF001

    observed: dict = {}

    def fake_monitor(url: str) -> str:
        recs = read_instances(paths)
        observed["during"] = recs
        return "stop_requested"

    monkeypatch.setattr(sup, "_monitor", fake_monitor)
    sup.run()

    assert len(observed["during"]) == 1
    live = observed["during"][0]
    assert live["model_alias"] == "m-live"
    assert live["state"] == "ready"
    assert live["base_url"].endswith("/v1")
    assert live["health_url"].endswith("/v1/models")
    # retracted on exit, and the catalog was refreshed as part of the run
    assert read_instances(paths) == []
    cat = json.loads(catalog_path(paths).read_text())
    assert cat["instances"] == []
    assert [m["id"] for m in cat["models"]] == ["m-live"]
