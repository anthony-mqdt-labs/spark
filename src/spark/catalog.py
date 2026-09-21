"""The live record of spark's models — kept, and published.

Everything spark knew about its own fleet used to be reachable only by running the
CLI: `spark list` printed a table and the registry was a directory of TOML files.
A consumer — banter's `backend-spark`, an agent CLI, a script — had no
machine-readable view, and no way to learn which port a server actually landed on,
so it guessed. It guessed wrong whenever the preferred port was occupied.

Two artifacts fix that, both under the data dir:

``catalog.json``
    The roster: every registered model with its *current* availability as observed
    on this host (state / location / bytes / when it was checked), plus the live
    instances. Refreshed on every event that can change it — list, download,
    forget, research accept, server ready, server exit. Read this instead of
    scraping `spark list`.

``run/instances/<model>.json``
    One record per live server: written the moment a server is ready,
    heartbeat-updated while it runs, deleted when it stops. The field set is the
    v1 manifest an existing consumer already validates
    (``banter/rust/crates/backend-spark/src/discovery.rs``): ``schema_version`` 1,
    ``api_contract == "openai.chat-completions.v1"``, the exact API ``model_id`` a
    client must send in the request body, and ``model_alias`` (the spark registry
    id). That reader requires a bounded regular non-symlink file, which is how
    these are written.

Neither file is a source of truth. The registry owns intentions, the filesystem
owns reality, and the catalog is a snapshot of the join — regenerable, safe to
delete, and never consulted by spark's own decisions.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config.paths import SparkPaths
from .config.schema import ModelEntry, SparkConfig
from .registry.availability import availability_map

SCHEMA_VERSION = 1
#: Versioned API contract announced in instance records. A consumer uses it to
#: decide whether it understands the endpoint before talking to it.
API_CONTRACT = "openai.chat-completions.v1"
#: Readers bound manifest size; keep well under it (64 KiB in banter's reader).
MAX_RECORD_BYTES = 64 * 1024


def instances_dir(paths: SparkPaths) -> Path:
    """Where live-instance manifests live (created on demand, user-only)."""
    return paths.run_dir / "instances"


def catalog_path(paths: SparkPaths) -> Path:
    return paths.data_dir / "catalog.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write JSON atomically: a reader never sees a half-written record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    tmp.write_text(text, encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return path


# --- live instances ------------------------------------------------------------
def instance_record(
    *,
    alias: str,
    api_model_id: str,
    backend: str,
    base_url: str,
    port: int,
    health_url: str,
    pid: int,
    session_id: str,
    started_at: str,
    updated_at: str | None = None,
    state: str = "ready",
) -> dict[str, Any]:
    """One live-server record, in the shape consumers already validate.

    ``api_model_id`` is what a client must put in the request body ``model``
    field — not the spark alias. For a store-backed model that is an absolute
    path, which is exactly why publishing it matters: nothing else tells a
    consumer what to send.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "pid": int(pid),
        "state": state,
        "base_url": base_url,
        "port": int(port),
        "api_contract": API_CONTRACT,
        "model_id": api_model_id,
        "model_alias": alias,
        "backend": backend,
        "health_url": health_url,
        "started_at": started_at,
        "updated_at": updated_at or _now_iso(),
    }


def publish_instance(paths: SparkPaths, record: dict[str, Any]) -> Path:
    """Write (or heartbeat) one instance manifest, keyed by model alias."""
    alias = str(record.get("model_alias") or "")
    if not alias:
        raise ValueError("instance record requires model_alias")
    d = instances_dir(paths)
    d.mkdir(parents=True, exist_ok=True)
    # These records advertise a live endpoint, so they stay owner-only — the same
    # posture as run_dir itself (paths.ensure chmods it 0700, but a newly created
    # child would otherwise inherit the umask).
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return _atomic_write_json(d / f"{alias}.json", record)


def remove_instance(paths: SparkPaths, alias: str) -> None:
    """Delete an instance manifest (server stopped). Missing is fine."""
    try:
        (instances_dir(paths) / f"{alias}.json").unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _readable(p: Path) -> bool:
    """Mirror the consumer's invariants: bounded, regular, not a symlink."""
    try:
        if p.is_symlink() or not p.is_file():
            return False
        if p.stat().st_size > MAX_RECORD_BYTES:
            return False
    except OSError:
        return False
    return True


def read_instances(paths: SparkPaths) -> list[dict[str, Any]]:
    """Every valid live-instance record, sorted by alias.

    Malformed, oversized or symlinked records are skipped rather than raising:
    this feeds a published file and a listing, and one bad record must not blind
    a consumer to the rest.
    """
    d = instances_dir(paths)
    if not d.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for f in sorted(d.glob("*.json")):
        if not _readable(f):
            continue
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict) and rec.get("schema_version") == SCHEMA_VERSION:
            out.append(rec)
    return out


# --- catalog -------------------------------------------------------------------
def model_record(entry: ModelEntry, availability=None) -> dict[str, Any]:
    """Registry intake joined with what is on disk right now."""
    rec: dict[str, Any] = {
        "id": entry.id,
        "aliases": list(entry.aliases),
        "backend": entry.backend,
        "model_format": entry.model_format,
        "quant": entry.quant,
        "params_billions": entry.params_billions,
        "size_bytes": entry.size_bytes,
        "context_length": entry.context_length,
        "hf_repo": entry.hf_repo,
        "path": entry.path,
        "research_status": entry.research_status,
    }
    if availability is not None:
        rec["availability"] = {
            "state": availability.state,
            "ok": availability.ok,
            "detail": availability.detail,
            "location": availability.location,
            "bytes_present": availability.bytes_present,
            "downloadable": availability.downloadable,
            "checked_at": _now_iso(),
        }
    return rec


def build_catalog(
    paths: SparkPaths,
    config: SparkConfig | None = None,
    *,
    entries: list[ModelEntry] | None = None,
) -> dict[str, Any]:
    """The roster: every registered model + its observed availability + live servers.

    Plus ``discovered``: servable weights on disk with no registry entry, and a
    count of cached-but-not-servable repos. Additive to the v1 contract —
    consumers ignore unknown keys.
    """
    if entries is None:
        from .registry import list_models

        entries = list_models(paths)
    avail = availability_map(entries, paths, config=config)
    try:
        from .inventory import build_inventory

        inv = build_inventory(paths, config=config, with_bytes=False)
        discovered = [
            {
                "id": m.display_id,
                "source": m.source,
                "location": m.location,
                "model_format": m.model_format,
                "quant": m.quant,
            }
            for m in inv.runnable_discovered
        ]
        non_servable_count = len(inv.non_servable)
    except Exception:
        discovered = []
        non_servable_count = 0
    return {
        "schema_version": SCHEMA_VERSION,
        "api_contract": API_CONTRACT,
        "generated_at": _now_iso(),
        "data_dir": str(paths.data_dir),
        "models": [model_record(e, avail.get(e.id)) for e in entries],
        "discovered": discovered,
        "non_servable_cached": non_servable_count,
        "instances": read_instances(paths),
    }


def write_catalog(
    paths: SparkPaths,
    config: SparkConfig | None = None,
    *,
    entries: list[ModelEntry] | None = None,
) -> Path:
    """Refresh ``catalog.json``. Cheap enough to call on every state change."""
    return _atomic_write_json(catalog_path(paths), build_catalog(paths, config, entries=entries))


# --- supervisor-facing helpers -------------------------------------------------
def started_at_stamp() -> str:
    return _now_iso()


class InstancePublisher:
    """Owns the lifecycle of one instance manifest: publish, heartbeat, remove.

    Kept as a small object so the supervisor does not have to remember the port,
    the API model id, or the heartbeat clock across its restart loop.
    """

    def __init__(
        self,
        paths: SparkPaths,
        *,
        alias: str,
        api_model_id: str,
        backend: str,
        base_url: str,
        port: int,
        health_url: str,
        session_id: str,
        heartbeat_s: float = 30.0,
    ) -> None:
        self.paths = paths
        self.alias = alias
        self.api_model_id = api_model_id
        self.backend = backend
        self.base_url = base_url
        self.port = port
        self.health_url = health_url
        self.session_id = session_id
        self.heartbeat_s = heartbeat_s
        self.started_at = started_at_stamp()
        self._next_beat = time.monotonic() + heartbeat_s

    def publish(self) -> Path:
        rec = instance_record(
            alias=self.alias,
            api_model_id=self.api_model_id,
            backend=self.backend,
            base_url=self.base_url,
            port=self.port,
            health_url=self.health_url,
            pid=os.getpid(),
            session_id=self.session_id,
            started_at=self.started_at,
        )
        return publish_instance(self.paths, rec)

    def maybe_heartbeat(self) -> None:
        """Rewrite ``updated_at`` at most every ``heartbeat_s`` seconds."""
        if time.monotonic() < self._next_beat:
            return
        self._next_beat = time.monotonic() + self.heartbeat_s
        self.publish()

    def remove(self) -> None:
        remove_instance(self.paths, self.alias)
