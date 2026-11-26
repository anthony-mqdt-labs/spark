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
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config.paths import SparkPaths
from .config.schema import ModelEntry, SparkConfig, id_file_stem
from .errors import HostError
from .registry.availability import availability_map
from .telemetry import get_telemetry

SCHEMA_VERSION = 1
#: Versioned API contract announced in instance records. A consumer uses it to
#: decide whether it understands the endpoint before talking to it.
API_CONTRACT = "openai.chat-completions.v1"
#: Readers bound manifest size; keep well under it (64 KiB in banter's reader).
MAX_RECORD_BYTES = 64 * 1024
#: A pid can be reused; treat start times this far apart as a different process.
PID_START_TOLERANCE_S = 2.0
#: Keep `ps` calls cheap: a heartbeat may rewrite a manifest every few seconds.
_PID_START_CACHE_TTL_S = 5.0
_PID_START_CACHE_MAX = 256
_PID_START_CACHE: dict[int, tuple[float, float]] = {}


class _LastKnownGood(dict[str, Any]):
    """A previously published roster, reused verbatim after a scan failure.

    A plain dict payload that ``write_catalog`` recognizes so it can leave the
    existing file untouched (same bytes, same inode, same mtime).
    """


def instances_dir(paths: SparkPaths) -> Path:
    """Where live-instance manifests live (created on demand, user-only)."""
    return paths.run_dir / "instances"


def catalog_path(paths: SparkPaths) -> Path:
    return paths.data_dir / "catalog.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dump_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write JSON atomically: a reader never sees a half-written record.

    Each writer gets its own temp file: the CLI refresh and the supervisor's
    ready/exit refresh can run concurrently. The descriptor is locked down
    *before* any byte is written, so the temp is never briefly world-readable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _dump_json(payload)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    fd_owned = True
    try:
        os.fchmod(fd, 0o600)
        fh = os.fdopen(fd, "w", encoding="utf-8")
        fd_owned = False  # the file object owns fd now
        with fh:
            fh.write(text)
            fh.flush()
        os.replace(tmp, path)
    except BaseException:
        if fd_owned:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
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
    pid_started_at: float | None = None,
) -> dict[str, Any]:
    """One live-server record, in the shape consumers already validate.

    ``api_model_id`` is what a client must put in the request body ``model``
    field — not the spark alias. For a store-backed model that is an absolute
    path, which is exactly why publishing it matters: nothing else tells a
    consumer what to send.

    ``pid_started_at`` is additive and optional: older consumers ignore it, and
    manifests written before it existed stay readable. It lets a reader tell a
    recycled PID from the process that actually published the endpoint.
    """
    rec: dict[str, Any] = {
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
    if pid_started_at is not None:
        rec["pid_started_at"] = float(pid_started_at)
    return rec


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
    return _atomic_write_json(d / f"{id_file_stem(alias)}.json", record)


def remove_instance(paths: SparkPaths, alias: str, *, owner_pid: int | None = None) -> None:
    """Delete an instance manifest (server stopped). Missing is fine.

    With ``owner_pid``, only a record that process wrote is removed: a second
    spark serving the same alias overwrites the record, and the first one's
    exit must not then delete the survivor's endpoint.
    """
    p = instances_dir(paths) / f"{id_file_stem(alias)}.json"
    try:
        if owner_pid is not None:
            rec = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(rec, dict) and rec.get("pid") != owner_pid:
                return
        p.unlink()
    except (OSError, ValueError):
        return


def _process_start_epoch(pid: int) -> float | None:
    """Best-effort process start time (epoch seconds), or None if unknown.

    ``ps -o lstart=`` is the one portable-enough source on macOS/Linux without
    adding a dependency. A short cache keeps heartbeats from spawning a process
    on every rewrite. Never raises: callers must stay able to read the catalog
    on a host where ``ps`` is restricted or the locale is unfamiliar.
    """
    now = time.monotonic()
    cached = _PID_START_CACHE.get(pid)
    if cached is not None and now - cached[0] < _PID_START_CACHE_TTL_S:
        return cached[1]
    try:
        out = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    stamp = " ".join(out.stdout.split())
    if not stamp:
        return None
    try:
        # lstart is local wall-clock time; timestamp() interprets it as such.
        epoch = datetime.strptime(stamp, "%a %b %d %H:%M:%S %Y").timestamp()
    except ValueError:
        return None
    if len(_PID_START_CACHE) >= _PID_START_CACHE_MAX:
        _PID_START_CACHE.clear()
    _PID_START_CACHE[pid] = (now, epoch)
    return epoch


def _pid_alive(pid: object, pid_started_at: object = None) -> bool:
    """Whether the recorded endpoint is still owned by the process that wrote it."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # exists, owned by someone else
    except OSError:
        return False

    if pid_started_at is None:
        # Legacy manifest: no start stamp to cross-check, keep prior behaviour.
        return True
    try:
        recorded = float(str(pid_started_at))
    except (TypeError, ValueError, OverflowError):
        return True  # malformed optional field is not grounds for hiding a server
    try:
        actual = _process_start_epoch(pid)
    except Exception:  # noqa: BLE001 - a failed lookup must never blind a reader
        return True
    if actual is None:
        return True
    return abs(actual - recorded) <= PID_START_TOLERANCE_S


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
        if not (isinstance(rec, dict) and rec.get("schema_version") == SCHEMA_VERSION):
            continue
        try:
            live = _pid_alive(rec.get("pid"), rec.get("pid_started_at"))
        except Exception:  # noqa: BLE001 - one odd record must not blind the rest
            live = False
        if not live:
            # Left by a spark that was SIGKILLed, or by an earlier process that
            # held this pid (the id was recycled). Publishing either would
            # advertise a dead endpoint.
            continue
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


def _last_known_good(paths: SparkPaths) -> _LastKnownGood | None:
    """The previously published roster, if it is readable and still schema v1."""
    try:
        prior = json.loads(catalog_path(paths).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(prior, dict) or prior.get("schema_version") != SCHEMA_VERSION:
        return None
    return _LastKnownGood(prior)


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
    except (OSError, ValueError) as exc:
        # An inventory failure is not an empty inventory. Publishing discovered=
        # [] would tell the Rust consumer the disk has no servable weights when
        # we simply failed to look, and would erase the cached-repo count.
        get_telemetry().warn(
            "catalog", "inventory_failed",
            error_type=type(exc).__name__, error=repr(exc),
            data_dir=str(paths.data_dir),
        )
        previous = _last_known_good(paths)
        if previous is not None:
            return previous
        raise HostError(
            "Cannot publish the catalog: the model inventory scan failed and "
            "there is no last-known-good catalog to reuse",
            remediation=[
                "Check read permissions on the spark data, cache and store directories",
                "Run `spark list` to confirm the registry and weights are readable",
                "Retry the failed operation once the filesystem or registry error is fixed",
            ],
            context={"data_dir": str(paths.data_dir)},
            cause=exc,
        ) from exc
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
    """Refresh ``catalog.json``. Cheap enough to call on every state change.

    If an inventory scan failed, ``build_catalog`` returns the last-known-good
    roster (or raises a typed error); an unchanged file is left untouched.
    """
    payload = build_catalog(paths, config, entries=entries)
    target = catalog_path(paths)
    if isinstance(payload, _LastKnownGood):
        return target
    return _atomic_write_json(target, payload)


def safe_write_catalog(
    paths: SparkPaths,
    config: SparkConfig | None = None,
    *,
    component: str = "catalog",
    event: str = "write_failed",
    **context: Any,
) -> Path | None:
    """Publishing is a courtesy: warn through telemetry, never fail the caller.

    Shared by the CLI refresh and the supervisor's ready/exit refreshes so the
    try/except and the structured warning live in one place. Extra keyword
    context (e.g. ``model_id``) is appended to the warning payload.
    """
    try:
        return write_catalog(paths, config)
    except Exception as exc:  # noqa: BLE001 - courtesy, never a gate
        get_telemetry().warn(
            component, event,
            error_type=type(exc).__name__, error=repr(exc), **context,
        )
        return None


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
            pid_started_at=_process_start_epoch(os.getpid()),
        )
        return publish_instance(self.paths, rec)

    def maybe_heartbeat(self) -> None:
        """Rewrite ``updated_at`` at most every ``heartbeat_s`` seconds."""
        if time.monotonic() < self._next_beat:
            return
        self._next_beat = time.monotonic() + self.heartbeat_s
        self.publish()

    def remove(self) -> None:
        remove_instance(self.paths, self.alias, owner_pid=os.getpid())
