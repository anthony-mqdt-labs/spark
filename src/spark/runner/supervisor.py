"""Process supervisor for inference servers.

Responsibilities:
- Preflight gates: backend available, local model path exists, free port chosen,
  memory/disk budget honored (unless forced).
- Spawn the server with secrets injected into the child env only.
- Wait for readiness via the backend's health endpoint.
- Monitor liveness; on unexpected exit, restart with exponential backoff up to a
  bound, then surface an actionable terminal failure.
- Graceful shutdown on SIGINT/SIGTERM (SIGTERM -> grace -> SIGKILL).

Every transition is logged as structured JSONL (component ``supervisor``).
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..budget import check_disk, check_memory
from ..catalog import InstancePublisher, read_instances, safe_write_catalog
from ..inventory import infer_quant_and_params
from ..config.paths import SparkPaths
from ..config.schema import ModelEntry, SparkConfig, id_file_stem
from ..errors import (
    HealthCheckError,
    InsufficientDiskError,
    InsufficientMemoryError,
    LaunchError,
    ModelNotFoundError,
    PortInUseError,
    RecoveryExhaustedError,
    WeightsMissingError,
)
from ..probe.host import HostProfile
from ..redaction import scrub_string
from ..registry.availability import MISSING, resolve_availability
from ..runtimes.base import Backend
from ..secrets.store import SecretStore
from ..telemetry import get_telemetry


@dataclass
class LaunchResult:
    endpoint: str
    backend: str
    port: int
    restarts: int


@dataclass
class ReadyInfo:
    """Handed to the ``on_ready`` callback the moment the server first goes healthy,
    so the CLI can print a clean startup banner instead of the child's raw logs."""

    base_url: str
    backend: str
    port: int
    elapsed_s: float
    log_path: str
    note: str = ""  # e.g. the preferred port was taken, so this is a fallback
    api_model_id: str = ""  # exactly what a client must send as request `model`


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _pick_port(host: str, preferred: int, lo: int, hi: int) -> int:
    if _port_free(host, preferred):
        return preferred
    for p in range(lo, hi + 1):
        if _port_free(host, p):
            return p
    raise PortInUseError(
        f"No free port in range {lo}-{hi} (preferred {preferred} busy).",
        remediation=[
            "Free a port or widen general.port_range in your config.",
            f"Find the holder: lsof -nP -iTCP:{preferred} -sTCP:LISTEN",
        ],
        context={"preferred": preferred, "range": [lo, hi]},
    )


def _rss_bytes(pid: int) -> int:
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2, check=False,
        )
        kb = out.stdout.strip()
        return int(kb) * 1024 if kb.isdigit() else 0
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


class _ScrubbingLogPump:
    """Drain a child's stdout/stderr through a pipe, scrubbing before disk.

    A child writes to a file *descriptor*, so wrapping the Python file object in
    a redactor would not intercept a single byte. The parent keeps the read end
    of a pipe and a daemon thread scrubs complete lines before appending them to
    the log file. One pump instance can serve several spawns (restarts).
    """

    def __init__(self, fh) -> None:
        self._fh = fh
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def start(self) -> int:
        """Return the write fd to hand to ``Popen`` (parent closes it thereafter)."""
        read_fd, write_fd = os.pipe()
        thread = threading.Thread(
            target=self._drain, args=(read_fd,), daemon=True,
            name="spark-server-log",
        )
        thread.start()
        self._threads.append(thread)
        return write_fd

    def join(self, timeout: float = 1.0) -> None:
        for thread in self._threads:
            thread.join(timeout)

    def _drain(self, read_fd: int) -> None:
        pending = b""
        try:
            while True:
                chunk = os.read(read_fd, 65536)
                if not chunk:
                    break
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    self._write(line + b"\n")
            if pending:
                self._write(pending)
        except OSError:
            pass  # child/parent closed the pipe mid-read; nothing left to log
        finally:
            try:
                os.close(read_fd)
            except OSError:
                pass

    def _write(self, raw: bytes) -> None:
        text = scrub_string(raw.decode("utf-8", errors="replace"))
        with self._lock:
            try:
                self._fh.write(text)
                self._fh.flush()
            except (OSError, ValueError):
                pass  # logging must never kill the supervised child


class Supervisor:
    def __init__(
        self,
        backend: Backend,
        entry: ModelEntry,
        config: SparkConfig,
        profile: HostProfile,
        paths: SparkPaths,
        secrets: SecretStore,
        *,
        force: bool = False,
    ) -> None:
        self.backend = backend
        self.entry = entry
        self.config = config
        self.profile = profile
        self.paths = paths
        self.secrets = secrets
        self.force = force
        self.log = get_telemetry()
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._peak_rss = 0
        self._publisher: InstancePublisher | None = None
        self._prev_signals: dict[int, object] = {}
        # Child stdout/stderr are redirected here (not the terminal) to keep the
        # CLI output clean; tailed on failure so the reason still surfaces.
        self._log_path = paths.log_dir / "servers" / f"{id_file_stem(entry.id)}.log"
        self._log_fh = None
        self._log_pump: _ScrubbingLogPump | None = None
        self._signal_note_logged = False
        self._monitor_ticks = 0

    @property
    def log_path(self) -> Path:
        return self._log_path

    def log_tail(self, n: int = 30) -> str:
        """Last ``n`` lines of the child server log (empty if none)."""
        try:
            text = "\n".join(self._log_path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return ""
        # Defense in depth: a log written by an older spark must also be safe to
        # render to the operator's terminal.
        return scrub_string(text)

    # -- preflight -------------------------------------------------------------
    def _preflight(self) -> int:
        host = self.config.general.host

        # Weights: are they on this host at all? Repeated here (not just in
        # `spark list`) because a launch is where absent weights turn into a
        # multi-GB fetch *after* the health gate has already passed.
        avail = resolve_availability(
            self.entry,
            self.paths,
            requires_weights=self.backend.requires_local_weights,
        )
        self.log.info(
            "supervisor", "weights_check",
            model=self.entry.id, state=avail.state,
            detail=avail.detail, location=avail.location,
        )

        # Local model path must exist (HF repo ids are allowed to be remote).
        ref = self.backend.resolve_model_ref(self.entry)
        if ("/" in ref and Path(ref).expanduser().is_absolute()) or ref.startswith("."):
            if not Path(ref).expanduser().exists():
                raise ModelNotFoundError(
                    f"Model path does not exist: {ref}",
                    remediation=[
                        "Re-download: spark download <hf-repo>",
                        f"Check the registry entry for '{self.entry.id}'.",
                        f"Or drop the entry: spark forget {self.entry.id}",
                    ],
                    context={"path": ref},
                )

        if avail.state == MISSING and not self.force:
            # Refuse rather than let the runtime fetch it invisibly: the fetch
            # happens on the first request, long after spark has declared the
            # server ready, so a failed/slow download surfaces as a server that
            # answers /v1/models and /health and then never produces a token.
            raise WeightsMissingError(
                f"'{self.entry.id}' has no weights on this host ({avail.detail}).",
                remediation=[
                    f"Fetch + register them: spark download {self.entry.hf_repo or '<hf-repo>'}",
                    "Then launch again — this also records the size for the disk gate.",
                    f"Or drop the entry: spark forget {self.entry.id}",
                    "Override the refusal (let the runtime fetch at first request): --force"
                    "  (the disk floor still applies)",
                ],
                context={
                    "model": self.entry.id, "state": avail.state,
                    "detail": avail.detail, "repo": self.entry.hf_repo,
                },
            )

        # Already serving? A second server for the same alias doubles resident
        # memory and clobbers the first one's published record (records are
        # keyed by alias), so a consumer loses a live endpoint when either exits.
        live = [
            r for r in read_instances(self.paths)
            if r.get("model_alias") == self.entry.id and r.get("pid") != os.getpid()
        ]
        if live and not self.force:
            raise LaunchError(
                f"'{self.entry.id}' is already being served at {live[0].get('base_url')}.",
                code="RUN_ALREADY_SERVING",
                remediation=[
                    f"Use that endpoint (spark pid {live[0].get('pid')}), or stop it first.",
                    "Run a second copy anyway (doubles memory): --force",
                ],
                context={"model": self.entry.id, "pid": live[0].get("pid")},
            )

        # Memory budget. Disk-first launches synthesize their entry from a
        # filename (quant/params) and the availability check (size), so hydrate
        # those fields here rather than letting check_memory see "unknown" and
        # silently return ok=True for a 70B model.
        budget_entry = self.entry
        inferred_quant, inferred_params = infer_quant_and_params(
            self.entry.hf_repo or self.entry.path or self.entry.id
        )
        updates: dict[str, object] = {}
        if not self.entry.quant and inferred_quant:
            updates["quant"] = inferred_quant
        if self.entry.params_billions is None and inferred_params is not None:
            updates["params_billions"] = inferred_params
        if not self.entry.size_bytes and avail.bytes_present > 1:
            updates["size_bytes"] = avail.bytes_present
        if updates:
            budget_entry = self.entry.model_copy(update=updates)
            self.log.info(
                "supervisor", "budget_metadata_hydrated",
                model=self.entry.id, fields=sorted(updates),
            )

        mem = check_memory(budget_entry, self.profile, self.config.memory)
        self.log.info("supervisor", "memory_check", ok=mem.ok, detail=mem.detail)
        if not mem.ok and self.config.memory.enforce and not self.force:
            raise InsufficientMemoryError(
                f"Model likely exceeds the memory budget ({mem.detail}).",
                remediation=[
                    "Use a smaller quant (e.g. q4) or a smaller model.",
                    "Override with --force to launch anyway (risk of OOM/swap).",
                    "Raise memory.usable_fraction in config if you accept the risk.",
                ],
                context={"detail": mem.detail},
            )

        # Disk headroom. Enforced whenever this launch can WRITE: absent weights
        # mean the runtime may fetch them, so the policy's floor is a hard gate
        # (a 27B fetch into ~4 GiB free filled the volume and killed the server
        # mid-download). With weights local nothing new is written, so the same
        # reading is advisory — being low on disk should not block a run.
        # Measured on the volume the bytes would land on: the store when a fetch
        # is possible, the data dir otherwise.
        need_bytes = (budget_entry.size_bytes or 0) if avail.state == MISSING else 0
        disk_target = (
            self.paths.model_store_dir if avail.state == MISSING else self.paths.data_dir
        )
        disk = check_disk(disk_target, need_bytes, self.config.disk)
        self.log.info(
            "supervisor", "disk_check", ok=disk.ok, detail=disk.detail,
            will_fetch=avail.state == MISSING, need_bytes=need_bytes,
        )
        if not disk.ok:
            if avail.state == MISSING and self.config.disk.enforce:
                # Deliberately NOT overridable with --force: --force is for risk
                # preferences (memory headroom), not for a volume that is out of
                # space — refusing here is the whole point. The deliberate way to
                # fetch onto a tight disk is `spark download --force`, which names
                # the intent at the moment of the download.
                raise InsufficientDiskError(
                    f"Launch would fetch weights onto a nearly-full disk ({disk.detail}).",
                    remediation=self._free_space_steps(),
                    context={"detail": disk.detail, "model": self.entry.id},
                )
            self.log.warn(
                "supervisor", "disk_low", detail=disk.detail,
                model=self.entry.id, enforced=False,
            )

        # Port. Daemon-style (attach) backends live on a fixed port; do not
        # reassign it just because the daemon is already listening there.
        if self.backend.attach_if_running:
            return self.backend.default_port
        lo, hi = self.config.general.port_range
        port = _pick_port(host, self.backend.default_port, lo, hi)
        if port != self.backend.default_port:
            # Consumers pin ports, so drift breaks them silently — say it out
            # loud, and publish the endpoint (run/instances/<model>.json) so a
            # consumer that reads it does not have to care.
            self.log.warn(
                "supervisor", "port_drift",
                preferred=self.backend.default_port, chosen=port, model=self.entry.id,
            )
        return port

    def _free_space_steps(self) -> list[str]:
        """How to make room, concretely — the failure this repo keeps hitting."""
        from ..budget import free_disk_bytes

        free_gib = free_disk_bytes(self.paths.data_dir) / 2**30
        return [
            f"Free space: need {self.config.disk.min_free_gib:g} GiB free, have {free_gib:.1f} GiB.",
            "Large regenerable caches here: Model caches under ~/.cache/huggingface,",
            "  ~/.cache/uv, and build trees (cargo target/).",
            "Then re-run — or `spark forget <model>` to drop entries you no longer want.",
        ]


    # -- env -------------------------------------------------------------------
    def _child_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(self.backend.extra_env())
        # Secrets: resolved from the vault, injected here only. Values are
        # registered with the telemetry redactor inside the store.
        secret_env = self.secrets.resolve_env(self.backend.secret_env_map())
        env.update(secret_env)
        self.log.info(
            "supervisor", "secret_env_resolved",
            # Log only which env vars were set, never values.
            vars_set=sorted(secret_env.keys()),
        )
        return env

    # -- health ----------------------------------------------------------------
    def _is_healthy(self, url: str) -> bool:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return 200 <= resp.status < 300
        except urllib.error.HTTPError as e:
            # Some servers return 4xx on the probe path before/after load; treat
            # any HTTP response as "socket is up" only if 2xx.
            return 200 <= e.code < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def _wait_ready(self, url: str, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._stop.is_set():
                return
            if self._proc and self._proc.poll() is not None:
                raise LaunchError(
                    f"Server exited during startup (code {self._proc.returncode}).",
                    remediation=[
                        f"See the server log: {self._log_path}",
                        "Run `spark doctor` to confirm the runtime is healthy.",
                    ],
                    context={"returncode": self._proc.returncode},
                )
            if self._is_healthy(url):
                return
            time.sleep(self.config.supervisor.health_interval_s)
        raise HealthCheckError(
            f"Server did not become healthy within {timeout_s:.0f}s.",
            remediation=[
                "Increase the runtime's ready_timeout_s in config.",
                "Large models load slowly on first run; retry once.",
                f"Probe manually: curl -s {url}",
            ],
            context={"url": url, "timeout_s": timeout_s},
        )

    # -- readiness --------------------------------------------------------------
    def _verify_generation(self, host: str, port: int) -> None:
        """Confirm the server can actually produce a token before calling it ready.

        Lazy runtimes (mlx_lm/mlx_vlm) bring up their HTTP surface before the
        model is resident, and their health path answers from an HF-cache scan —
        so a passing probe says nothing about residency. Without this check a
        server whose model failed to load (or is quietly fetching gigabytes) is
        reported ready indefinitely: the process never exits, so the
        crash-restart path never fires, and every request hangs.
        """
        if not self.config.supervisor.verify_generation:
            return
        if not self.backend.lazy_loads_model or self.backend.attach_if_running:
            return  # eager loader, or a shared daemon whose roster is not ours

        timeout = self.backend.ready_timeout_s()
        ok, detail = self.backend.warmup(self.entry, host, port, timeout)
        self.log.info(
            "supervisor", "warmup", ok=ok, detail=detail,
            model=self.entry.id, timeout_s=timeout,
        )
        if not ok:
            raise LaunchError(
                f"'{self.entry.id}' answered health but produced no tokens ({detail}).",
                remediation=[
                    f"See the server log: {self._log_path}",
                    "A download failure in that log means disk space, not spark — "
                    "free room, then fetch deliberately with `spark download <hf-repo>`.",
                    "Skip this gate with supervisor.verify_generation=false.",
                ],
                context={"detail": detail, "model": self.entry.id},
            )

    # -- monitor ---------------------------------------------------------------
    def _monitor(self, url: str) -> str:
        """Block until the process exits or a stop is requested. Returns reason."""
        interval = self.config.supervisor.health_interval_s
        self._note_stop_signal()
        while not self._stop.is_set():
            assert self._proc is not None
            code = self._proc.poll()
            if code is not None:
                return "exited"
            # Peak RSS is a log field, not a gate: sampling `ps` on every
            # health tick cost ~12 fork+exec/minute for the server's lifetime.
            self._monitor_ticks += 1
            if self._monitor_ticks % 6 == 0:
                rss = _rss_bytes(self._proc.pid)
                if rss:
                    self._peak_rss = max(self._peak_rss, rss)
            # Keep the published instance record fresh: a consumer uses
            # updated_at to tell a live endpoint from a stale file left by a
            # hard kill (SIGKILL, power loss), which is the one case where
            # removal on exit never happens.
            if self._publisher is not None:
                try:
                    self._publisher.maybe_heartbeat()
                except OSError as exc:
                    self.log.warn("supervisor", "instance_heartbeat_failed", error=repr(exc))
            self._stop.wait(interval)
        return "stop_requested"

    def _schedule_restart(
        self, attempt: int, backoff: float, policy, *, reason: str
    ) -> tuple[int, float] | None:
        """One bounded restart step. Returns the next (attempt, backoff), or
        ``None`` when the incident budget is exhausted or a stop was requested.

        Startup failures and mid-run crashes must share this logic — two copies
        meant a fix to backoff/remediation applied to only one of them.
        """
        if attempt >= policy.max_restarts or self._stop.is_set():
            return None
        self.log.warn(
            "supervisor", "restart_scheduled",
            attempt=attempt + 1, backoff_s=backoff, reason=reason,
        )
        self._stop.wait(backoff)
        return attempt + 1, min(backoff * 2, policy.backoff_max_s)

    # -- lifecycle -------------------------------------------------------------
    def _spawn(self, cmd: list[str], env: dict[str, str]) -> None:
        self.log.info(
            "supervisor", "process_start",
            exe=cmd[0], argc=len(cmd), backend=self.backend.name,
            model=self.entry.id, cwd=str(Path.cwd()),
        )
        if self._log_fh is None:
            self._log_fh = self._open_server_log()
        assert self._log_pump is not None
        # Child stdout+stderr go through a scrubbing pipe, not straight to a file
        # descriptor: a runtime that echoes its HF_TOKEN must not put it on disk.
        self._log_fh.write(scrub_string(
            f"\n=== spawn {time.strftime('%Y-%m-%dT%H:%M:%S')} :: {' '.join(cmd)} ===\n"
        ))
        self._log_fh.flush()
        write_fd = self._log_pump.start()
        try:
            self._proc = subprocess.Popen(
                cmd, env=env, stdout=write_fd, stderr=write_fd
            )
        except FileNotFoundError as exc:
            raise LaunchError(
                f"Runtime binary not found: {cmd[0]}",
                remediation=[
                    f"Install it: {self.backend.rt.install_hint or 'see `spark doctor`'}",
                    "Uninstalled since the last probe? Re-detect: spark run <model> --reprobe",
                ],
                context={"binary": cmd[0]},
                cause=exc,
            ) from exc
        finally:
            # The child owns a dup of this fd; the parent's copy must close or
            # the pump never sees EOF and the log stays open indefinitely.
            try:
                os.close(write_fd)
            except OSError:
                pass

    #: Roll the server log once it passes this size (one previous copy kept).
    LOG_ROLL_BYTES = 64 * 2**20

    def _open_server_log(self):
        """Opened at first spawn — after preflight — and in append mode.

        Opening it ``"w"`` up front meant a launch refused in preflight (e.g. the
        model is already being served) truncated the *live* server's log while
        its child kept writing at the old offset: a NUL-padded, sparse file.
        """
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if self._log_path.stat().st_size > self.LOG_ROLL_BYTES:
                self._log_path.replace(self._log_path.with_suffix(".log.1"))
        except OSError:
            pass  # absent (first run) or unrollable: append regardless
        fd = os.open(self._log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.fchmod(fd, 0o600)  # tighten a log created by an older spark
            fh = os.fdopen(fd, "a", buffering=1)
        except BaseException:
            os.close(fd)
            raise
        self._log_pump = _ScrubbingLogPump(fh)
        return fh

    def _terminate(self) -> None:
        if not self._proc or self._proc.poll() is not None:
            return
        grace = self.config.supervisor.shutdown_grace_s
        self.log.info("supervisor", "shutdown_initiated", child_pid=self._proc.pid, grace_s=grace)
        try:
            self._proc.terminate()
            self._proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            self.log.warn("supervisor", "shutdown_force_kill", child_pid=self._proc.pid)
            self._proc.kill()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def _note_stop_signal(self) -> None:
        """Log the stop once, off the signal handler's stack.

        Python runs signal handlers on the main thread. Logging inside one can
        re-enter the telemetry lock while the main thread already holds it in
        ``_write`` — a self-deadlock. The event is still required by the logging
        contract, so the monitor loop emits it after observing the stop event.
        """
        if self._stop.is_set() and not self._signal_note_logged:
            self._signal_note_logged = True
            self.log.info("supervisor", "signal_received", signal="stop")

    def _install_signals(self) -> None:
        def handler(signum, _frame):
            # Do the absolute minimum here: set an Event (async-signal-safe
            # enough for this CLI). No logging, no file IO, no locks.
            self._stop.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._prev_signals[sig] = signal.signal(sig, handler)
            except ValueError:  # not the main thread (tests, embedding)
                pass

    def _restore_signals(self) -> None:
        for sig, prev in self._prev_signals.items():
            signal.signal(sig, prev)
        self._prev_signals.clear()

    def _monitor_attached(self, url: str) -> str:
        """Watch an attached (not-owned) endpoint until stop requested."""
        self._note_stop_signal()
        while not self._stop.wait(self.config.supervisor.health_interval_s):
            pass
        return "stop_requested"

    # -- public ----------------------------------------------------------------
    def run(self, on_ready: Callable[[ReadyInfo], None] | None = None) -> LaunchResult:
        try:
            return self._run(on_ready)
        finally:
            # Any exit path — including an exception spark did not anticipate —
            # must not orphan a multi-GB server holding the port and the RAM.
            self._terminate()
            self._restore_signals()
            # Whatever happened — clean stop, crash, exhausted recovery, or an
            # error before launch — a published instance must not outlive the
            # server it describes, and the catalog must reflect the exit.
            if self._publisher is not None:
                self._publisher.remove()
                self._publisher = None
            safe_write_catalog(
                self.paths, self.config,
                component="supervisor", event="catalog_write_failed",
            )
            if self._log_pump is not None:
                self._log_pump.join()
                self._log_pump = None
            if self._log_fh:
                self._log_fh.close()
                self._log_fh = None

    def _announce_ready(
        self, on_ready, base: str, port: int, since: float
    ) -> None:
        preferred = self.backend.default_port
        note = ""
        if port != preferred:
            note = f"{preferred} was busy; consumers pinning it must follow this endpoint"
        if on_ready is not None:
            on_ready(ReadyInfo(
                base_url=base, backend=self.backend.name, port=port,
                elapsed_s=time.monotonic() - since, log_path=str(self._log_path),
                note=note,
                api_model_id=self.backend.api_model_id(self.entry),
            ))

    def _publish_instance(self, base: str, port: int, health_url: str) -> None:
        """Publish the live-instance manifest consumers read to find this endpoint.

        Publishing is a courtesy, never a gate: a failure to write it must not
        take down a server that is otherwise serving.
        """
        self._publisher = InstancePublisher(
            self.paths,
            alias=self.entry.id,
            api_model_id=self.backend.api_model_id(self.entry),
            backend=self.backend.name,
            base_url=base,
            port=port,
            health_url=health_url,
            session_id=str(getattr(self.log, "session_id", "unbound")),
        )
        try:
            path = self._publisher.publish()
        except OSError as exc:
            self.log.warn(
                "supervisor", "instance_publish_failed",
                model=self.entry.id, error=repr(exc),
            )
            self._publisher = None
            return
        self.log.info(
            "supervisor", "instance_published",
            model=self.entry.id, port=port, path=str(path),
            api_model_id=self.backend.api_model_id(self.entry),
        )
        # The roster promises "refreshed on server ready"; without this a
        # consumer reading catalog.json sees no live instance until exit.
        safe_write_catalog(
            self.paths, self.config,
            component="supervisor", event="catalog_write_failed",
        )

    def _run(self, on_ready: Callable[[ReadyInfo], None] | None) -> LaunchResult:
        t0 = time.monotonic()
        port = self._preflight()
        host = self.config.general.host
        url = self.backend.health_url(host, port)
        base = self.backend.openai_base_url(host, port)
        self._install_signals()

        # Pre-launch hook (pull weights, verify, etc.); may abort with remediation.
        self.backend.prepare(self.entry, secrets=self.secrets)

        # Attach mode: if a daemon-style backend is already serving, don't spawn.
        if self.backend.attach_if_running and self._is_healthy(url):
            self.log.info(
                "supervisor", "attached", endpoint=base, port=port,
                backend=self.backend.name, model=self.entry.id,
            )
            self._announce_ready(on_ready, base, port, t0)
            # An attached daemon is still a live server: consumers read the
            # instance directory, so it must be published like a spawned one.
            self._publish_instance(base, port, url)
            self._monitor_attached(url)
            self.log.info("supervisor", "detached", port=port, model=self.entry.id)
            return LaunchResult(base, self.backend.name, port, 0)

        cmd = self.backend.build_launch_cmd(self.entry, host, port)
        if not cmd:
            # Attach-only backends (relay) have nothing to spawn: an unhealthy
            # endpoint is terminal here, not a crash inside Popen([]).
            raise LaunchError(
                f"'{self.backend.name}' endpoint is not healthy and spark has nothing to spawn.",
                remediation=[
                    f"Start the external service, then retry. Probe: curl -s {url}",
                    "Check the runtime's relay_base_url in your spark config.",
                ],
                context={"url": url, "backend": self.backend.name},
            )
        env = self._child_env()

        policy = self.config.supervisor
        max_restarts = policy.max_restarts
        backoff = policy.backoff_base_s
        started_at = time.time()
        announced = False
        attempt = 0   # position in the current restart budget
        restarts = 0  # total respawns this session (what gets reported)
        first_spawn = True

        while attempt <= max_restarts:
            if self._stop.is_set():
                break
            if not first_spawn:
                restarts += 1
            first_spawn = False
            self._spawn(cmd, env)
            try:
                self._wait_ready(url, self.backend.ready_timeout_s())
                if self._stop.is_set():
                    break  # Ctrl+C during startup: never announce a server we're stopping
                self._verify_generation(host, port)
            except (LaunchError, HealthCheckError) as exc:
                self._terminate()
                scheduled = self._schedule_restart(
                    attempt, backoff, policy, reason=exc.code
                )
                if scheduled is not None:
                    attempt, backoff = scheduled
                    continue
                raise RecoveryExhaustedError(
                    f"'{self.entry.id}' failed to start after {attempt + 1} attempt(s).",
                    remediation=exc.remediation
                    or ["Run `spark doctor`; inspect logs in the data dir."],
                    context={"last_error": exc.code, "attempts": attempt + 1},
                    cause=exc,
                ) from exc

            self.log.info(
                "supervisor", "server_ready",
                endpoint=url, backend=self.backend.name, port=port,
                attempt=attempt + 1,
            )
            if not announced:
                self._announce_ready(on_ready, base, port, t0)
                self._publish_instance(base, port, url)
                announced = True
            ready_at = time.monotonic()
            reason = self._monitor(url)

            if reason == "stop_requested":
                self._terminate()
                self._emit_exit(started_at, port, restarts, graceful=True)
                return LaunchResult(base, self.backend.name, port, restarts)

            # Unexpected exit.
            code = self._proc.returncode if self._proc else None
            uptime = time.monotonic() - ready_at
            self.log.error(
                "supervisor", "process_crash",
                returncode=code, attempt=attempt + 1, uptime_s=round(uptime, 1),
                context_window=self.log.context_window()[-10:],
            )
            if uptime >= policy.stable_after_s:
                # A crash after a long healthy run is a new incident, not the
                # same one recurring: without this a server that crashes once a
                # day exhausts the budget for good by day four.
                self.log.info("supervisor", "restart_budget_reset", uptime_s=round(uptime, 1))
                attempt, backoff = 0, policy.backoff_base_s
            scheduled = self._schedule_restart(
                attempt, backoff, policy, reason="crash"
            )
            if scheduled is not None:
                attempt, backoff = scheduled
                continue
            self._emit_exit(started_at, port, restarts, graceful=False)
            raise RecoveryExhaustedError(
                f"'{self.entry.id}' crashed and exhausted {max_restarts} restart(s).",
                remediation=[
                    f"See the server log: {self._log_path}",
                    "Try `--force` off / a smaller quant if this is an OOM.",
                    "Run `spark doctor` to verify the runtime.",
                ],
                context={"returncode": code, "attempts": attempt + 1},
            )

        # Stop requested before/at start (or between restarts).
        self._terminate()
        self._emit_exit(started_at, port, restarts, graceful=True)
        return LaunchResult(base, self.backend.name, port, restarts)

    def _emit_exit(self, started_at: float, port: int, restarts: int, *, graceful: bool) -> None:
        self.log.info(
            "supervisor", "process_exit",
            graceful=graceful,
            wall_time_s=round(time.time() - started_at, 2),
            peak_rss_bytes=self._peak_rss,
            restarts=restarts, port=port, model=self.entry.id,
        )
