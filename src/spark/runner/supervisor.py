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
from dataclasses import dataclass
from pathlib import Path

from ..budget import check_disk, check_memory
from ..config.paths import SparkPaths
from ..config.schema import ModelEntry, SparkConfig
from ..errors import (
    HealthCheckError,
    InsufficientMemoryError,
    LaunchError,
    ModelNotFoundError,
    PortInUseError,
    RecoveryExhaustedError,
)
from ..probe.host import HostProfile
from ..runtimes.base import Backend
from ..secrets.store import SecretStore
from ..telemetry import get_telemetry


@dataclass
class LaunchResult:
    endpoint: str
    backend: str
    port: int
    restarts: int


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
            f"Free a port or widen general.port_range in your config.",
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

    # -- preflight -------------------------------------------------------------
    def _preflight(self) -> int:
        host = self.config.general.host

        # Local model path must exist (HF repo ids are allowed to be remote).
        ref = self.backend.resolve_model_ref(self.entry)
        if ("/" in ref and Path(ref).expanduser().is_absolute()) or ref.startswith("."):
            if not Path(ref).expanduser().exists():
                raise ModelNotFoundError(
                    f"Model path does not exist: {ref}",
                    remediation=[
                        "Re-download: spark download <hf-repo>",
                        f"Check the registry entry for '{self.entry.id}'.",
                    ],
                    context={"path": ref},
                )

        # Memory budget.
        mem = check_memory(self.entry, self.profile, self.config.memory)
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

        # Disk headroom (weights already on disk; this guards swap/scratch space).
        disk = check_disk(self.paths.data_dir, 0, self.config.disk)
        self.log.info("supervisor", "disk_check", ok=disk.ok, detail=disk.detail)

        # Port. Daemon-style (attach) backends live on a fixed port; do not
        # reassign it just because the daemon is already listening there.
        if self.backend.attach_if_running:
            return self.backend.default_port
        lo, hi = self.config.general.port_range
        port = _pick_port(host, self.backend.default_port, lo, hi)
        return port

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
                        "Check the server output above for the failure reason.",
                        "Run `spark doctor` to confirm the runtime is healthy.",
                    ],
                    context={"returncode": self._proc.returncode},
                )
            if self._is_healthy(url):
                return
            time.sleep(1.0)
        raise HealthCheckError(
            f"Server did not become healthy within {timeout_s:.0f}s.",
            remediation=[
                "Increase the runtime's ready_timeout_s in config.",
                "Large models load slowly on first run; retry once.",
                f"Probe manually: curl -s {url}",
            ],
            context={"url": url, "timeout_s": timeout_s},
        )

    # -- monitor ---------------------------------------------------------------
    def _monitor(self, url: str) -> str:
        """Block until the process exits or a stop is requested. Returns reason."""
        interval = self.config.supervisor.health_interval_s
        while not self._stop.is_set():
            assert self._proc is not None
            code = self._proc.poll()
            if code is not None:
                return "exited"
            rss = _rss_bytes(self._proc.pid)
            if rss:
                self._peak_rss = max(self._peak_rss, rss)
            time.sleep(interval)
        return "stop_requested"

    # -- lifecycle -------------------------------------------------------------
    def _spawn(self, cmd: list[str], env: dict[str, str]) -> None:
        self.log.info(
            "supervisor", "process_start",
            exe=cmd[0], argc=len(cmd), backend=self.backend.name,
            model=self.entry.id, cwd=str(Path.cwd()),
        )
        try:
            # Child inherits stdout/stderr so the operator sees server logs live.
            self._proc = subprocess.Popen(cmd, env=env)
        except FileNotFoundError as exc:
            raise LaunchError(
                f"Runtime binary not found: {cmd[0]}",
                remediation=[
                    f"Install it: {self.backend.rt.install_hint or 'see `spark doctor`'}",
                ],
                context={"binary": cmd[0]},
                cause=exc,
            ) from exc

    def _terminate(self) -> None:
        if not self._proc or self._proc.poll() is not None:
            return
        grace = self.config.supervisor.shutdown_grace_s
        self.log.info("supervisor", "shutdown_initiated", pid=self._proc.pid, grace_s=grace)
        try:
            self._proc.terminate()
            self._proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            self.log.warn("supervisor", "shutdown_force_kill", pid=self._proc.pid)
            self._proc.kill()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def _install_signals(self) -> None:
        def handler(signum, _frame):
            self.log.info("supervisor", "signal_received", signal=signum)
            self._stop.set()

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _monitor_attached(self, url: str) -> str:
        """Watch an attached (not-owned) endpoint until stop requested."""
        interval = self.config.supervisor.health_interval_s
        while not self._stop.is_set():
            time.sleep(interval)
        return "stop_requested"

    # -- public ----------------------------------------------------------------
    def run(self) -> LaunchResult:
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
            self._monitor_attached(url)
            self.log.info("supervisor", "detached", port=port, model=self.entry.id)
            return LaunchResult(base, self.backend.name, port, 0)

        cmd = self.backend.build_launch_cmd(self.entry, host, port)
        env = self._child_env()

        max_restarts = self.config.supervisor.max_restarts
        backoff = self.config.supervisor.backoff_base_s
        started_at = time.time()

        for attempt in range(max_restarts + 1):
            if self._stop.is_set():
                break
            self._spawn(cmd, env)
            try:
                self._wait_ready(url, self.backend.ready_timeout_s())
            except (LaunchError, HealthCheckError) as exc:
                self._terminate()
                if attempt < max_restarts and not self._stop.is_set():
                    self.log.warn(
                        "supervisor", "restart_scheduled",
                        attempt=attempt + 1, backoff_s=backoff, reason=exc.code,
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, self.config.supervisor.backoff_max_s)
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
            reason = self._monitor(url)

            if reason == "stop_requested":
                self._terminate()
                self._emit_exit(started_at, port, attempt, graceful=True)
                return LaunchResult(base, self.backend.name, port, attempt)

            # Unexpected exit.
            code = self._proc.returncode if self._proc else None
            self.log.error(
                "supervisor", "process_crash",
                returncode=code, attempt=attempt + 1,
                context_window=self.log.context_window()[-10:],
            )
            if attempt < max_restarts and not self._stop.is_set():
                self.log.warn(
                    "supervisor", "restart_scheduled",
                    attempt=attempt + 1, backoff_s=backoff, reason="crash",
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, self.config.supervisor.backoff_max_s)
                continue
            self._emit_exit(started_at, port, attempt, graceful=False)
            raise RecoveryExhaustedError(
                f"'{self.entry.id}' crashed and exhausted {max_restarts} restart(s).",
                remediation=[
                    "Inspect the server output above and the JSONL logs.",
                    "Try `--force` off / a smaller quant if this is an OOM.",
                    "Run `spark doctor` to verify the runtime.",
                ],
                context={"returncode": code, "attempts": attempt + 1},
            )

        # Stop requested before/at start.
        self._terminate()
        self._emit_exit(started_at, port, 0, graceful=True)
        return LaunchResult(base, self.backend.name, port, 0)

    def _emit_exit(self, started_at: float, port: int, restarts: int, *, graceful: bool) -> None:
        self.log.info(
            "supervisor", "process_exit",
            graceful=graceful,
            wall_time_s=round(time.time() - started_at, 2),
            peak_rss_bytes=self._peak_rss,
            restarts=restarts, port=port, model=self.entry.id,
        )
