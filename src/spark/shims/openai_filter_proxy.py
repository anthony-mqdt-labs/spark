"""Filtering OpenAI-compatible proxy: one port, spark's view of the models.

Some runtimes answer ``GET /v1/models`` from their own scan of the hub cache
(``mlx_lm.server`` lists every MLX-shaped repo: embedders, TTS, STT, partials).
Harness pickers read that endpoint and offer models that cannot serve chat.
This proxy binds the published port, spawns the real server on an ephemeral
loopback port as its child, and passes everything through untouched — except
``GET /v1/models``, which it answers locally from disk truth:

* first: the served model id (``--served-id``),
* then: every other model this server could serve, i.e. registered entries
  with weights on disk whose format is mlx-compatible, as the API id a client
  must send (hub repo id, or store path — the same rule the launch path uses).

Anything else cached (embeddings, speech, partials, unregistered weights,
foreign formats) is dropped from the listing. Generation is never blocked:
a direct request for an unlisted id still proxies through (the backend
lazy-loads it or fails on its own terms).

Lifecycle (single child from the supervisor's view):
* the proxy waits for the backend's health gate before reporting ready itself
  (until then its ``/v1/models`` is 503, so the supervisor keeps waiting);
* a monitor thread exits the proxy if the backend dies or stays unhealthy, so
  the supervisor's restart path fires for the unit as a whole;
* SIGTERM/SIGINT terminates the backend child first, then exits.

Runs under spark's own interpreter (needs ``spark.inventory``; imports no ML
libraries). Stdlib only.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Formats mlx_lm.server can plausibly serve (mirrors backend selection).
SERVABLE_FORMATS = ("mlx", "any", "")

#: Request bodies are chat payloads; anything near this is a misbehaving client.
MAX_BODY_BYTES = 64 * 2**20

#: A model roster changes only when the registry or disk changes. Harness pickers
#: poll ``/v1/models`` far more often than that, so a computed roster stays valid
#: for this long before the disk is scanned again.
ALLOWED_IDS_TTL_S = 30.0

_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Filter /v1/models to spark's roster.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True, help="Published port.")
    p.add_argument(
        "--upstream", required=True, help="Backend base URL, e.g. http://127.0.0.1:8099."
    )
    p.add_argument("--served-id", default="", help="API id of the launched model (listed first).")
    p.add_argument("--health-path", default="/v1/models")
    p.add_argument("--ready-timeout-s", type=float, default=240.0)
    p.add_argument(
        "--unhealthy-exit-after",
        type=int,
        default=3,
        help="Monitor exits after this many consecutive failed checks.",
    )
    p.add_argument(
        "--monitor-interval-s",
        type=float,
        default=5.0,
        help="Seconds between backend health checks.",
    )
    p.add_argument("backend", nargs=argparse.REMAINDER, help="Backend argv after `--`.")
    args = p.parse_args(argv)
    if args.backend and args.backend[0] == "--":
        args.backend = args.backend[1:]
    if not args.backend:
        p.error("backend argv is required after `--`")
    return args


def allowed_ids(served_id: str) -> list[str]:
    """Served id first, then every other registered+present mlx-servable id."""
    try:
        from spark.config.paths import resolve_paths
        from spark.inventory import build_inventory
    except ImportError:
        return [served_id] if served_id else []
    try:
        paths = resolve_paths()
        inv = build_inventory(paths, config=None, with_bytes=False)
    except Exception:  # noqa: BLE001 — listing must degrade to served-only, never 500
        return [served_id] if served_id else []
    ids = [served_id] if served_id else []
    for entry, _avail in inv.runnable_registered:
        if entry.model_format not in SERVABLE_FORMATS:
            continue
        api_id = entry.path or entry.hf_repo or entry.id
        if api_id and api_id not in ids:
            ids.append(api_id)
    return ids


#: One cached roster per process, primed when the backend reports ready and
#: rebuilt only on TTL expiry. ``ids=None`` means "never computed".
_MODELS_CACHE: dict = {"ids": None, "built_at": 0.0}
_MODELS_LOCK = threading.RLock()


def _refresh_allowed_ids(served_id: str | None = None) -> list[str]:
    """Recompute and cache the roster — the single disk-scanning seam.

    Called once at startup (see :func:`main`) and again after the TTL expires.
    A lock makes concurrent handler threads share one rebuild rather than each
    independently walking the registry/weights on the same poll.
    """
    with _MODELS_LOCK:
        sid = _STATE.get("served_id", "") if served_id is None else served_id
        ids = allowed_ids(sid)
        _MODELS_CACHE["ids"] = ids
        _MODELS_CACHE["built_at"] = time.monotonic()
        return ids


def _cached_ids() -> list[str]:
    """The cached roster, refreshing it at most once per TTL window."""
    ids = _MODELS_CACHE["ids"]
    if ids is None or time.monotonic() - _MODELS_CACHE["built_at"] >= ALLOWED_IDS_TTL_S:
        with _MODELS_LOCK:
            ids = _MODELS_CACHE["ids"]
            if ids is None or time.monotonic() - _MODELS_CACHE["built_at"] >= ALLOWED_IDS_TTL_S:
                ids = _refresh_allowed_ids()
    return ids


def models_payload() -> dict:
    return {
        "object": "list",
        "data": [{"id": i, "object": "model"} for i in _cached_ids()],
    }


_STATE: dict = {}


def _upstream_host(parts: urllib.parse.ParseResult) -> str:
    """The upstream hostname, or a loud, actionable error if the URL has none."""
    host = parts.hostname
    if not host:
        raise ValueError(
            f"upstream URL {parts.geturl()!r} has no host; "
            "pass --upstream http://127.0.0.1:<port> (the backend's bind address)."
        )
    return host


def _upstream(parts: urllib.parse.ParseResult):
    if parts.scheme != "http":
        raise ValueError(f"only http upstreams supported: {parts.geturl()}")
    host = _upstream_host(parts)
    return http.client.HTTPConnection(host, parts.port or 80, timeout=None)


def _is_models_request(handler: BaseHTTPRequestHandler) -> bool:
    path = urllib.parse.urlsplit(handler.path).path.rstrip("/") or "/"
    return handler.command == "GET" and path == "/v1/models"


class _Handler(BaseHTTPRequestHandler):
    server_version = "spark-filter-proxy/1"

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        sys.stderr.write(f"[proxy] {self.address_string()} {format % args}\n")

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle(self) -> None:
        is_models = _is_models_request(self)
        if is_models and _STATE.get("upstream_ready"):
            self._send_json(200, models_payload())
            return
        if is_models:
            self._send_json(503, {"error": "backend not ready"})
            return
        # Passthrough: same method/path/body, hop headers stripped.
        raw_len = self.headers.get("Content-Length") or "0"
        if not raw_len.isdigit() or int(raw_len) > MAX_BODY_BYTES:
            self._send_json(400, {"error": "missing, invalid, or oversized Content-Length"})
            return
        length = int(raw_len)
        if not length and "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            # Forwarding would silently drop the body; refuse loudly instead.
            self._send_json(411, {"error": "chunked request bodies are not supported"})
            return
        body = self.rfile.read(length) if length else None
        fwd = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_HEADERS}
        conn = None
        try:
            conn = _upstream(_STATE["upstream"])
            conn.request(self.command, self.path, body=body, headers=fwd)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            if conn is not None:
                conn.close()
            self._send_json(502, {"error": f"backend unreachable: {exc}"})
            return
        try:
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in _HOP_HEADERS:
                    self.send_header(k, v)
            self.end_headers()
            # Stream, never buffer: `stream: true` completions are SSE, and a
            # buffered proxy turns time-to-first-token into total time. The
            # response is close-delimited (HTTP/1.0 handler) unless upstream
            # sent a Content-Length, which is forwarded as-is.
            while chunk := resp.read1(64 * 1024):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException):
            pass  # client went away, or upstream died mid-body
        finally:
            conn.close()

    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle
    do_DELETE = _handle
    do_PATCH = _handle
    do_OPTIONS = _handle
    do_HEAD = _handle


def _spawn_backend(argv: list[str]) -> "subprocess.Popen[bytes]":
    # Backend output inherits ours: it lands in spark's server log unmodified.
    return subprocess.Popen(argv)


def _adopted_server(fd: int, host: str, port: int) -> ThreadingHTTPServer:
    """Serve on spark's pre-bound listening socket instead of binding our own.

    ``SPARK_LISTEN_FD`` arrives via ``pass_fds`` from the supervisor, which held
    the port from preflight to exec. Adopting that fd means no second bind: a
    same-user process cannot take the port in the gap. Without the env var the
    ordinary bind path below stays unchanged (back-compat).
    """

    class _AdoptedServer(ThreadingHTTPServer):
        def server_bind(self) -> None:
            self.socket = socket.socket(fileno=fd)  # takes over the inherited fd
            self.server_address = (host, port)
            self.server_name = host
            self.server_port = port

    return _AdoptedServer((host, port), _Handler)


def _probe_health(parts: urllib.parse.ParseResult, health_path: str) -> bool:
    """One health poll: drain the body and close both response and connection.

    A long-lived server is polled every few seconds for its whole life; leaking
    one ``HTTPConnection`` per poll accumulates sockets/fds until the process
    runs out. Reading the body also lets the socket close cleanly instead of
    stranding unread bytes on a half-open connection.
    """
    host = _upstream_host(parts)
    with contextlib.closing(http.client.HTTPConnection(host, parts.port or 80, timeout=5)) as conn:
        conn.request("GET", health_path)
        with contextlib.closing(conn.getresponse()) as resp:
            resp.read()
            return 200 <= resp.status < 300


def _wait_ready(upstream: str, health_path: str, timeout_s: float) -> bool:
    parts = urllib.parse.urlparse(upstream)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if _probe_health(parts, health_path):
                return True
        except (OSError, http.client.HTTPException):
            pass
        time.sleep(1.0)
    return False


def _terminate_child(child: "subprocess.Popen[bytes]") -> None:
    """Stop the backend, escalating to kill. The proxy must never exit while
    its child still runs: an orphaned server would hold VRAM/RAM and serve
    stale weights invisibly beside the restarted unit."""
    try:
        child.terminate()
        child.wait(timeout=10)
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        try:
            child.kill()
        except (OSError, subprocess.SubprocessError):
            pass


def _monitor(
    child: "subprocess.Popen[bytes]",
    upstream: str,
    health_path: str,
    max_failures: int,
    stop: threading.Event,
    interval_s: float = 5.0,
) -> None:
    """Exit the proxy when the backend is gone: supervisor restarts the unit."""
    failures = 0
    parts = urllib.parse.urlparse(upstream)
    while not stop.wait(interval_s):
        if child.poll() is not None:
            print(
                f"[proxy] backend exited (code {child.returncode}); exiting",
                file=sys.stderr,
                flush=True,
            )
            os._exit(1)
        try:
            ok = _probe_health(parts, health_path)
        except (OSError, http.client.HTTPException):
            ok = False
        failures = 0 if ok else failures + 1
        if failures >= max_failures:
            print("[proxy] backend unhealthy; exiting", file=sys.stderr, flush=True)
            _terminate_child(child)
            os._exit(1)


def main(argv=None) -> int:
    args = _parse_args(argv)
    _STATE.update(
        upstream=urllib.parse.urlparse(args.upstream),
        served_id=args.served_id,
        upstream_ready=False,
    )
    # Consume the handoff: the fd belongs to this proxy, not to the backend we
    # spawn below (which would otherwise inherit a stale env value).
    listen_fd = os.environ.pop("SPARK_LISTEN_FD", None)
    child = _spawn_backend(args.backend)
    stop = threading.Event()

    def _shutdown(signum, _frame):
        print(f"[proxy] signal {signum}; terminating backend", file=sys.stderr, flush=True)
        stop.set()
        _terminate_child(child)
        raise SystemExit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _shutdown)
        except (ValueError, OSError):
            pass

    # Bind the published port *now*, not after the model loads: the supervisor
    # picked it as free seconds ago, and a load can take minutes. When spark
    # reserved the port it also passed the listening fd; adopt it rather than
    # binding again. Until the backend is healthy, /v1/models answers 503 so the
    # supervisor keeps waiting.
    try:
        if listen_fd is None:
            server = ThreadingHTTPServer((args.host, args.port), _Handler)
        else:
            server = _adopted_server(int(listen_fd), args.host, args.port)
    except (OSError, ValueError) as exc:
        print(
            f"[proxy] FATAL: cannot listen on {args.host}:{args.port}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        _terminate_child(child)
        return 1
    server.daemon_threads = True
    serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
    serve_thread.start()
    print(
        f"[proxy] listening on http://{args.host}:{args.port} "
        f"(upstream {args.upstream}); waiting for backend{args.health_path} …",
        flush=True,
    )
    try:
        if not _wait_ready(args.upstream, args.health_path, args.ready_timeout_s):
            print("[proxy] FATAL: backend never became ready", file=sys.stderr)
            _terminate_child(child)
            return 1
        # Prime the roster while still single-threaded, before the ready flag
        # lets handler threads answer /v1/models.
        _refresh_allowed_ids(args.served_id)
        _STATE["upstream_ready"] = True
        threading.Thread(
            target=_monitor,
            args=(
                child,
                args.upstream,
                args.health_path,
                args.unhealthy_exit_after,
                stop,
                args.monitor_interval_s,
            ),
            daemon=True,
        ).start()
        while not stop.wait(1.0):  # timed: the signal handler stays responsive
            if child.poll() is not None:
                return 1  # monitor normally catches this first
    except KeyboardInterrupt:
        _terminate_child(child)
    finally:
        # Close the adopted fd (or a normal listener) on every exit path: a
        # supervisor restart must be able to take the same port again.
        if serve_thread.is_alive():
            server.shutdown()
        serve_thread.join(timeout=1.0)
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
