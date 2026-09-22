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
import http.client
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Formats mlx_lm.server can plausibly serve (mirrors backend selection).
SERVABLE_FORMATS = ("mlx", "any", "")

_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Filter /v1/models to spark's roster.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True, help="Published port.")
    p.add_argument("--upstream", required=True,
                   help="Backend base URL, e.g. http://127.0.0.1:8099.")
    p.add_argument("--served-id", default="",
                   help="API id of the launched model (listed first).")
    p.add_argument("--health-path", default="/v1/models")
    p.add_argument("--ready-timeout-s", type=float, default=240.0)
    p.add_argument("--unhealthy-exit-after", type=int, default=3,
                   help="Monitor exits after this many consecutive failed checks.")
    p.add_argument("--monitor-interval-s", type=float, default=5.0,
                   help="Seconds between backend health checks.")
    p.add_argument("backend", nargs=argparse.REMAINDER,
                   help="Backend argv after `--`.")
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


def models_payload() -> dict:
    ids = allowed_ids(_STATE.get("served_id", ""))
    return {
        "object": "list",
        "data": [{"id": i, "object": "model"} for i in ids],
    }


_STATE: dict = {}


def _upstream(parts: urllib.parse.ParseResult):
    if parts.scheme != "http":
        raise ValueError(f"only http upstreams supported: {parts.geturl()}")
    return http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=None)


def _is_models_request(handler: BaseHTTPRequestHandler) -> bool:
    path = urllib.parse.urlsplit(handler.path).path.rstrip("/") or "/"
    return handler.command == "GET" and path == "/v1/models"


class _Handler(BaseHTTPRequestHandler):
    server_version = "spark-filter-proxy/1"

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[proxy] {self.address_string()} {fmt % args}\n")

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle(self) -> None:
        if _is_models_request(self) and _STATE.get("upstream_ready"):
            self._send_json(200, models_payload())
            return
        if _is_models_request(self):
            self._send_json(503, {"error": "backend not ready"})
            return
        # Passthrough: same method/path/body, hop headers stripped.
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        fwd = {k: v for k, v in self.headers.items()
               if k.lower() not in _HOP_HEADERS}
        try:
            conn = _upstream(_STATE["upstream"])
            conn.request(self.command, self.path, body=body, headers=fwd)
            resp = conn.getresponse()
            data = resp.read()
        except (OSError, http.client.HTTPException) as exc:
            self._send_json(502, {"error": f"backend unreachable: {exc}"})
            return
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in _HOP_HEADERS:
                self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(data)
        except (OSError, BrokenPipeError):
            pass

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


def _wait_ready(upstream: str, health_path: str, timeout_s: float) -> bool:
    parts = urllib.parse.urlparse(upstream)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection(parts.hostname, parts.port or 80,
                                              timeout=5)
            conn.request("GET", health_path)
            if 200 <= conn.getresponse().status < 300:
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


def _monitor(child: "subprocess.Popen[bytes]", upstream: str, health_path: str,
             max_failures: int, stop: threading.Event,
             interval_s: float = 5.0) -> None:
    """Exit the proxy when the backend is gone: supervisor restarts the unit."""
    failures = 0
    parts = urllib.parse.urlparse(upstream)
    while not stop.wait(interval_s):
        if child.poll() is not None:
            print(f"[proxy] backend exited (code {child.returncode}); exiting",
                  file=sys.stderr, flush=True)
            os._exit(1)
        try:
            conn = http.client.HTTPConnection(parts.hostname, parts.port or 80,
                                              timeout=5)
            conn.request("GET", health_path)
            ok = 200 <= conn.getresponse().status < 300
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
    child = _spawn_backend(args.backend)
    stop = threading.Event()

    def _shutdown(signum, _frame):
        print(f"[proxy] signal {signum}; terminating backend", file=sys.stderr,
              flush=True)
        stop.set()
        _terminate_child(child)
        raise SystemExit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _shutdown)
        except (ValueError, OSError):
            pass

    print(f"[proxy] waiting for backend {args.upstream}{args.health_path} …",
          flush=True)
    if not _wait_ready(args.upstream, args.health_path, args.ready_timeout_s):
        print("[proxy] FATAL: backend never became ready", file=sys.stderr)
        try:
            child.terminate()
        except (OSError, subprocess.SubprocessError):
            pass
        return 1
    _STATE["upstream_ready"] = True
    threading.Thread(target=_monitor,
                     args=(child, args.upstream, args.health_path,
                           args.unhealthy_exit_after, stop,
                           args.monitor_interval_s),
                     daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"[proxy] listening on http://{args.host}:{args.port} "
          f"(upstream {args.upstream})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
