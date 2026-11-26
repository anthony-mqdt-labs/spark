"""Hosted-provider proxy: a loopback OpenAI surface that injects the API key.

Consumers point an OpenAI SDK at ``http://127.0.0.1:<port>/v1`` with no key;
this proxy forwards ``/v1/<rest>`` to ``<provider_base_url>/<rest>`` over
verified TLS with ``Authorization: Bearer <key>`` added. The key arrives in the
env (``SPARK_PROVIDER_API_KEY``, resolved by the supervisor from the Keychain),
is popped immediately, registered with the redactor, and never logged or echoed.

Loopback is not a trust boundary against a browser: any web page can POST to
127.0.0.1 and spend the key. Three cheap guards close that:
* ``Host`` must be a loopback name — defeats DNS rebinding;
* ``POST`` must be ``application/json`` — a cross-origin JSON POST needs a CORS
  preflight, and ``OPTIONS`` is unsupported here (501), so it never passes;
* no CORS headers are ever sent.

Responses stream (``read1``), never buffer: time-to-first-token is preserved
for ``stream: true``. Health is local (``GET /health``) so the supervisor's
5-second probe never hits the paid upstream.

Runs under spark's own interpreter. Stdlib only, plus spark's telemetry.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import resource
import signal
import ssl
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from spark.shims.openai_filter_proxy import _HOP_HEADERS, MAX_BODY_BYTES, _adopted_server

KEY_ENV = "SPARK_PROVIDER_API_KEY"

#: Never forwarded: hop-by-hop, the client's own credentials (spark's replace
#: them), and anything http.client recomputes. ``Accept-Encoding`` is forced to
#: identity so SSE arrives unbuffered and the byte counts are real.
_DROP_REQUEST_HEADERS = _HOP_HEADERS | {
    "host",
    "authorization",
    "x-api-key",
    "cookie",
    "content-length",
    "accept-encoding",
}

_LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})

_TLS = ssl.create_default_context()  # system trust store, hostname verification on

#: Process-wide config, set once in :func:`main` before the server starts.
_CFG: dict = {}


def open_upstream(base_url: str, timeout_s: float) -> tuple[http.client.HTTPConnection, str]:
    """A connection to the provider plus its base path. Never follows redirects
    (urllib does, and re-sends ``Authorization`` to the redirect target)."""
    parts = urllib.parse.urlsplit(base_url)
    if parts.scheme == "https":
        conn: http.client.HTTPConnection = http.client.HTTPSConnection(
            parts.hostname or "", parts.port or 443, timeout=timeout_s, context=_TLS
        )
    else:  # schema validation admits plain http for loopback test doubles only
        conn = http.client.HTTPConnection(parts.hostname or "", parts.port or 80, timeout=timeout_s)
    return conn, parts.path.rstrip("/")


def _host_is_loopback(host_header: str) -> bool:
    host = host_header.strip().lower()
    if host.startswith("["):  # [::1]:port
        host = host[1:].split("]", 1)[0]
    elif host.count(":") == 1:
        host = host.split(":", 1)[0]
    return host in _LOOPBACK_NAMES


def _peek(body: bytes | None) -> tuple[str, bool]:
    """Model id and stream flag for the log line — never the content."""
    if not body:
        return "", False
    try:
        doc = json.loads(body)
    except ValueError:
        return "", False
    if not isinstance(doc, dict):
        return "", False
    model = doc.get("model")
    return (model[:128] if isinstance(model, str) else ""), doc.get("stream") is True


class _Handler(BaseHTTPRequestHandler):
    server_version = "spark-provider-proxy/1"
    # Client-socket timeout: a stalled client cannot pin a handler thread. Long
    # enough for any SDK; upstream stalls are governed by --timeout-s instead.
    timeout = 120

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass  # request_forwarded telemetry is the access log

    def _json(self, code: int, payload: dict) -> int:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return len(body)

    def _error(self, code: int, message: str) -> int:
        # OpenAI-shaped, so SDKs surface the message instead of a parse error.
        return self._json(code, {"error": {"message": message, "type": "spark_proxy_error"}})

    def _handle(self) -> None:
        t0 = time.monotonic()
        rid = uuid.uuid4().hex[:12]
        split = urllib.parse.urlsplit(self.path)
        rec: dict = {"request_id": rid, "method": self.command, "path": split.path}
        try:
            self._dispatch(split, rec)
        finally:
            rec["total_ms"] = round((time.monotonic() - t0) * 1000, 1)
            status = rec.get("status", 0)
            level = "info" if 200 <= status < 400 else "warn"
            if rec.get("outcome") != "health":  # the supervisor's 5 s probe; it logs its own
                _log(level, "request_forwarded", **rec)

    def _dispatch(self, split: urllib.parse.SplitResult, rec: dict) -> None:
        if not _host_is_loopback(self.headers.get("Host", "")):
            rec.update(status=403, outcome="host_rejected")
            self._error(403, "Host header must be a loopback name")
            return
        path = split.path
        if path == "/health":
            rec.update(status=200, outcome="health")
            self._json(200, {"status": "ok"})
            return
        if not path.startswith("/v1/") or ".." in urllib.parse.unquote(path).split("/"):
            rec.update(status=404, outcome="path_rejected")
            self._error(404, "only /v1/... is proxied")
            return

        body = None
        if self.command == "POST":
            ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
            if ctype != "application/json":
                rec.update(status=415, outcome="content_type_rejected")
                self._error(415, "POST bodies must be application/json")
                return
            raw_len = self.headers.get("Content-Length") or ""
            if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                rec.update(status=411, outcome="chunked_rejected")
                self._error(411, "chunked request bodies are not supported")
                return
            if not raw_len.isdigit() or int(raw_len) > MAX_BODY_BYTES:
                rec.update(status=400, outcome="length_rejected")
                self._error(400, "missing, invalid, or oversized Content-Length")
                return
            body = self.rfile.read(int(raw_len))
            rec["req_bytes"] = len(body)
            rec["model"], rec["stream"] = _peek(body)

        headers = {k: v for k, v in self.headers.items() if k.lower() not in _DROP_REQUEST_HEADERS}
        headers["Authorization"] = f"Bearer {_CFG['key']}"
        headers["Accept-Encoding"] = "identity"

        conn, base_path = open_upstream(_CFG["base_url"], _CFG["timeout_s"])
        target = base_path + path[len("/v1") :] + (f"?{split.query}" if split.query else "")
        t_send = time.monotonic()
        try:
            conn.request(self.command, target, body=body, headers=headers)
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException, ValueError) as exc:
            # ValueError: http.client refusing a malformed header value.
            conn.close()
            rec.update(status=502, outcome="upstream_unreachable", error_type=type(exc).__name__)
            self._error(502, f"provider unreachable ({type(exc).__name__})")
            return
        rec["ttfb_ms"] = round((time.monotonic() - t_send) * 1000, 1)
        rec["status"] = resp.status
        sent = 0
        try:
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in _HOP_HEADERS:
                    self.send_header(k, v)
            self.end_headers()
            # Close-delimited (HTTP/1.0 handler) unless upstream sent a
            # Content-Length, which is forwarded as-is.
            while chunk := resp.read1(64 * 1024):
                self.wfile.write(chunk)
                self.wfile.flush()
                sent += len(chunk)
            rec["outcome"] = "ok"
        except (OSError, http.client.HTTPException) as exc:
            # Client hung up, or upstream died/timed out mid-stream.
            rec.update(outcome="aborted_mid_body", error_type=type(exc).__name__)
        finally:
            rec["resp_bytes"] = sent
            conn.close()

    do_GET = _handle
    do_POST = _handle


def _log(level: str, event: str, **payload) -> None:
    tel = _CFG.get("telemetry")
    if tel is not None:
        tel.emit(level, _CFG["component"], event, **payload)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Loopback proxy to a hosted OpenAI-compatible API.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--upstream", required=True, help="Provider base URL (https).")
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument("--component", default="provider_proxy", help="Telemetry component name.")
    return p.parse_args(argv)


def _init_telemetry(component: str):
    from spark.config.paths import resolve_paths
    from spark.telemetry import init_telemetry

    session = os.environ.get("SPARK_SESSION_ID") or f"proxy-{os.getpid()}"
    return init_telemetry(resolve_paths().log_dir, session)


def main(argv=None) -> int:
    started = time.monotonic()
    args = _parse_args(argv)
    from spark.config.schema import check_provider_base_url
    from spark.redaction import register_secret_value

    key = os.environ.pop(KEY_ENV, "").strip()
    _CFG.update(
        component=args.component,
        telemetry=_init_telemetry(args.component),
        key=key,
        base_url=check_provider_base_url(args.upstream),
        timeout_s=args.timeout_s,
    )
    upstream_host = urllib.parse.urlsplit(args.upstream).hostname
    _log(
        "info",
        "process_start",
        exe=sys.executable,
        argc=len(sys.argv),
        ppid=os.getppid(),
        cwd=os.getcwd(),
        upstream_host=upstream_host,
        timeout_s=args.timeout_s,
        key_present=bool(key),  # presence only; "api_key" would be masked by the redactor
    )
    if not key:
        _log("error", "api_key_missing", env_var=KEY_ENV)
        print(f"[provider-proxy] FATAL: {KEY_ENV} not set", file=sys.stderr, flush=True)
        return 2
    register_secret_value(key)

    listen_fd = os.environ.pop("SPARK_LISTEN_FD", None)
    try:
        if listen_fd is None:
            server = ThreadingHTTPServer((args.host, args.port), _Handler)
        else:
            server = _adopted_server(int(listen_fd), args.host, args.port, _Handler)
    except (OSError, ValueError) as exc:
        _log("error", "listen_failed", port=args.port, error_type=type(exc).__name__)
        print(f"[provider-proxy] FATAL: cannot listen on {args.port}: {exc}", file=sys.stderr)
        return 1
    server.daemon_threads = True

    stop = threading.Event()
    # Handler does nothing but set the event: logging here could re-enter the
    # telemetry lock the main thread holds (see Supervisor._note_stop_signal).
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    serve = threading.Thread(target=server.serve_forever, daemon=True)
    serve.start()
    _log("info", "listening", host=args.host, port=args.port, adopted_fd=listen_fd is not None)
    print(f"[provider-proxy] http://{args.host}:{args.port}/v1 -> {upstream_host}", flush=True)
    try:
        while not stop.wait(1.0):
            pass
    finally:
        server.shutdown()
        serve.join(timeout=2.0)
        server.server_close()
        _log(
            "info",
            "process_exit",
            exit_code=0,
            signal="stop",
            wall_time_s=round(time.monotonic() - started, 2),
            # macOS reports ru_maxrss in bytes.
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
