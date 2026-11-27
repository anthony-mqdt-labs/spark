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

Two auth modes:
* **API key** (OpenRouter, DeepSeek): a static key from ``SPARK_PROVIDER_API_KEY``.
* **ChatGPT plan** (``--oauth-runtime``, spec 0005): short-lived access tokens
  from the supervisor's token broker over the inherited ``SPARK_TOKEN_FD``
  socketpair (``spark.secrets.broker``). This process never holds the
  rotating refresh token and never reads the Keychain. In this mode ``/v1/models`` is answered locally from a cached catalog,
  ``/v1/chat/completions`` is translated to the Responses API
  (``chatgpt_responses``), ``/v1/responses`` passes through, and an upstream
  401 gets one forced refresh and one retry.

Runs under spark's own interpreter. Stdlib only, plus spark's own modules.
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

from spark.errors import ProviderAuthError, SparkError
from spark.shims.chatgpt_responses import (
    ResponseCollector,
    TranslationError,
    chat_to_responses,
    error_status,
    models_payload,
    to_chat_completion,
)
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


class _StaticKey:
    refreshable = False

    def __init__(self, key: str) -> None:
        self.key = key

    def bearer(self) -> str:
        return self.key

    def rejected(self, token: str) -> str:
        return token


#: Plan model catalog (OAuth mode), refreshed in the background.
_MODELS: dict = {"ids": []}


def _catalog_ids() -> list[str]:
    ids = list(_MODELS["ids"])
    default = _CFG.get("default_model") or ""
    # Consumers (Symbiote) read data[0].id: the configured model leads.
    return [default, *[i for i in ids if i != default]] if default else ids


def _refresh_catalog() -> None:
    from spark.secrets.oauth import list_models

    try:
        ids = list_models(_CFG["base_url"], _CFG["auth"].bearer(), _CFG["runtime"])
    except SparkError as exc:
        _log("warn", "catalog_refresh_failed", error_type=type(exc).__name__, code=exc.code)
        return
    _MODELS["ids"] = ids
    _log("info", "catalog_refreshed", models=len(ids))


def _base_path() -> str:
    return urllib.parse.urlsplit(_CFG["base_url"]).path.rstrip("/")


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

        if _CFG.get("oauth"):
            if self.command == "GET" and path == "/v1/models":
                rec.update(status=200, outcome="models_local")
                self._json(200, models_payload(_catalog_ids()))
                return
            if self.command == "POST" and path == "/v1/chat/completions":
                self._chat_via_responses(body or b"", rec)
                return
            if path != "/v1/responses":
                rec.update(status=404, outcome="route_not_offered")
                self._error(
                    404,
                    "the ChatGPT plan provider offers /v1/models, /v1/chat/completions "
                    "and /v1/responses only",
                )
                return
        self._forward(split, body, rec)

    # -- upstream -------------------------------------------------------------
    def _auth_failed(self, exc: SparkError, rec: dict) -> None:
        if isinstance(exc, ProviderAuthError):
            rec.update(status=401, outcome="auth_failed", code=exc.code)
            hint = exc.remediation[0] if exc.remediation else ""
            self._error(401, f"{exc.message} {hint}".strip())
        else:
            rec.update(status=502, outcome="auth_unavailable", code=exc.code)
            self._error(502, exc.message)

    def _send(self, method: str, target: str, body: bytes | None, headers: dict, rec: dict):
        """(conn, resp), or None after an error response was written. An
        upstream 401 gets one forced refresh and one retry (OAuth mode only)."""
        auth = _CFG["auth"]
        try:
            token = auth.bearer()
        except SparkError as exc:
            self._auth_failed(exc, rec)
            return None
        for attempt in (1, 2):
            headers["Authorization"] = f"Bearer {token}"
            conn, _ = open_upstream(_CFG["base_url"], _CFG["timeout_s"])
            t_send = time.monotonic()
            try:
                conn.request(method, target, body=body, headers=headers)
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException, ValueError) as exc:
                # ValueError: http.client refusing a malformed header value.
                conn.close()
                rec.update(
                    status=502, outcome="upstream_unreachable", error_type=type(exc).__name__
                )
                self._error(502, f"provider unreachable ({type(exc).__name__})")
                return None
            rec["ttfb_ms"] = round((time.monotonic() - t_send) * 1000, 1)
            if resp.status == 401 and attempt == 1 and auth.refreshable:
                resp.read()
                conn.close()
                try:
                    token = auth.rejected(token)
                except SparkError as exc:
                    self._auth_failed(exc, rec)
                    return None
                rec["auth_retried"] = True
                continue
            return conn, resp
        raise AssertionError("unreachable")  # the loop always returns

    def _forward(self, split: urllib.parse.SplitResult, body: bytes | None, rec: dict) -> None:
        headers = {k: v for k, v in self.headers.items() if k.lower() not in _DROP_REQUEST_HEADERS}
        headers["Accept-Encoding"] = "identity"
        target = (
            _base_path() + split.path[len("/v1") :] + (f"?{split.query}" if split.query else "")
        )
        sent_req = self._send(self.command, target, body, headers, rec)
        if sent_req is None:
            return
        conn, resp = sent_req
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

    def _chat_via_responses(self, body: bytes, rec: dict) -> None:
        """Chat Completions in → Responses (stream) upstream → one chat.completion out."""
        try:
            req, dropped = chat_to_responses(json.loads(body or b"null"))
        except ValueError as exc:  # TranslationError is a ValueError, as is bad JSON
            param = exc.param if isinstance(exc, TranslationError) else "body"
            rec.update(status=400, outcome="translation_refused", param=param)
            self._error(400, f"{param}: {exc}")
            return
        rec.update(translated=True, dropped=dropped)
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        sent_req = self._send(
            "POST", _base_path() + "/responses", json.dumps(req).encode(), headers, rec
        )
        if sent_req is None:
            return
        conn, resp = sent_req
        col = ResponseCollector()
        try:
            if resp.status != 200:
                # Pre-stream refusal: pass it through unchanged (a usage-limit 429
                # must stay 429 so the consumer goes offline and keeps checking).
                raw = resp.read(1 << 20)
                rec.update(status=resp.status, outcome="upstream_refused")
                try:
                    err = json.loads(raw).get("error") or {}
                except (ValueError, AttributeError):
                    err = {}
                if isinstance(err, dict) and err.get("code"):
                    rec["code"] = err["code"]
                    if err["code"] == "subscription_sharing_unsupported_capability":
                        # A field spark should have dropped: a translation bug.
                        _log("error", "translation_gap", param=err.get("param"), dropped=dropped)
                self.send_response(resp.status)
                self.send_header(
                    "Content-Type", resp.getheader("Content-Type") or "application/json"
                )
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            for line in iter(resp.readline, b""):
                col.feed_line(line.decode("utf-8", "replace"))
                if col.done:
                    break
            col.close()
        except (OSError, http.client.HTTPException) as exc:
            rec.update(status=502, outcome="stream_aborted", error_type=type(exc).__name__)
            self._error(502, f"provider stream aborted ({type(exc).__name__})")
            return
        finally:
            conn.close()
        if col.error is not None:
            status = error_status(col.error)
            rec.update(status=status, outcome="response_failed", code=col.error.get("code"))
            if col.error.get("code") == "subscription_sharing_unsupported_capability":
                _log("error", "translation_gap", param=col.error.get("param"), dropped=dropped)
            self._json(status, {"error": col.error})
            return
        if not col.done:
            rec.update(status=502, outcome="stream_incomplete")
            self._error(502, "provider stream ended before response.completed")
            return
        out = to_chat_completion(req["model"], col.text, col.usage, col.finish, rec["request_id"])
        rec.update(
            status=200,
            outcome="ok",
            finish=col.finish,
            completion_tokens=out["usage"]["completion_tokens"],
        )
        self._json(200, out)

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
    p.add_argument(
        "--oauth-runtime",
        default="",
        help="ChatGPT-plan mode: runtime whose [server.provider_oauth] session to use.",
    )
    p.add_argument("--default-model", default="", help="Listed first in /v1/models (OAuth mode).")
    p.add_argument("--models-refresh-s", type=float, default=600.0)
    return p.parse_args(argv)


def _init_telemetry(component: str):
    from spark.config.paths import resolve_paths
    from spark.telemetry import init_telemetry

    session = os.environ.get("SPARK_SESSION_ID") or f"proxy-{os.getpid()}"
    return init_telemetry(resolve_paths().log_dir, session)


def _catalog_loop(stop: threading.Event, every_s: float) -> None:
    while not stop.wait(every_s):
        _refresh_catalog()


def main(argv=None) -> int:
    started = time.monotonic()
    args = _parse_args(argv)
    from spark.config.schema import check_provider_base_url
    from spark.redaction import register_secret_value

    key = os.environ.pop(KEY_ENV, "").strip()
    _CFG.update(
        component=args.component,
        telemetry=_init_telemetry(args.component),
        base_url=check_provider_base_url(args.upstream),
        timeout_s=args.timeout_s,
        oauth=bool(args.oauth_runtime),
        runtime=args.oauth_runtime,
        default_model=args.default_model,
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
        mode="chatgpt_oauth" if args.oauth_runtime else "api_key",
        key_present=bool(key),  # presence only; "api_key" would be masked by the redactor
    )
    token_fd = os.environ.pop("SPARK_TOKEN_FD", None)
    if args.oauth_runtime:
        if token_fd is None or not token_fd.isdigit():
            _log("error", "token_channel_missing")
            print(
                "[provider-proxy] FATAL: SPARK_TOKEN_FD not set (run via spark)",
                file=sys.stderr,
                flush=True,
            )
            return 3
        from spark.secrets.broker import BrokerClient

        auth = BrokerClient(int(token_fd))
        _CFG["auth"] = auth
        try:
            auth.bearer()  # fail fast, before the supervisor sees /health
        except SparkError as exc:
            _log("error", "oauth_unusable", code=exc.code, error_type=type(exc).__name__)
            print(f"[provider-proxy] FATAL: {exc.message}", file=sys.stderr, flush=True)
            return 3
        _refresh_catalog()
    else:
        if not key:
            _log("error", "api_key_missing", env_var=KEY_ENV)
            print(f"[provider-proxy] FATAL: {KEY_ENV} not set", file=sys.stderr, flush=True)
            return 2
        register_secret_value(key)
        _CFG["auth"] = _StaticKey(key)

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
    if args.oauth_runtime:
        threading.Thread(
            target=_catalog_loop, args=(stop, args.models_refresh_s), daemon=True
        ).start()
    _log("info", "listening", host=args.host, port=args.port, adopted_fd=listen_fd is not None)
    print(f"[provider-proxy] http://{args.host}:{args.port}/v1 -> {upstream_host}", flush=True)
    parent = os.getppid()
    try:
        while not stop.wait(1.0):
            if os.getppid() != parent:
                # Supervisor gone (SIGKILL): an orphaned proxy would keep serving
                # with a token nobody refreshes. Exit instead.
                _log("warn", "parent_gone", ppid=os.getppid())
                break
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
