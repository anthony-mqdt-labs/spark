"""Hosted-provider proxy (spec 0004): key injection, guards, streaming, gating.

A fake upstream on loopback stands in for OpenRouter/DeepSeek. It records what
it receives, so the tests can prove the client's credentials never reach it,
spark's key always does, and the key never comes back in a body or a log.
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from spark.config.schema import ModelEntry, ServerSpec, check_provider_base_url
from spark.errors import (
    ConfigError,
    ProviderAuthError,
    ProviderUnreachableError,
    SecretNotFoundError,
)
from spark.runtimes import get_backend
from spark.runtimes.provider import ProviderBackend
from spark.shims.openai_filter_proxy import MAX_BODY_BYTES
from spark.shims.provider_proxy import _host_is_loopback, _peek
from tests.helpers import FakeStore

KEY = "sk-test-0123456789abcdef0123456789"
SHIM = Path(__file__).resolve().parents[1] / "src" / "spark" / "shims" / "provider_proxy.py"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Upstream:
    """Fake provider at http://127.0.0.1:<port>/api/v1."""

    def __init__(self, auth_status: int = 200, tls: tuple[str, str] | None = None):
        self.seen: list[dict] = []
        self.redirect_to = ""  # Location for 302 replies
        self.release = threading.Event()  # gates the second SSE chunk
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                outer.seen.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "headers": dict(self.headers),
                        "body": body,
                    }
                )
                if self.path.endswith("/key") or self.path.endswith("/models"):
                    code = auth_status
                    payload = b'{"data":[]}'
                    self.send_response(code)
                    if code == 302:
                        self.send_header("Location", outer.redirect_to)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if "/redirect" in self.path:
                    self.send_response(302)
                    self.send_header("Location", outer.redirect_to)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if "/ratelimit" in self.path:
                    payload = b'{"error":{"message":"slow down","code":429}}'
                    self.send_response(429)
                    self.send_header("Retry-After", "7")
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if b'"stream": true' in body:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    self.wfile.write(b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')
                    self.wfile.flush()
                    outer.release.wait(5)
                    self.wfile.write(b"data: [DONE]\n\n")
                    return
                payload = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = _reply
            do_POST = _reply

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        scheme = "http"
        if tls:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(*tls)
            self.server.socket = ctx.wrap_socket(self.server.socket, server_side=True)
            scheme = "https"
        self.base = f"{scheme}://127.0.0.1:{self.server.server_port}/api/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def upstream():
    u = _Upstream()
    yield u
    u.close()


def _wait_health(port: int, proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        assert proc.poll() is None, proc.stderr.read().decode() if proc.stderr else "exited"
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            c.request("GET", "/health")
            if c.getresponse().status == 200:
                return
        except OSError:
            time.sleep(0.1)
    raise AssertionError("proxy never became healthy")


def _spawn(base: str, tmp_path, *extra: str) -> tuple[int, subprocess.Popen]:
    port = _free_port()
    env = {
        **os.environ,
        "SPARK_PROVIDER_API_KEY": KEY,
        "SPARK_HOME": str(tmp_path),
        "SPARK_SESSION_ID": "sess-test",
    }
    argv = [sys.executable, str(SHIM), "--port", str(port), "--upstream", base]
    proc = subprocess.Popen(
        [*argv, "--component", "fakeprov", *extra],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    _wait_health(port, proc)
    return port, proc


def _stop(proc: subprocess.Popen) -> int:
    proc.terminate()
    return proc.wait(10)


@pytest.fixture
def proxy(upstream, tmp_path):
    port, proc = _spawn(upstream.base, tmp_path)
    yield port
    _stop(proc)


def _raw(port, method, path, headers, body=b""):
    """Request with full header control (no auto Host / Content-Length)."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
    for k, v in headers.items():
        c.putheader(k, v)
    c.endheaders(body or None)
    r = c.getresponse()
    return r.status, r.read()


def _records(tmp_path, event="request_forwarded") -> list[dict]:
    return [json.loads(line) for line in _logs(tmp_path).splitlines() if f'"{event}"' in line]


def _req(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    h = {"Host": f"127.0.0.1:{port}", **(headers or {})}
    c.request(method, path, body=body, headers=h)
    r = c.getresponse()
    return r.status, r.read()


def _logs(tmp_path) -> str:
    d = tmp_path / "data" / "logs" / "fakeprov"
    return "".join(p.read_text() for p in d.glob("*.jsonl"))


# --- schema ---------------------------------------------------------------------
@pytest.mark.parametrize(
    "url",
    [
        "http://api.example.com/v1",
        "ftp://x",
        "https://",
        "https://a.com/v1?x=1",
        "https://user:pw@a.com/v1",
        "api.example.com",
    ],
)
def test_provider_url_must_be_tls_and_plain(url):
    with pytest.raises(ValueError):
        check_provider_base_url(url)


def test_provider_url_accepts_https_and_loopback_http():
    assert (
        check_provider_base_url("https://openrouter.ai/api/v1/") == "https://openrouter.ai/api/v1"
    )
    assert check_provider_base_url("http://127.0.0.1:9/api") == "http://127.0.0.1:9/api"
    assert ServerSpec(provider_base_url="").provider_base_url == ""


def test_bundled_providers_dispatch_to_provider_backend(config):
    for name, secret in (("openrouter", "openrouter_api_key"), ("deepseek", "deepseek_api_key")):
        b = get_backend(name, config)
        assert isinstance(b, ProviderBackend)
        assert b.secret_name == secret
        assert b.rt.server.provider_base_url.startswith("https://")
        assert not b.requires_local_weights and b.accepts_reserved_fd
        cmd = b.build_launch_cmd(ModelEntry(id=name, backend=name), "127.0.0.1", 8098)
        assert KEY not in " ".join(cmd) and cmd[0] == sys.executable
        e = ModelEntry(id=name, backend=name, remote_model="deepseek-chat")
        assert b.api_model_id(e) == "deepseek-chat"
    assert not isinstance(get_backend("relay", config), ProviderBackend)


# --- prepare gate ----------------------------------------------------------------
def _backend(config, base):
    b = get_backend("openrouter", config)
    b.rt = b.rt.model_copy(deep=True)
    b.rt.server.provider_base_url = base
    return b


def test_prepare_missing_key_names_the_command(config, upstream):
    with pytest.raises(SecretNotFoundError) as ei:
        _backend(config, upstream.base).prepare(ModelEntry(id="o"), secrets=FakeStore())
    assert any("spark secret set openrouter_api_key" in s for s in ei.value.remediation)


@pytest.mark.parametrize(
    "status,raises",
    [
        (200, None),
        (429, None),
        (401, ProviderAuthError),
        (403, ProviderAuthError),
        (503, ProviderUnreachableError),
    ],
)
def test_prepare_auth_check(config, status, raises):
    u = _Upstream(auth_status=status)
    try:
        store = FakeStore()
        store.set("openrouter_api_key", KEY)
        b = _backend(config, u.base)
        if raises:
            with pytest.raises(raises):
                b.prepare(ModelEntry(id="o"), secrets=store)
        else:
            b.prepare(ModelEntry(id="o"), secrets=store)
        assert u.seen[0]["path"] == "/api/v1/key"
        assert u.seen[0]["headers"]["Authorization"] == f"Bearer {KEY}"
    finally:
        u.close()


def test_prepare_unreachable(config):
    store = FakeStore()
    store.set("openrouter_api_key", KEY)
    with pytest.raises(ProviderUnreachableError):
        _backend(config, f"http://127.0.0.1:{_free_port()}/v1").prepare(
            ModelEntry(id="o"), secrets=store
        )


# --- proxy end to end --------------------------------------------------------------
def test_injects_key_strips_client_credentials_maps_path(proxy, upstream, tmp_path):
    body = json.dumps({"model": "deepseek/deepseek-chat", "messages": []}).encode()
    status, out = _req(
        proxy,
        "POST",
        "/v1/chat/completions?x=1",
        body,
        {
            "Content-Type": "application/json",
            "Authorization": "Bearer client-junk",
            "Cookie": "s=1",
            "Accept-Encoding": "gzip",
            "HTTP-Referer": "https://app.example",
        },
    )
    assert status == 200 and json.loads(out)["choices"][0]["message"]["content"] == "ok"
    seen = upstream.seen[-1]
    h = {k.lower(): v for k, v in seen["headers"].items()}
    assert seen["path"] == "/api/v1/chat/completions?x=1"
    assert h["authorization"] == f"Bearer {KEY}"
    assert "cookie" not in h and h["accept-encoding"] == "identity"
    assert h["host"].endswith(str(upstream.server.server_port))  # upstream's, not ours
    assert h["http-referer"] == "https://app.example"  # provider attribution passes through
    assert seen["body"] == body
    logs = _logs(tmp_path)
    assert KEY not in logs
    rec = [json.loads(line) for line in logs.splitlines() if '"request_forwarded"' in line][-1]
    assert rec["model"] == "deepseek/deepseek-chat" and rec["status"] == 200
    assert rec["session_id"] == "sess-test" and rec["outcome"] == "ok"


def test_streams_sse_without_buffering(proxy, upstream):
    c = http.client.HTTPConnection("127.0.0.1", proxy, timeout=10)
    c.request(
        "POST",
        "/v1/chat/completions",
        body=json.dumps({"model": "m", "stream": True}).encode(),
        headers={"Content-Type": "application/json", "Host": "localhost"},
    )
    r = c.getresponse()
    assert r.status == 200
    t0 = time.monotonic()
    first = r.read1(4096)  # upstream is still blocked on `release`
    # A buffering proxy would block here until upstream's 5 s wait ends and
    # then deliver both events at once.
    assert b'"hi"' in first and b"[DONE]" not in first
    assert time.monotonic() - t0 < 2.0
    # The open stream must not block other clients (threaded server).
    status, _ = _req(
        proxy,
        "POST",
        "/v1/chat/completions",
        b'{"model":"m"}',
        {"Content-Type": "application/json"},
    )
    assert status == 200
    upstream.release.set()
    assert b"[DONE]" in first + r.read()


@pytest.mark.parametrize(
    "method,path,headers,code",
    [
        (
            "POST",
            "/v1/chat/completions",
            {"Host": "evil.example", "Content-Type": "application/json"},
            403,
        ),
        ("POST", "/v1/chat/completions", {"Content-Type": "text/plain"}, 415),
        (
            "POST",
            "/v1/chat/completions",
            {"Content-Type": "application/x-www-form-urlencoded"},
            415,
        ),
        ("OPTIONS", "/v1/chat/completions", {}, 501),
        ("GET", "/admin", {}, 404),
        ("GET", "/v1/../key", {}, 404),
        ("GET", "/v1/%2e%2e/key", {}, 404),
        ("DELETE", "/v1/models", {}, 501),
    ],
)
def test_browser_abuse_and_path_guards(proxy, upstream, method, path, headers, code):
    before = len(upstream.seen)
    c = http.client.HTTPConnection("127.0.0.1", proxy, timeout=5)
    c.putrequest(method, path, skip_host=True)
    c.putheader("Host", headers.pop("Host", f"127.0.0.1:{proxy}"))
    for k, v in headers.items():
        c.putheader(k, v)
    c.putheader("Content-Length", "2")
    c.endheaders(b"{}")
    r = c.getresponse()
    assert r.status == code
    assert KEY.encode() not in r.read()
    assert len(upstream.seen) == before  # nothing reached the paid API


def test_get_models_passthrough(proxy, upstream):
    status, _ = _req(proxy, "GET", "/v1/models")
    assert status == 200 and upstream.seen[-1]["path"] == "/api/v1/models"


def test_upstream_down_is_openai_shaped_502(tmp_path):
    port, proc = _spawn(f"http://127.0.0.1:{_free_port()}/v1", tmp_path)
    try:
        status, out = _req(
            port, "POST", "/v1/chat/completions", b"{}", {"Content-Type": "application/json"}
        )
        assert status == 502 and KEY.encode() not in out
        assert json.loads(out)["error"]["type"] == "spark_proxy_error"
    finally:
        proc.terminate()
        proc.wait(10)


def test_missing_key_exits_nonzero(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "SPARK_PROVIDER_API_KEY"}
    env["SPARK_HOME"] = str(tmp_path)
    r = subprocess.run(
        [
            sys.executable,
            str(SHIM),
            "--port",
            str(_free_port()),
            "--upstream",
            "https://api.example.com",
        ],
        env=env,
        capture_output=True,
        timeout=20,
    )
    assert r.returncode == 2 and b"SPARK_PROVIDER_API_KEY" in r.stderr


# --- supervisor integration -------------------------------------------------------
def test_supervisor_serves_provider_end_to_end(paths, config, upstream, monkeypatch):
    from spark.probe.host import HostProfile, RuntimeStatus
    from spark.runner import Supervisor

    monkeypatch.setenv("SPARK_HOME", str(paths.data_dir.parent))
    port = _free_port()
    cfg = config.model_copy(deep=True)
    cfg.general.port_range = (port, port)
    rt = cfg.runtimes["openrouter"].server
    rt.default_port = port
    rt.provider_base_url = upstream.base
    store = FakeStore()
    store.set("openrouter_api_key", KEY)
    profile = HostProfile(
        os="Darwin",
        arch="arm64",
        chip="Apple M2",
        total_memory_bytes=16 * 2**30,
        cpu_count=8,
        probed_at=0.0,
        runtimes={"openrouter": RuntimeStatus(name="openrouter", available=True)},
    )
    entry = ModelEntry(
        id="openrouter",
        backend="openrouter",
        research_status="manual",
        remote_model="deepseek/deepseek-chat",
    )
    sup = Supervisor(get_backend("openrouter", cfg), entry, cfg, profile, paths, store)
    seen = {}

    def on_ready(info):
        seen["info"] = info
        seen["resp"] = _req(
            port,
            "POST",
            "/v1/chat/completions",
            b'{"model":"deepseek/deepseek-chat"}',
            {"Content-Type": "application/json"},
        )
        sup._stop.set()

    result = sup.run(on_ready=on_ready)
    assert seen["info"].base_url == f"http://127.0.0.1:{port}/v1"
    assert seen["info"].api_model_id == "deepseek/deepseek-chat"
    assert seen["resp"][0] == 200
    assert upstream.seen[-1]["headers"]["Authorization"] == f"Bearer {KEY}"
    assert result.restarts == 0
    server_log = sup.log_path.read_text()
    assert KEY not in server_log


# --- added coverage: edges the first pass missed ------------------------------------
@pytest.mark.parametrize(
    "host,ok",
    [
        ("127.0.0.1:8098", True),
        ("127.0.0.1", True),
        ("localhost:8098", True),
        ("LOCALHOST", True),
        ("[::1]:8098", True),
        ("", False),
        ("127.0.0.1.evil.com", False),
        ("localhost.evil.com:80", False),
        ("evil.com:127.0.0.1", False),
        ("0.0.0.0:8098", False),
        ("[::2]:8098", False),
    ],
)
def test_host_is_loopback(host, ok):
    assert _host_is_loopback(host) is ok


@pytest.mark.parametrize(
    "body,expect",
    [
        (None, ("", False)),
        (b"not json", ("", False)),
        (b"[1,2]", ("", False)),
        (b'{"model": 5, "stream": "true"}', ("", False)),
        (b'{"model": "m", "stream": true}', ("m", True)),
        (json.dumps({"model": "x" * 500}).encode(), ("x" * 128, False)),
    ],
)
def test_peek_never_trusts_body_shape(body, expect):
    assert _peek(body) == expect


def test_missing_key_mapping_is_config_error(config):
    b = get_backend("openrouter", config)
    b.rt = b.rt.model_copy(deep=True)
    b.rt.server.secret_env = {}
    with pytest.raises(ConfigError):
        b.prepare(ModelEntry(id="o"), secrets=FakeStore())


@pytest.mark.parametrize(
    "headers,body,code",
    [
        ({"Transfer-Encoding": "chunked"}, b"2\r\n{}\r\n0\r\n\r\n", 411),
        ({}, b"", 400),  # no Content-Length at all
        ({"Content-Length": "abc"}, b"", 400),
        ({"Content-Length": str(MAX_BODY_BYTES + 1)}, b"", 400),  # refused before reading
    ],
)
def test_request_body_guards(proxy, upstream, headers, body, code):
    before = len(upstream.seen)
    h = {"Host": f"127.0.0.1:{proxy}", "Content-Type": "application/json", **headers}
    status, _ = _raw(proxy, "POST", "/v1/chat/completions", h, body)
    assert status == code and len(upstream.seen) == before


def test_json_content_type_with_charset_is_accepted(proxy, upstream):
    status, _ = _req(
        proxy,
        "POST",
        "/v1/chat/completions",
        b'{"model":"m"}',
        {"Content-Type": "Application/JSON; charset=utf-8"},
    )
    assert status == 200


def test_upstream_errors_pass_through_verbatim(proxy, upstream):
    c = http.client.HTTPConnection("127.0.0.1", proxy, timeout=10)
    c.request("GET", "/v1/ratelimit", headers={"Host": "localhost"})
    r = c.getresponse()
    assert r.status == 429 and r.getheader("Retry-After") == "7"
    assert json.loads(r.read())["error"]["message"] == "slow down"


def test_redirects_are_never_followed_with_the_key(proxy, upstream, config):
    trap = _Upstream()
    try:
        upstream.redirect_to = trap.base + "/chat/completions"
        status, _ = _req(proxy, "GET", "/v1/redirect")
        assert status == 302  # handed to the client as-is; the client holds no key
        # prepare's key check must not chase a redirect either.
        u = _Upstream(auth_status=302)
        u.redirect_to = trap.base + "/key"
        try:
            store = FakeStore()
            store.set("openrouter_api_key", KEY)
            _backend(config, u.base).prepare(ModelEntry(id="o"), secrets=store)
        finally:
            u.close()
        assert trap.seen == []
    finally:
        trap.close()


def test_upstream_stall_mid_stream_times_out_and_is_logged(upstream, tmp_path):
    port, proc = _spawn(upstream.base, tmp_path, "--timeout-s", "1")
    try:
        t0 = time.monotonic()
        status, out = _req(
            port,
            "POST",
            "/v1/chat/completions",
            json.dumps({"model": "m", "stream": True}).encode(),
            {"Content-Type": "application/json"},
        )
        assert status == 200 and b'"hi"' in out and b"[DONE]" not in out
        assert time.monotonic() - t0 < 4.0  # cut at ~1 s, not upstream's 5 s
        rec = _records(tmp_path)[-1]
        assert rec["outcome"] == "aborted_mid_body" and rec["error_type"] == "TimeoutError"
    finally:
        _stop(proc)


@pytest.fixture
def self_signed(tmp_path):
    if not shutil.which("openssl"):
        pytest.skip("openssl CLI not available")
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-subj",
            "/CN=127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    return str(cert), str(key)


def test_untrusted_tls_certificate_is_refused(self_signed, tmp_path, config):
    u = _Upstream(tls=self_signed)
    try:
        port, proc = _spawn(u.base, tmp_path)
        try:
            status, out = _req(
                port, "POST", "/v1/chat/completions", b"{}", {"Content-Type": "application/json"}
            )
            assert status == 502 and b"SSLCertVerificationError" in out
        finally:
            _stop(proc)
        store = FakeStore()
        store.set("openrouter_api_key", KEY)
        with pytest.raises(ProviderUnreachableError):
            _backend(config, u.base).prepare(ModelEntry(id="o"), secrets=store)
        assert u.seen == []  # the key never crossed an unverified channel
    finally:
        u.close()


def test_sigterm_exits_zero_and_logs_exit(upstream, tmp_path):
    _port, proc = _spawn(upstream.base, tmp_path)
    assert _stop(proc) == 0
    start = _records(tmp_path, "process_start")[-1]
    assert start["key_present"] is True and KEY not in json.dumps(start)
    assert _records(tmp_path, "process_exit")[-1]["exit_code"] == 0


def test_health_probes_are_not_logged(proxy, tmp_path):
    for _ in range(3):
        _req(proxy, "GET", "/health")
    assert all(r["path"] != "/health" for r in _records(tmp_path))


def test_supervisor_restarts_crashed_proxy_on_same_port(paths, config, upstream, monkeypatch):
    from spark.probe.host import HostProfile, RuntimeStatus
    from spark.runner import Supervisor

    monkeypatch.setenv("SPARK_HOME", str(paths.data_dir.parent))
    port = _free_port()
    cfg = config.model_copy(deep=True)
    cfg.general.port_range = (port, port)
    cfg.supervisor.health_interval_s = 0.2
    cfg.supervisor.backoff_base_s = 0.0
    rt = cfg.runtimes["openrouter"].server
    rt.default_port = port
    rt.provider_base_url = upstream.base
    store = FakeStore()
    store.set("openrouter_api_key", KEY)
    profile = HostProfile(
        os="Darwin",
        arch="arm64",
        chip="Apple M2",
        total_memory_bytes=16 * 2**30,
        cpu_count=8,
        probed_at=0.0,
        runtimes={"openrouter": RuntimeStatus(name="openrouter", available=True)},
    )
    entry = ModelEntry(id="openrouter", backend="openrouter", research_status="manual")
    sup = Supervisor(get_backend("openrouter", cfg), entry, cfg, profile, paths, store)
    seen: dict = {}

    def after_restart(first_pid: int) -> None:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            proc = sup._proc
            if proc is not None and proc.pid != first_pid and proc.poll() is None:
                try:
                    seen["resp"] = _req(
                        port,
                        "POST",
                        "/v1/chat/completions",
                        b"{}",
                        {"Content-Type": "application/json"},
                    )
                    break
                except OSError:
                    pass
            time.sleep(0.1)
        sup._stop.set()

    def on_ready(_info):
        pid = sup._proc.pid
        sup._proc.kill()  # simulate a crash after ready
        threading.Thread(target=after_restart, args=(pid,), daemon=True).start()

    result = sup.run(on_ready=on_ready)
    assert result.restarts == 1 and result.port == port
    assert seen["resp"][0] == 200


def test_provider_ports_never_collide_with_local_runtimes(config):
    """Providers prefer ports above the local range, so running one never makes a
    local server (or the other provider) drift off its pinned port."""
    lo, hi = config.general.port_range
    providers = {n: rt for n, rt in config.runtimes.items() if rt.server.provider_base_url}
    assert providers, "bundled providers missing"
    local_ports = {
        rt.server.default_port for n, rt in config.runtimes.items() if n not in providers
    }
    ports = [rt.server.default_port for rt in providers.values()]
    assert len(set(ports)) == len(ports)
    for p in ports:
        assert not lo <= p <= hi and p not in local_ports
