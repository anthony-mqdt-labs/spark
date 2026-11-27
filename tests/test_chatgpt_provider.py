"""ChatGPT-plan provider (spec 0005): SIWC login/refresh/logout, Chat→Responses
translation, and the proxy's OAuth mode, against a fake OpenAI that implements
the documented protocol: PKCE-checked code exchange, rotating single-use
refresh tokens, SSE responses."""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from spark.config.schema import ModelEntry, OAuthSpec
from spark.errors import ProviderAuthError, SecretNotFoundError
from spark.secrets.oauth import (
    check_id_token,
    ensure_access_token,
    load_session,
    login,
    logout,
    oauth_lock_path,
    session_lock,
)
from spark.shims.chatgpt_responses import (
    ResponseCollector,
    TranslationError,
    chat_to_responses,
    error_status,
    to_chat_completion,
)
from tests.helpers import FakeStore

FULL_SCOPE = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
SYMBIOTE_REQ = {
    "model": "gpt-big",
    "messages": [
        {"role": "system", "content": "You are a terse coach. Max 15 lines."},
        {"role": "user", "content": "Why does my test fail?"},
    ],
    "temperature": 0.3,
    "max_tokens": 700,
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _b64(d: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()


class FakeOpenAI:
    """auth.openai.com + api.openai.com, per the SIWC docs."""

    def __init__(self, *, scope: str = FULL_SCOPE):
        self.scope = scope
        self.issued = "oaiapp_test_issued"
        self.tamper_state = self.bad_nonce = self.deny = False
        self.authz: list[dict] = []
        self.token_calls: list[dict] = []
        self.revoked: list[dict] = []
        self.codes: dict[str, dict] = {}
        self.live_refresh: set[str] = set()
        self.spent_refresh: set[str] = set()
        self.access: set[str] = set()
        self.responses_seen: list[dict] = []
        self.models_calls = 0
        self.mode = "ok"
        self.fail_401_once = False
        self.expires_in = 3600
        self.release = threading.Event()
        self._n = 0
        self._lock = threading.RLock()  # _token -> _issue -> _next re-enter
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, payload, ctype="application/json", extra=None):
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _bearer_ok(self):
                return (self.headers.get("Authorization") or "")[7:] in outer.access

            def do_GET(self):
                u = urllib.parse.urlsplit(self.path)
                q = dict(urllib.parse.parse_qsl(u.query))
                if u.path == "/.well-known/openid-configuration":
                    self._send(
                        200,
                        {"issuer": outer.base, "revocation_endpoint": outer.base + "/oauth/revoke"},
                    )
                elif u.path == "/api/accounts/authorize":
                    outer.authz.append(q)
                    host = q.get("ext_agent_host_id", "")
                    if not host.startswith(
                        ("urn:uuid:", "did:key:", "urn:ietf:params:oauth:jwk-thumbprint:")
                    ):
                        # What the real endpoint returned for a bare random id.
                        self._send(
                            400,
                            {
                                "error": {
                                    "message": "Invalid authorize request",
                                    "type": "invalid_request_error",
                                    "param": "ext_agent_host_id",
                                    "code": "invalid_authorize_request",
                                }
                            },
                        )
                        return
                    code = outer._next("code")
                    outer.codes[code] = q
                    back = {"state": q["state"] + ("x" if outer.tamper_state else "")}
                    if outer.deny:
                        back["error"] = "access_denied"
                    else:
                        back.update(code=code, scope=outer.scope)
                        if q["client_id"] == "dynamic_agent_client":
                            back["client_id"] = outer.issued
                    loc = q["redirect_uri"] + "?" + urllib.parse.urlencode(back)
                    self.send_response(302)
                    self.send_header("Location", loc)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                elif u.path == "/v1/models":
                    outer.models_calls += 1
                    if not self._bearer_ok():
                        self._send(401, {"error": {"code": "invalid_token"}})
                        return
                    self._send(
                        200,
                        {
                            "models": [
                                {"slug": "gpt-fast", "visibility": "list"},
                                {"slug": "gpt-hidden", "visibility": "hide"},
                                {"slug": "gpt-big", "visibility": "list"},
                            ]
                        },
                    )
                else:
                    self._send(404, {})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n)
                u = urllib.parse.urlsplit(self.path)
                if u.path == "/api/accounts/oauth/token":
                    form = dict(urllib.parse.parse_qsl(raw.decode()))
                    outer.token_calls.append(form)
                    self._send(*outer._token(form))
                elif u.path == "/oauth/revoke":
                    outer.revoked.append(dict(urllib.parse.parse_qsl(raw.decode())))
                    self._send(200, b"")
                elif u.path == "/v1/responses":
                    self._responses(json.loads(raw))
                else:
                    self._send(404, {})

            def _responses(self, body):
                outer.responses_seen.append(
                    {"body": body, "auth": self.headers.get("Authorization")}
                )
                if outer.fail_401_once:
                    outer.fail_401_once = False
                    self._send(401, {"error": {"code": "invalid_token"}})
                    return
                if not self._bearer_ok():
                    self._send(401, {"error": {"code": "invalid_token"}})
                    return
                if outer.mode == "limit429":
                    self._send(
                        429,
                        {
                            "error": {
                                "code": "subscription_sharing_usage_limit_exceeded",
                                "message": "plan limit",
                            }
                        },
                    )
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                def ev(d):
                    self.wfile.write(f"event: {d['type']}\ndata: {json.dumps(d)}\n\n".encode())
                    self.wfile.flush()

                ev({"type": "response.created", "response": {}})
                ev({"type": "response.output_text.delta", "delta": "Hel"})
                ev({"type": "response.output_text.delta", "delta": "lo"})
                if outer.mode == "truncated":
                    return
                if outer.mode == "failed":
                    ev(
                        {
                            "type": "response.failed",
                            "response": {
                                "error": {
                                    "code": "subscription_sharing_usage_limit_exceeded",
                                    "message": "x",
                                }
                            },
                        }
                    )
                    return
                kind = "response.incomplete" if outer.mode == "incomplete" else "response.completed"
                ev(
                    {
                        "type": kind,
                        "response": {
                            "usage": {"input_tokens": 12, "output_tokens": 2, "total_tokens": 14}
                        },
                    }
                )

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _next(self, prefix: str) -> str:
        with self._lock:
            self._n += 1
            return f"{prefix}-{self._n}-" + "q" * 24

    def _issue(self, client_id: str, nonce: str = "") -> dict:
        access, refresh = self._next("at"), self._next("rt")
        self.access.add(access)
        self.live_refresh.add(refresh)
        doc = {
            "access_token": access,
            "refresh_token": refresh,
            "token_type": "Bearer",
            "expires_in": self.expires_in,
            "scope": self.scope,
        }
        if nonce:
            claims = {
                "iss": self.base,
                "aud": client_id,
                "sub": "user-1",
                "exp": time.time() + 3600,
                "nonce": "wrong" if self.bad_nonce else nonce,
            }
            doc["id_token"] = f"{_b64({'alg': 'RS256'})}.{_b64(claims)}.sig"
        return doc

    def _token(self, form: dict) -> tuple[int, dict]:
        with self._lock:
            if form.get("grant_type") == "authorization_code":
                q = self.codes.pop(form.get("code", ""), None)
                if q is None:
                    return 400, {"error": "invalid_grant"}
                challenge = (
                    base64.urlsafe_b64encode(
                        hashlib.sha256(form["code_verifier"].encode()).digest()
                    )
                    .rstrip(b"=")
                    .decode()
                )
                ok = (
                    challenge == q["code_challenge"]
                    and form["redirect_uri"] == q["redirect_uri"]
                    and form["client_id"] == self.issued
                    and form.get("resource") == q["resource"]
                )
                if not ok:
                    return 400, {"error": "invalid_grant"}
                return 200, self._issue(form["client_id"], q["nonce"])
            if form.get("grant_type") == "refresh_token":
                rt = form.get("refresh_token", "")
                if rt in self.spent_refresh:
                    return 400, {"error": "refresh_token_reused"}
                if rt not in self.live_refresh or form.get("client_id") != self.issued:
                    return 400, {"error": "invalid_grant"}
                self.live_refresh.discard(rt)
                self.spent_refresh.add(rt)
                return 200, self._issue(form["client_id"])
            return 400, {"error": "unsupported_grant_type"}

    def spec(self, **kw) -> OAuthSpec:
        base = dict(
            discovery_url=self.base + "/.well-known/openid-configuration",
            authorize_url=self.base + "/api/accounts/authorize",
            token_url=self.base + "/api/accounts/oauth/token",
            scopes=FULL_SCOPE.split(),
            resource=self.base + "/v1",
            session_secret="test_chatgpt_session",
            redirect_port=_free_port(),
            login_timeout_s=10.0,
        )
        base.update(kw)
        return OAuthSpec(**base)

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _browser(url: str) -> None:
    urllib.request.urlopen(url, timeout=10).read()  # follows the 302 to our callback


@pytest.fixture
def fake():
    f = FakeOpenAI()
    yield f
    f.close()


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture
def lock(tmp_path):
    return tmp_path / "locks" / "session.lock"


def _login(fake, store, lock, spec=None):
    spec = spec or fake.spec()
    return spec, login(store, spec, "openai", lock, open_browser=_browser, announce=lambda u: None)


# --- translation ---------------------------------------------------------------------
def test_symbiote_request_translates_per_contract():
    req, dropped = chat_to_responses(SYMBIOTE_REQ)
    assert req == {
        "model": "gpt-big",
        "instructions": "You are a terse coach. Max 15 lines.",
        "input": [{"role": "user", "content": "Why does my test fail?"}],
        "store": False,
        "stream": True,
    }
    assert dropped == ["max_tokens", "temperature"]


def test_translation_keeps_roles_joins_systems_and_text_parts():
    req, _ = chat_to_responses(
        {
            "model": "m",
            "messages": [
                {"role": "system", "content": "a"},
                {"role": "developer", "content": "d"},
                {"role": "system", "content": [{"type": "text", "text": "b"}]},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "u1"}, {"type": "text", "text": "u2"}],
                },
                {"role": "assistant", "content": "prev"},
            ],
            "stream": False,
            "n": 1,
        }
    )
    assert req["instructions"] == "a\n\nb"
    assert req["input"] == [
        {"role": "developer", "content": "d"},
        {"role": "user", "content": "u1u2"},
        {"role": "assistant", "content": "prev"},
    ]


@pytest.mark.parametrize(
    "body,param",
    [
        ({"model": "m", "messages": [{"role": "user", "content": "x"}], "stream": True}, "stream"),
        (
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "tools": [{"type": "function"}],
            },
            "tools",
        ),
        ({"model": "m", "messages": [{"role": "user", "content": "x"}], "n": 2}, "n"),
        (
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "response_format": {"type": "json_object"},
            },
            "response_format",
        ),
        ({"messages": [{"role": "user", "content": "x"}]}, "model"),
        ({"model": "m", "messages": []}, "messages"),
        ({"model": "m", "messages": [{"role": "system", "content": "only"}]}, "messages"),
        ({"model": "m", "messages": [{"role": "tool", "content": "x"}]}, "messages[0].role"),
        (
            {"model": "m", "messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
            "messages[0].content[0]",
        ),
        ({"model": "m", "messages": [{"role": "user", "content": 5}]}, "messages[0].content"),
        ([1, 2], "body"),
    ],
)
def test_out_of_contract_requests_are_refused_by_name(body, param):
    with pytest.raises(TranslationError) as ei:
        chat_to_responses(body)
    assert ei.value.param == param


def _collect(lines):
    c = ResponseCollector()
    for line in lines:
        c.feed_line(line)
    c.close()
    return c


def test_collector_concatenates_deltas_and_reads_usage():
    c = _collect(
        [
            "event: response.output_text.delta\r\n",
            'data: {"type":"response.output_text.delta","delta":"Hel"}\n',
            "\n",
            ": keepalive\n",
            'data: {"type":"response.output_text.delta","delta":"lo"}\n',
            "\n",
            'data: {"type":"response.completed","response":{"usage":{"output_tokens":2}}}\n',
        ]
    )  # final event without a trailing blank line: close() still dispatches it
    assert c.done and c.text == "Hello" and c.usage == {"output_tokens": 2}
    assert c.finish == "stop" and c.error is None


def test_collector_falls_back_to_final_output_text():
    final = {
        "type": "response.completed",
        "response": {
            "output": [
                {"type": "reasoning"},
                {"type": "message", "content": [{"type": "output_text", "text": "Full"}]},
            ]
        },
    }
    assert _collect([f"data: {json.dumps(final)}\n", "\n"]).text == "Full"


def test_collector_incomplete_failed_error_and_truncation():
    inc = _collect(['data: {"type":"response.incomplete","response":{}}\n', "\n"])
    assert inc.done and inc.finish == "length"
    failed = _collect(
        [
            'data: {"type":"response.failed","response":{"error":{"code":"subscription_sharing_usage_limit_exceeded"}}}\n',
            "\n",
        ]
    )
    assert failed.error["code"] == "subscription_sharing_usage_limit_exceeded"
    err = _collect(['data: {"type":"error","code":"server_error","message":"boom"}\n', "\n"])
    assert err.error["code"] == "server_error"
    trunc = _collect(
        [
            'data: {"type":"response.output_text.delta","delta":"x"}\n',
            "\n",
            "data: [DONE]\n",
            "\n",
            "data: not-json\n",
            "\n",
        ]
    )
    assert not trunc.done and trunc.text == "x"


def test_error_status_mapping_and_chat_completion_shape():
    assert error_status({"code": "subscription_sharing_usage_limit_exceeded"}) == 429
    assert error_status({"code": "subscription_sharing_user_not_eligible"}) == 403
    assert error_status({"code": "subscription_sharing_unsupported_capability"}) == 400
    assert error_status({"code": "subscription_sharing_usage_unavailable"}) == 503
    assert error_status({"code": "whatever"}) == 502
    out = to_chat_completion("m", "hi", {"input_tokens": 3, "output_tokens": 2}, "stop", "r1")
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "hi"}
    assert out["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


# --- login ----------------------------------------------------------------------------
def test_first_login_registers_and_stores_session(fake, store, lock):
    spec, sess = _login(fake, store, lock)
    q = fake.authz[0]
    assert q["client_id"] == "dynamic_agent_client"
    assert q["agent_name_hint"] == "spark" and q["code_challenge_method"] == "S256"
    assert q["resource"] == spec.resource and q["scope"] == FULL_SCOPE
    assert q["redirect_uri"] == f"http://127.0.0.1:{spec.redirect_port}/auth/callback"
    assert q["ext_agent_host_id"] and q["nonce"] and q["state"]
    exchange = fake.token_calls[0]
    assert exchange["client_id"] == fake.issued  # the issued id, never dynamic_agent_client
    with session_lock(lock):
        saved = load_session(store, spec, "openai")
    assert saved.client_id == fake.issued and saved.subject == "user-1"
    assert saved.refresh_token in fake.live_refresh and saved.access_token in fake.access
    assert "chatgpt.tokens.use.direct" in saved.scopes and saved.host_id == q["ext_agent_host_id"]


def test_relogin_reuses_issued_client_and_host_id(fake, store, lock):
    spec, first = _login(fake, store, lock)
    _login(fake, store, lock, spec.model_copy(update={"redirect_port": _free_port()}))
    q = fake.authz[1]
    assert q["client_id"] == fake.issued and "agent_name_hint" not in q
    assert q["ext_agent_host_id"] == first.host_id


@pytest.mark.parametrize("attr", ["tamper_state", "bad_nonce", "deny"])
def test_login_rejects_forged_or_failed_callbacks(fake, store, lock, attr):
    setattr(fake, attr, True)
    with pytest.raises(ProviderAuthError):
        _login(fake, store, lock)
    assert not store.exists("test_chatgpt_session") or not json.loads(
        store.get("test_chatgpt_session")
    ).get("refresh_token")


def test_ineligible_account_keeps_registration_but_no_tokens(fake, store, lock):
    fake.scope = "openid profile email offline_access"
    with pytest.raises(ProviderAuthError) as ei:
        _login(fake, store, lock)
    assert ei.value.code == "SEC_NOT_ELIGIBLE"
    saved = json.loads(store.get("test_chatgpt_session"))
    assert saved["client_id"] == fake.issued and saved["refresh_token"] == ""


def test_id_token_checks():
    tok = lambda **c: f"h.{_b64(c)}.s"  # noqa: E731
    good = dict(iss="I", aud="C", sub="u", exp=time.time() + 60, nonce="N")
    assert check_id_token(tok(**good), client_id="C", nonce="N", issuer="I")["sub"] == "u"
    for bad in (
        {"iss": "X"},
        {"aud": ["other"]},
        {"exp": time.time() - 3600},
        {"nonce": "Z"},
        {"sub": ""},
    ):
        with pytest.raises(ProviderAuthError):
            check_id_token(tok(**{**good, **bad}), client_id="C", nonce="N", issuer="I")
    with pytest.raises(ProviderAuthError):
        check_id_token("garbage", client_id="C", nonce="N", issuer="I")


# --- refresh ---------------------------------------------------------------------------
def test_valid_token_is_reused_without_refresh(fake, store, lock):
    spec, sess = _login(fake, store, lock)
    calls = len(fake.token_calls)
    assert ensure_access_token(store, spec, "openai", lock).access_token == sess.access_token
    assert len(fake.token_calls) == calls


def test_near_expiry_refresh_rotates_and_persists(fake, store, lock):
    fake.expires_in = 60  # inside the 300 s margin
    spec, sess = _login(fake, store, lock)
    new = ensure_access_token(store, spec, "openai", lock)
    assert new.access_token != sess.access_token
    assert sess.refresh_token in fake.spent_refresh and new.refresh_token in fake.live_refresh
    with session_lock(lock):
        assert load_session(store, spec, "openai").refresh_token == new.refresh_token


def test_concurrent_forced_refreshes_spend_the_token_once(fake, store, lock):
    spec, sess = _login(fake, store, lock)
    results, errors = [], []

    def worker():
        try:
            results.append(
                ensure_access_token(store, spec, "openai", lock, stale=sess.access_token)
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    refreshes = [c for c in fake.token_calls if c["grant_type"] == "refresh_token"]
    assert errors == [] and len(refreshes) == 1
    assert len({r.access_token for r in results}) == 1


def test_earliest_refresh_at_defers_a_proactive_refresh(fake, store, lock):
    fake.expires_in = 60
    spec, sess = _login(fake, store, lock)
    with session_lock(lock):
        s = load_session(store, spec, "openai")
        s.earliest_refresh_at = time.time() + 600
        from spark.secrets.oauth import save_session

        save_session(store, spec, s)
    assert ensure_access_token(store, spec, "openai", lock).access_token == sess.access_token


def test_reused_refresh_token_demands_login_and_clears_tokens(fake, store, lock):
    spec, sess = _login(fake, store, lock)
    fake.live_refresh.discard(sess.refresh_token)
    fake.spent_refresh.add(sess.refresh_token)  # another client spent it
    with pytest.raises(ProviderAuthError) as ei:
        ensure_access_token(store, spec, "openai", lock, stale=sess.access_token)
    assert "spark login openai" in ei.value.remediation[0]
    saved = json.loads(store.get("test_chatgpt_session"))
    assert saved["refresh_token"] == saved["access_token"] == ""
    assert saved["client_id"] == fake.issued  # registration survives


def test_not_signed_in(fake, store, lock):
    with pytest.raises(ProviderAuthError):
        ensure_access_token(store, fake.spec(), "openai", lock)


def test_logout_revokes_and_keeps_registration(fake, store, lock):
    spec, sess = _login(fake, store, lock)
    assert logout(store, spec, "openai", lock) is True
    assert fake.revoked[0]["token"] == sess.refresh_token
    assert fake.revoked[0]["token_type_hint"] == "refresh_token"
    saved = json.loads(store.get("test_chatgpt_session"))
    assert saved["refresh_token"] == "" and saved["client_id"] == fake.issued
    assert logout(store, spec, "openai", lock) is False


# --- launch gate ----------------------------------------------------------------------
def test_prepare_gate(fake, store, config, paths, monkeypatch):
    from spark.runtimes import get_backend

    monkeypatch.setenv("SPARK_HOME", str(paths.data_dir.parent))
    b = get_backend("openai", config)
    spec = fake.spec(session_secret=b.oauth.session_secret)
    b.rt = b.rt.model_copy(deep=True)
    b.rt.server.provider_oauth = spec
    with pytest.raises(SecretNotFoundError) as ei:
        b.prepare(ModelEntry(id="openai"), secrets=store)
    assert "spark login openai" in ei.value.remediation[0]
    login(
        store,
        spec,
        "openai",
        oauth_lock_path(paths, spec),
        open_browser=_browser,
        announce=lambda u: None,
    )
    b.prepare(ModelEntry(id="openai"), secrets=store)  # healthy session: passes
    assert b.secret_env_map() == {}  # nothing rides in env for rotating tokens
    cmd = b.build_launch_cmd(ModelEntry(id="openai", remote_model="gpt-big"), "127.0.0.1", 8102)
    assert cmd[-6:-2] == ["--oauth-runtime", "openai", "--default-model", "gpt-big"]
    assert not any(t.startswith(("at-", "rt-")) for t in cmd)


# --- proxy (OAuth mode, in-process) ----------------------------------------------------
@pytest.fixture
def proxy(fake, store, lock, tmp_path):
    from spark.shims import provider_proxy as pp
    from spark.telemetry import get_telemetry

    from spark.secrets.broker import BrokerClient, TokenBroker

    spec, _ = _login(fake, store, lock)
    broker = TokenBroker(store, spec, "openai", lock)
    pp._CFG.clear()
    pp._CFG.update(
        component="openai",
        telemetry=get_telemetry(),
        base_url=fake.base + "/v1",
        timeout_s=10.0,
        oauth=True,
        runtime="openai",
        default_model="gpt-big",
        auth=BrokerClient(broker.new_channel().detach()),  # the real child-side path
    )
    pp._MODELS["ids"] = []
    pp._refresh_catalog()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), pp._Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_port
    srv.shutdown()
    srv.server_close()
    broker.close()
    pp._CFG.clear()


def _call(port, method, path, body=None):
    import http.client

    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = json.dumps(body).encode() if body is not None else None
    c.request(
        method,
        path,
        body=data,
        headers={
            "Host": f"127.0.0.1:{port}",
            "Content-Type": "application/json",
            "Authorization": "Bearer symbiote-junk",
        },
    )
    r = c.getresponse()
    raw = r.read()
    if "json" not in (r.getheader("Content-Type") or ""):
        return r.status, raw  # e.g. a passed-through SSE stream
    return r.status, (json.loads(raw) if raw else None)


def test_symbiote_round_trip(proxy, fake, tmp_path):
    status, out = _call(proxy, "POST", "/v1/chat/completions", SYMBIOTE_REQ)
    assert status == 200
    assert out["object"] == "chat.completion" and out["model"] == "gpt-big"
    assert out["choices"][0]["message"]["content"] == "Hello"
    assert out["usage"]["completion_tokens"] == 2
    seen = fake.responses_seen[-1]
    assert seen["body"]["instructions"].startswith("You are a terse coach")
    assert seen["body"]["store"] is False and seen["body"]["stream"] is True
    assert "temperature" not in seen["body"] and "max_output_tokens" not in seen["body"]
    assert seen["auth"] != "Bearer symbiote-junk" and seen["auth"][7:] in fake.access
    logs = "".join(p.read_text() for p in (tmp_path / "logs").rglob("*.jsonl"))
    for tok in fake.access | fake.live_refresh | fake.spent_refresh:
        assert tok not in logs


def test_models_answered_locally_default_first(proxy, fake):
    calls = fake.models_calls
    t0 = time.monotonic()
    status, out = _call(proxy, "GET", "/v1/models")
    assert status == 200 and time.monotonic() - t0 < 0.5
    assert [m["id"] for m in out["data"]] == ["gpt-big", "gpt-fast"]  # hidden filtered
    assert fake.models_calls == calls  # nothing forwarded


def test_usage_limit_429_passes_through_unchanged(proxy, fake):
    fake.mode = "limit429"
    status, out = _call(proxy, "POST", "/v1/chat/completions", SYMBIOTE_REQ)
    assert status == 429
    assert out["error"]["code"] == "subscription_sharing_usage_limit_exceeded"


@pytest.mark.parametrize("mode,status", [("failed", 429), ("truncated", 502)])
def test_in_stream_failures_map_to_statuses(proxy, fake, mode, status):
    fake.mode = mode
    assert _call(proxy, "POST", "/v1/chat/completions", SYMBIOTE_REQ)[0] == status


def test_incomplete_sets_length_finish(proxy, fake):
    fake.mode = "incomplete"
    status, out = _call(proxy, "POST", "/v1/chat/completions", SYMBIOTE_REQ)
    assert status == 200 and out["choices"][0]["finish_reason"] == "length"


def test_upstream_401_refreshes_once_and_retries(proxy, fake):
    fake.fail_401_once = True
    status, out = _call(proxy, "POST", "/v1/chat/completions", SYMBIOTE_REQ)
    assert status == 200 and out["choices"][0]["message"]["content"] == "Hello"
    assert len([c for c in fake.token_calls if c["grant_type"] == "refresh_token"]) == 1


def test_dead_session_is_401_naming_login(proxy, fake, store):
    fake.access.clear()  # every access token now rejected
    fake.spent_refresh |= fake.live_refresh
    fake.live_refresh.clear()  # and the refresh token was spent elsewhere
    status, out = _call(proxy, "POST", "/v1/chat/completions", SYMBIOTE_REQ)
    assert status == 401 and "spark login openai" in out["error"]["message"]


@pytest.mark.parametrize(
    "path,body,status",
    [
        ("/v1/chat/completions", {**SYMBIOTE_REQ, "stream": True}, 400),
        ("/v1/chat/completions", {**SYMBIOTE_REQ, "tools": [{"type": "function"}]}, 400),
        ("/v1/embeddings", {"model": "m", "input": "x"}, 404),
    ],
)
def test_out_of_contract_routes_and_bodies(proxy, fake, path, body, status):
    before = len(fake.responses_seen)
    assert _call(proxy, "POST", path, body)[0] == status
    assert len(fake.responses_seen) == before  # nothing billed


def test_responses_passthrough(proxy, fake):
    status, raw = _call(
        proxy,
        "POST",
        "/v1/responses",
        {"model": "gpt-big", "input": "hi", "store": False, "stream": True},
    )
    assert status == 200 and fake.responses_seen[-1]["body"]["input"] == "hi"
    assert b"response.completed" in raw  # SSE streamed back untouched


def test_bundled_openai_runtime_is_valid(config):
    rt = config.runtimes["openai"]
    spec = rt.server.provider_oauth
    assert spec is not None and rt.server.provider_base_url == "https://api.openai.com/v1"
    assert spec.required_scope in spec.scopes and spec.resource == "https://api.openai.com/v1"
    assert spec.authorize_url.startswith("https://auth.openai.com/")


# --- token broker (option A: the child never holds the refresh token) -----------------
def _raw_ask(sock, req: bytes) -> dict:
    sock.sendall(req)
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf += chunk
    return json.loads(buf) if buf else {}


def test_broker_hands_out_access_tokens_never_refresh_tokens(fake, store, lock):
    from spark.secrets.broker import TokenBroker

    spec, sess = _login(fake, store, lock)
    broker = TokenBroker(store, spec, "openai", lock)
    chan = broker.new_channel()
    try:
        got = _raw_ask(chan, b'{"op":"get"}\n')
        assert got == {"ok": True, "token": sess.access_token}
        rotated = _raw_ask(
            chan, json.dumps({"op": "rejected", "token": sess.access_token}).encode() + b"\n"
        )
        assert rotated["ok"] and rotated["token"] != sess.access_token
        refreshes = [c for c in fake.token_calls if c["grant_type"] == "refresh_token"]
        assert len(refreshes) == 1
        # No reply ever carried any refresh token, old or new.
        for reply in (got, rotated):
            blob = json.dumps(reply)
            assert not any(rt in blob for rt in fake.live_refresh | fake.spent_refresh)
    finally:
        broker.close()


def test_broker_rejects_malformed_and_reports_dead_sessions(fake, store, lock):
    from spark.secrets.broker import MAX_LINE, TokenBroker

    spec, sess = _login(fake, store, lock)
    broker = TokenBroker(store, spec, "openai", lock)
    chan = broker.new_channel()
    try:
        assert _raw_ask(chan, b"not json\n")["code"] == "BROKER_BAD_REQUEST"
        assert _raw_ask(chan, b'{"op":"export_refresh_token"}\n')["code"] == "BROKER_BAD_REQUEST"
        fake.spent_refresh |= fake.live_refresh
        fake.live_refresh.clear()
        dead = _raw_ask(
            chan, json.dumps({"op": "rejected", "token": sess.access_token}).encode() + b"\n"
        )
        assert dead["ok"] is False and dead["kind"] == "auth"
        assert "spark login openai" in dead["remediation"][0]
        # An oversized line ends the channel rather than buffering without bound.
        chan.sendall(b"x" * (MAX_LINE + 10) + b"\n")
        chan.settimeout(5)
        assert chan.recv(10) == b""
    finally:
        broker.close()


def test_broker_client_maps_refusals_to_typed_errors(fake, store, lock):
    from spark.secrets.broker import BrokerClient, TokenBroker

    spec, _ = _login(fake, store, lock)
    broker = TokenBroker(store, spec, "openai", lock)
    client = BrokerClient(broker.new_channel().detach())
    assert client.bearer() in fake.access
    fake.spent_refresh |= fake.live_refresh
    fake.live_refresh.clear()
    with pytest.raises(ProviderAuthError) as ei:
        client.rejected(client.bearer())
    assert "spark login openai" in ei.value.remediation[0]
    broker.close()
    from spark.errors import SparkError

    with pytest.raises(SparkError) as gone:
        client.bearer()
    assert gone.value.code == "BROKER_GONE"


def test_proxy_child_code_never_touches_the_keychain():
    """Static guard on option A: the network-facing child has no path to the
    session store or the refresh logic."""
    import inspect

    from spark.shims import provider_proxy

    src = inspect.getsource(provider_proxy)
    for forbidden in (
        "get_secret_store",
        "KeychainStore",
        "ensure_access_token",
        "load_session",
        "save_session",
        "refresh_token",
    ):
        assert forbidden not in src, forbidden


def test_child_refuses_oauth_mode_without_token_channel(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    shim = Path(__file__).resolve().parents[1] / "src" / "spark" / "shims" / "provider_proxy.py"
    env = {k: v for k, v in os.environ.items() if k != "SPARK_TOKEN_FD"}
    env["SPARK_HOME"] = str(tmp_path)
    r = subprocess.run(
        [
            sys.executable,
            str(shim),
            "--port",
            str(_free_port()),
            "--upstream",
            "https://api.openai.com/v1",
            "--oauth-runtime",
            "openai",
        ],
        env=env,
        capture_output=True,
        timeout=20,
    )
    assert r.returncode == 3 and b"SPARK_TOKEN_FD" in r.stderr


def test_supervisor_brokers_tokens_to_a_real_child_and_survives_restart(
    fake, store, config, paths, monkeypatch
):
    """End to end: a real proxy subprocess, spawned by the real supervisor, gets
    its token only over the broker socket. A crash-restart gets a fresh channel.
    The child's environment never contains a token."""
    import subprocess

    from spark.probe.host import HostProfile, RuntimeStatus
    from spark.runner import Supervisor
    from spark.runtimes import get_backend

    monkeypatch.setenv("SPARK_HOME", str(paths.data_dir.parent))
    port = _free_port()
    cfg = config.model_copy(deep=True)
    cfg.general.port_range = (port, port)
    cfg.supervisor.health_interval_s = 0.2
    cfg.supervisor.backoff_base_s = 0.0
    rt = cfg.runtimes["openai"].server
    rt.default_port = port
    rt.provider_base_url = fake.base + "/v1"
    spec = fake.spec(session_secret=rt.provider_oauth.session_secret)
    rt.provider_oauth = spec
    login(
        store,
        spec,
        "openai",
        oauth_lock_path(paths, spec),
        open_browser=_browser,
        announce=lambda u: None,
    )
    profile = HostProfile(
        os="Darwin",
        arch="arm64",
        chip="Apple M2",
        total_memory_bytes=16 * 2**30,
        cpu_count=8,
        probed_at=0.0,
        runtimes={"openai": RuntimeStatus(name="openai", available=True)},
    )
    entry = ModelEntry(
        id="openai", backend="openai", research_status="manual", remote_model="gpt-big"
    )
    sup = Supervisor(get_backend("openai", cfg), entry, cfg, profile, paths, store)
    seen: dict = {}

    def child_env(pid: int) -> str:
        return subprocess.run(
            ["ps", "eww", "-o", "command=", "-p", str(pid)], capture_output=True, text=True
        ).stdout

    def after_restart(first_pid: int) -> None:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            proc = sup._proc
            if proc is not None and proc.pid != first_pid and proc.poll() is None:
                try:
                    seen["resp2"] = _call(port, "POST", "/v1/chat/completions", SYMBIOTE_REQ)
                    break
                except OSError:
                    pass
            time.sleep(0.1)
        sup._stop.set()

    def on_ready(info):
        seen["info"] = info
        seen["resp1"] = _call(port, "POST", "/v1/chat/completions", SYMBIOTE_REQ)
        seen["models"] = _call(port, "GET", "/v1/models")
        pid = sup._proc.pid
        seen["env"] = child_env(pid)
        sup._proc.kill()
        threading.Thread(target=after_restart, args=(pid,), daemon=True).start()

    result = sup.run(on_ready=on_ready)
    assert seen["resp1"][0] == 200
    assert seen["resp1"][1]["choices"][0]["message"]["content"] == "Hello"
    assert seen["models"][1]["data"][0]["id"] == "gpt-big"
    assert result.restarts == 1 and seen["resp2"][0] == 200
    assert "SPARK_TOKEN_FD=" in seen["env"]  # the channel was handed over...
    server_log = sup.log_path.read_text()
    for tok in fake.access | fake.live_refresh | fake.spent_refresh:
        assert tok not in seen["env"]  # ...and no token rode in env or argv
        assert tok not in server_log


def test_host_id_is_a_documented_urn_uuid(fake, store, lock):
    import re
    import uuid as _uuid

    _login(fake, store, lock)
    host = fake.authz[0]["ext_agent_host_id"]
    assert re.fullmatch(
        r"urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", host
    )
    assert _uuid.UUID(host[len("urn:uuid:") :]).version == 4


def test_legacy_invalid_host_id_is_replaced_not_reused(fake, store, lock):
    from spark.secrets.oauth import Session, save_session

    spec = fake.spec()
    with session_lock(lock):
        save_session(store, spec, Session(client_id=fake.issued, host_id="kZ3_qP0aV-8sXc1mNb2LwQ"))
    _login(fake, store, lock, spec)  # would 400 at authorize if the bare id were reused
    with session_lock(lock):
        saved = load_session(store, spec, "openai")
    assert saved.host_id.startswith("urn:uuid:") and saved.refresh_token
