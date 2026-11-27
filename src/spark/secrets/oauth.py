"""Sign in with ChatGPT (SIWC): login, refresh, revoke, model catalog (spec 0005).

OpenAI's sanctioned way for a local, personal tool to spend the operator's
ChatGPT plan instead of API billing. The protocol is taken verbatim from
developers.openai.com/siwc/token-sharing-open-source/*. Endpoints and scopes
are config (``OAuthSpec``), never literals here.

Session storage: one Keychain entry (``spec.session_secret``) holding JSON:
the issued client id, a stable host id, the rotating refresh token, and the
current access token with its expiry. Refresh tokens are single-use, so every
read-modify-write of that entry runs under one ``flock``. This follows the
docs' "serialize refreshes … so two processes do not race a rotating token",
and it also covers the Keychain store's delete-then-add overwrite window.

Never logged: any token, the authorize URL's PKCE/state/nonce values, or the
ID token. Every token value is registered with the redactor as soon as it is
seen.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import hmac
import http.client
import json
import os
import secrets as pysecrets
import threading
import time
import urllib.parse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable, Iterator

from ..config.paths import SparkPaths
from ..config.schema import OAuthSpec, check_provider_base_url
from ..errors import (
    ConfigError,
    PortInUseError,
    ProviderAuthError,
    ProviderUnreachableError,
)
from ..redaction import register_secret_value
from ..telemetry import get_telemetry
from .store import SecretStore

#: Refresh failures after which only a fresh sign-in helps (errors-and-recovery page).
UNUSABLE_REFRESH = frozenset(
    {
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    }
)

_HTTP_TIMEOUT_S = 15.0  # one OAuth round trip; not a runtime flag


@dataclass
class Session:
    client_id: str
    host_id: str
    subject: str = ""
    scopes: list[str] = field(default_factory=list)
    refresh_token: str = ""
    access_token: str = ""
    expires_at: float = 0.0
    earliest_refresh_at: float = 0.0
    saved_at: float = 0.0

    def access_valid(self, margin_s: float) -> bool:
        return bool(self.access_token) and self.expires_at - time.time() > margin_s


# --- storage ---------------------------------------------------------------------
def oauth_lock_path(paths: SparkPaths, spec: OAuthSpec) -> Path:
    return paths.run_dir / "locks" / f"{spec.session_secret}.oauth.lock"


@contextmanager
def session_lock(lock_file: Path) -> Iterator[None]:
    """Exclusive cross-process lock (flock is per open file, so threads that each
    take it also exclude each other)."""
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def login_needed(runtime: str, why: str, *, code: str = "SEC_REJECTED") -> ProviderAuthError:
    return ProviderAuthError(
        f"ChatGPT sign-in for '{runtime}' is not usable ({why}).",
        code=code,
        remediation=[f"Sign in again: spark login {runtime}", "Then retry."],
        context={"runtime": runtime, "reason": why},
    )


def load_session(store: SecretStore, spec: OAuthSpec, runtime: str) -> Session | None:
    """Caller holds the session lock."""
    if not store.exists(spec.session_secret):
        return None
    raw = store.get(spec.session_secret)
    try:
        doc = json.loads(raw)
        known = {k: doc[k] for k in Session.__dataclass_fields__ if k in doc}
        sess = Session(**known)
    except (ValueError, TypeError) as exc:
        raise login_needed(runtime, "stored session is unreadable") from exc
    for tok in (sess.refresh_token, sess.access_token):
        if tok:
            register_secret_value(tok)
    return sess


def save_session(store: SecretStore, spec: OAuthSpec, sess: Session) -> None:
    """Caller holds the session lock."""
    sess.saved_at = time.time()
    # ponytail: the store overwrites by delete-then-add; a crash in between loses
    # the session (re-login fixes it). The lock keeps readers out of that window.
    store.set(spec.session_secret, json.dumps(asdict(sess)))


# --- http ------------------------------------------------------------------------
def _request(
    method: str,
    url: str,
    *,
    form: dict[str, str] | None = None,
    bearer: str = "",
    timeout_s: float = _HTTP_TIMEOUT_S,
) -> tuple[int, dict]:
    """One request, no redirects followed, JSON body or {}."""
    from ..shims.provider_proxy import open_upstream

    parts = urllib.parse.urlsplit(url)
    conn, _ = open_upstream(f"{parts.scheme}://{parts.netloc}", timeout_s)
    headers = {"Accept": "application/json"}
    body = None
    if form is not None:
        body = urllib.parse.urlencode(form).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    target = parts.path + (f"?{parts.query}" if parts.query else "")
    try:
        conn.request(method, target, body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read(1 << 20)
        status = resp.status
    except (OSError, http.client.HTTPException) as exc:
        raise ProviderUnreachableError(
            f"Cannot reach {parts.hostname} ({type(exc).__name__}).",
            remediation=["Check network connectivity, then retry."],
            context={"host": parts.hostname, "error_type": type(exc).__name__},
            cause=exc,
        ) from exc
    finally:
        conn.close()
    try:
        doc = json.loads(raw) if raw else {}
    except ValueError:
        doc = {}
    return status, doc if isinstance(doc, dict) else {}


def error_code(doc: dict) -> str:
    """OAuth errors are ``{"error": "code"}``; API errors ``{"error": {"code": …}}``."""
    err = doc.get("error")
    if isinstance(err, str):
        return err
    if isinstance(err, dict):
        return str(err.get("code") or err.get("type") or "")
    return ""


def _epoch(v: object) -> float:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v) / 1000 if v > 1e12 else float(v)
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _apply_tokens(sess: Session, doc: dict, runtime: str) -> None:
    now = time.time()
    access = doc.get("access_token")
    if not isinstance(access, str) or not access:
        raise login_needed(runtime, "token response carried no access_token")
    register_secret_value(access)
    sess.access_token = access
    refresh = doc.get("refresh_token")
    if isinstance(refresh, str) and refresh:
        register_secret_value(refresh)
        sess.refresh_token = refresh  # rotated: the old one is now spent
    expires_in = doc.get("expires_in")
    sess.expires_at = now + (float(expires_in) if isinstance(expires_in, (int, float)) else 3600.0)
    sess.earliest_refresh_at = _epoch(doc.get("earliest_refresh_at"))
    scope = doc.get("scope")
    if isinstance(scope, str) and scope.strip():
        sess.scopes = scope.split()


# --- refresh ---------------------------------------------------------------------
def ensure_access_token(
    store: SecretStore,
    spec: OAuthSpec,
    runtime: str,
    lock_file: Path,
    *,
    stale: str = "",
) -> Session:
    """A session whose access token is usable, refreshing at most once.

    ``stale`` is an access token the caller just saw rejected (HTTP 401): it
    forces a refresh, unless another process has already replaced it, in which
    case the replacement is returned without spending the refresh token.
    """
    log = get_telemetry()
    with session_lock(lock_file):
        sess = load_session(store, spec, runtime)
        if sess is None or not (sess.refresh_token or sess.access_token):
            raise login_needed(runtime, "not signed in")
        if sess.access_token != stale and sess.access_valid(spec.refresh_margin_s):
            return sess
        if not stale and sess.access_valid(0) and time.time() < sess.earliest_refresh_at:
            return sess  # still valid, and OpenAI asked us not to refresh yet
        if not sess.refresh_token:
            raise login_needed(runtime, "no refresh token (signed out)")
        status, doc = _request(
            "POST",
            spec.token_url,
            form={
                "grant_type": "refresh_token",
                "client_id": sess.client_id,
                "refresh_token": sess.refresh_token,
                "resource": spec.resource,
            },
        )
        if status == 200:
            _apply_tokens(sess, doc, runtime)
            save_session(store, spec, sess)
            log.info(
                "oauth",
                "token_refreshed",
                runtime=runtime,
                expires_in_s=round(sess.expires_at - time.time()),
                forced=bool(stale),
            )
            return sess
        code = error_code(doc)
        log.error("oauth", "token_refresh_failed", runtime=runtime, status=status, code=code)
        if code in UNUSABLE_REFRESH:
            # Clear the dead tokens; keep the issued client id + host id for re-login.
            sess.refresh_token = sess.access_token = ""
            sess.expires_at = 0.0
            save_session(store, spec, sess)
            raise login_needed(runtime, code)
        if code == "invalid_client":
            raise ConfigError(
                f"OpenAI rejected the issued client id for '{runtime}' (invalid_client).",
                remediation=[
                    f"Sign out and back in to re-register: spark logout {runtime} && "
                    f"spark login {runtime}",
                ],
                context={"runtime": runtime},
            )
        raise ProviderUnreachableError(
            f"Token refresh for '{runtime}' failed (HTTP {status}{', ' + code if code else ''}).",
            remediation=["Retry shortly; if it persists, sign in again: spark login " + runtime],
            context={"runtime": runtime, "status": status, "code": code},
        )


# --- login -----------------------------------------------------------------------
def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def pkce_pair() -> tuple[str, str]:
    verifier = pysecrets.token_urlsafe(64)  # 86 chars, inside RFC 7636's 43..128
    return verifier, _b64url(hashlib.sha256(verifier.encode()).digest())


def discovery(spec: OAuthSpec) -> dict:
    status, doc = _request("GET", spec.discovery_url)
    if status != 200 or not doc.get("issuer"):
        raise ProviderUnreachableError(
            f"OpenID discovery failed (HTTP {status}).",
            remediation=["Retry shortly; check network connectivity."],
            context={"status": status},
        )
    return doc


def check_id_token(token: object, *, client_id: str, nonce: str, issuer: str) -> dict:
    """Claims checks per the SIWC docs (iss, aud, exp, nonce). The signature is
    not verified: OIDC Core §3.1.3.7(6) allows TLS server validation in place
    of it for a token taken directly from the token endpoint, as here.
    ponytail: add JWKS signature checks if the token ever arrives indirectly
    (needs an RSA dependency)."""
    try:
        claims = json.loads(_b64url_decode(str(token).split(".")[1]))
    except (IndexError, ValueError):
        raise ProviderAuthError("Sign-in returned a malformed ID token.") from None
    aud = claims.get("aud")
    auds = aud if isinstance(aud, list) else [aud]
    problems = []
    if claims.get("iss") != issuer:
        problems.append("issuer")
    if client_id not in auds:
        problems.append("audience")
    if not isinstance(claims.get("exp"), (int, float)) or claims["exp"] < time.time() - 60:
        problems.append("expiry")
    if not hmac.compare_digest(str(claims.get("nonce", "")), nonce):
        problems.append("nonce")
    if not isinstance(claims.get("sub"), str) or not claims["sub"]:
        problems.append("subject")
    if problems:
        raise ProviderAuthError(
            f"Sign-in ID token failed checks: {', '.join(problems)}.",
            remediation=["Retry the sign-in; if it repeats, do not trust this network path."],
            context={"failed": problems},
        )
    return claims


class _CallbackServer(HTTPServer):
    allow_reuse_address = True

    def __init__(self, port: int) -> None:
        self.query: dict[str, str] = {}
        self.got = threading.Event()
        super().__init__(("127.0.0.1", port), _CallbackHandler)


class _CallbackHandler(BaseHTTPRequestHandler):
    server: _CallbackServer

    def log_message(self, format, *args):  # noqa: A002 - query carries the code
        pass

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        split = urllib.parse.urlsplit(self.path)
        if split.path != "/auth/callback" or self.server.got.is_set():
            self.send_error(404)
            return
        self.server.query = dict(urllib.parse.parse_qsl(split.query))
        ok = "error" not in self.server.query
        body = b"<!doctype html><title>spark</title><p>" + (
            b"Signed in. You can close this tab." if ok else b"Sign-in failed; see the terminal."
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.server.got.set()


def _listen(port: int, *, fixed: bool) -> _CallbackServer:
    try:
        return _CallbackServer(port)
    except OSError as exc:
        if not fixed:
            return _CallbackServer(0)  # re-sign-in: "only the port may vary"
        raise PortInUseError(
            f"Port {port} is needed for the first ChatGPT sign-in and is busy.",
            remediation=[
                f"Free it (find the holder: lsof -nP -iTCP:{port} -sTCP:LISTEN), then retry.",
                "Codex CLI uses the same port during its own login; finish or cancel that first.",
            ],
            context={"port": port},
            cause=exc,
        ) from exc


def login(
    store: SecretStore,
    spec: OAuthSpec,
    runtime: str,
    lock_file: Path,
    *,
    open_browser: Callable[[str], object],
    announce: Callable[[str], None],
) -> Session:
    """Interactive SIWC sign-in. Registers the app on first use."""
    log = get_telemetry()
    with session_lock(lock_file):
        try:
            prior = load_session(store, spec, runtime)
        except ProviderAuthError:
            prior = None  # unreadable session: start over
    registering = prior is None or not prior.client_id
    client_id = spec.registration_client_id if registering else prior.client_id
    host_id = prior.host_id if prior and prior.host_id else pysecrets.token_urlsafe(16)
    issuer = str(discovery(spec)["issuer"])
    verifier, challenge = pkce_pair()
    state, nonce = pysecrets.token_urlsafe(24), pysecrets.token_urlsafe(24)

    server = _listen(spec.redirect_port, fixed=registering)
    redirect_uri = f"http://127.0.0.1:{server.server_port}/auth/callback"
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": " ".join(spec.scopes),
        "resource": spec.resource,
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "ext_agent_host_id": host_id,
    }
    if registering:
        params["agent_name_hint"] = spec.agent_name
    url = f"{spec.authorize_url}?{urllib.parse.urlencode(params)}"
    serve = threading.Thread(target=server.serve_forever, daemon=True)
    serve.start()  # listener first, then the browser (sign-in page instruction)
    log.info("oauth", "login_started", runtime=runtime, registering=registering)
    try:
        announce(url)
        open_browser(url)
        if not server.got.wait(spec.login_timeout_s):
            raise ProviderAuthError(
                f"No sign-in callback within {spec.login_timeout_s:.0f}s.",
                code="SEC_LOGIN_TIMEOUT",
                remediation=[f"Retry: spark login {runtime} (finish the browser step sooner)."],
            )
    finally:
        server.shutdown()
        server.server_close()
        serve.join(timeout=2)

    q = server.query
    if not hmac.compare_digest(q.get("state", ""), state):
        raise ProviderAuthError(
            "Sign-in callback state did not match (possible forged redirect).",
            remediation=[f"Retry: spark login {runtime}"],
        )
    if q.get("error"):
        raise ProviderAuthError(
            f"Sign-in was not completed ({q['error']}).",
            remediation=[f"Retry and approve access: spark login {runtime}"],
            context={"error": q["error"]},
        )
    code = q.get("code", "")
    issued = q.get("client_id", "")
    if registering:
        if not issued or issued == spec.registration_client_id:
            raise ProviderAuthError(
                "Sign-in did not return an issued client id.",
                remediation=[f"Retry: spark login {runtime}"],
            )
        client_id = issued
    elif issued and issued != client_id:
        raise ProviderAuthError(
            "Sign-in returned a different client id than this host registered.",
            remediation=[
                f"Sign out, then in again: spark logout {runtime} && spark login {runtime}"
            ],
        )
    if not code:
        raise ProviderAuthError("Sign-in callback carried no authorization code.")

    status, doc = _request(
        "POST",
        spec.token_url,
        form={
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
            "resource": spec.resource,
        },
    )
    if status != 200:
        raise ProviderAuthError(
            f"Authorization code exchange failed (HTTP {status}, {error_code(doc) or 'no code'}).",
            remediation=[f"Retry: spark login {runtime}"],
            context={"status": status, "code": error_code(doc)},
        )
    claims = check_id_token(doc.get("id_token"), client_id=client_id, nonce=nonce, issuer=issuer)
    sess = Session(client_id=client_id, host_id=host_id, subject=str(claims["sub"]))
    if q.get("scope"):
        sess.scopes = q["scope"].split()
    _apply_tokens(sess, doc, runtime)
    eligible = spec.required_scope in sess.scopes
    if not eligible:
        # Keep the registration (client id, host id); drop tokens that cannot spend plan usage.
        sess.refresh_token = sess.access_token = ""
    with session_lock(lock_file):
        save_session(store, spec, sess)
    log.info(
        "oauth",
        "login_completed",
        runtime=runtime,
        registered=registering,
        eligible=eligible,
        scopes=sess.scopes,
    )
    if not eligible:
        raise ProviderAuthError(
            f"This ChatGPT account was not granted '{spec.required_scope}' (plan usage).",
            code="SEC_NOT_ELIGIBLE",
            remediation=[
                "Plan usage needs an eligible ChatGPT plan (OpenAI lists Plus/Pro for "
                "open-source and local apps).",
                "Check the plan in ChatGPT settings, then sign in again.",
            ],
            context={"granted": sess.scopes},
        )
    return sess


def logout(store: SecretStore, spec: OAuthSpec, runtime: str, lock_file: Path) -> bool:
    """Revoke the refresh token and clear tokens. Returns whether one was revoked.
    The issued client id and host id are kept (docs: revoking does not delete
    the client)."""
    with session_lock(lock_file):
        sess = load_session(store, spec, runtime)
    if sess is None or not sess.refresh_token:
        return False
    endpoint = str(discovery(spec).get("revocation_endpoint") or "")
    revoked = False
    if endpoint:
        status, _ = _request(
            "POST",
            check_provider_base_url(endpoint),
            form={
                "token": sess.refresh_token,
                "token_type_hint": "refresh_token",
                "client_id": sess.client_id,
            },
        )
        revoked = status == 200
    with session_lock(lock_file):
        sess = load_session(store, spec, runtime) or sess
        sess.refresh_token = sess.access_token = ""
        sess.expires_at = 0.0
        save_session(store, spec, sess)
    get_telemetry().info("oauth", "logout", runtime=runtime, revoked=revoked)
    return revoked


# --- model catalog -----------------------------------------------------------------
def list_models(base_url: str, access_token: str, runtime: str) -> list[str]:
    """Model ids the plan may use (``visibility == "list"``), catalog order."""
    status, doc = _request("GET", f"{base_url}/models", bearer=access_token)
    if status in (401, 403):
        raise login_needed(runtime, f"model catalog refused (HTTP {status}, {error_code(doc)})")
    if status != 200:
        raise ProviderUnreachableError(
            f"Model catalog request failed (HTTP {status}).",
            remediation=["Retry shortly."],
            context={"status": status},
        )
    items = doc.get("models")
    if not isinstance(items, list):
        items = doc.get("data") if isinstance(doc.get("data"), list) else []
    out: list[str] = []
    for m in items:
        if not isinstance(m, dict):
            continue
        slug = m.get("slug") or m.get("id")
        if isinstance(slug, str) and slug and m.get("visibility", "list") == "list":
            out.append(slug)
    return out
