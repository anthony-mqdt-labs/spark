"""Token broker: the supervisor hands a proxy child *access tokens only* (spec 0005 §5).

A ChatGPT-plan session holds a 30-day, single-use, rotating refresh token. That
token must not live in the network-facing proxy child, the process that parses
HTTP from anything on loopback. So the supervisor (``spark run``) owns the
session, and the child gets an anonymous ``socketpair`` end through
``pass_fds``. It has no filesystem path, so no other process can connect.

Protocol: newline-delimited JSON, one request then one reply, child → parent.

    {"op": "get"}                       → {"ok": true, "token": "<access>"}
    {"op": "rejected", "token": "<t>"}  → force a refresh unless t was already
                                           replaced; reply as for "get"
    any failure                         → {"ok": false, "kind": "auth"|"unavailable",
                                           "code": …, "message": …, "remediation": [...]}

The refresh token never crosses the socket. Refreshes still run under the
session ``flock`` (``spark.secrets.oauth``), so a concurrent ``spark login`` or
a second supervisor never spends the same refresh token twice.
"""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

from ..config.schema import OAuthSpec
from ..errors import ProviderAuthError, SparkError
from ..telemetry import get_telemetry
from .oauth import Session, ensure_access_token
from .store import SecretStore

#: Env var naming the inherited fd in the child (mirrors SPARK_LISTEN_FD).
TOKEN_FD_ENV = "SPARK_TOKEN_FD"
#: A request is a few dozen bytes; anything near this is not our child.
MAX_LINE = 16 * 1024


class TokenBroker:
    def __init__(self, store: SecretStore, spec: OAuthSpec, runtime: str, lock_file: Path):
        self.store, self.spec, self.runtime, self.lock_file = store, spec, runtime, lock_file
        self._sess: Session | None = None
        self._lock = threading.Lock()
        self._socks: list[socket.socket] = []
        self._closed = False

    # -- token logic (parent side) ------------------------------------------------
    def access_token(self, *, stale: str = "") -> str:
        with self._lock:
            sess = self._sess
            usable = sess is not None and sess.access_valid(self.spec.refresh_margin_s)
            if usable and sess.access_token != stale:
                return sess.access_token
            self._sess = ensure_access_token(
                self.store, self.spec, self.runtime, self.lock_file, stale=stale
            )
            return self._sess.access_token

    def _reply(self, req: object) -> dict:
        if not isinstance(req, dict) or req.get("op") not in ("get", "rejected"):
            return {
                "ok": False,
                "kind": "unavailable",
                "code": "BROKER_BAD_REQUEST",
                "message": "malformed broker request",
                "remediation": [],
            }
        stale = req.get("token", "") if req["op"] == "rejected" else ""
        if not isinstance(stale, str):
            stale = ""
        try:
            return {"ok": True, "token": self.access_token(stale=stale)}
        except SparkError as exc:
            return {
                "ok": False,
                "kind": "auth" if isinstance(exc, ProviderAuthError) else "unavailable",
                "code": exc.code,
                "message": exc.message,
                "remediation": list(exc.remediation),
            }

    # -- channels -------------------------------------------------------------------
    def new_channel(self) -> socket.socket:
        """A fresh socketpair for one child. Returns the child's end; the caller
        passes its fd via ``pass_fds`` and closes it after ``Popen``."""
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socks.append(parent)
        threading.Thread(target=self._serve, args=(parent,), daemon=True).start()
        return child

    def _serve(self, sock: socket.socket) -> None:
        log = get_telemetry()
        try:
            with sock.makefile("rwb") as f:
                while not self._closed:
                    line = f.readline(MAX_LINE + 1)
                    if not line:
                        break  # child exited (or was restarted): channel done
                    if len(line) > MAX_LINE or not line.endswith(b"\n"):
                        log.warn("broker", "oversized_request", runtime=self.runtime)
                        break
                    try:
                        req = json.loads(line)
                    except ValueError:
                        req = None
                    reply = self._reply(req)
                    if not reply["ok"]:
                        log.warn(
                            "broker", "token_refused", runtime=self.runtime, code=reply["code"]
                        )
                    f.write(json.dumps(reply).encode() + b"\n")
                    f.flush()
        except OSError:
            pass  # peer gone mid-reply
        finally:
            try:
                sock.close()
            except OSError:
                pass

    def close(self) -> None:
        self._closed = True
        for s in self._socks:
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                s.close()
            except OSError:
                pass
        self._socks.clear()


class BrokerClient:
    """Child side: asks the parent for access tokens. Holds no refresh token and
    never touches the Keychain."""

    refreshable = True

    def __init__(self, fd: int) -> None:
        self._sock = socket.socket(fileno=fd)
        self._f = self._sock.makefile("rwb")
        self._lock = threading.Lock()

    def _ask(self, req: dict) -> str:
        from ..redaction import register_secret_value

        with self._lock:
            try:
                self._f.write(json.dumps(req).encode() + b"\n")
                self._f.flush()
                line = self._f.readline(MAX_LINE + 1)
            except OSError as exc:
                raise SparkError(
                    "Token broker unreachable (supervisor gone).",
                    code="BROKER_GONE",
                    remediation=["Restart the provider: spark run <runtime>"],
                    cause=exc,
                ) from exc
        if not line:
            raise SparkError(
                "Token broker closed the channel (supervisor gone).",
                code="BROKER_GONE",
                remediation=["Restart the provider: spark run <runtime>"],
            )
        try:
            reply = json.loads(line)
        except ValueError:
            reply = {}
        if reply.get("ok") and isinstance(reply.get("token"), str):
            register_secret_value(reply["token"])
            return reply["token"]
        cls = ProviderAuthError if reply.get("kind") == "auth" else SparkError
        raise cls(
            str(reply.get("message") or "token broker refused the request"),
            code=str(reply.get("code") or "BROKER_ERROR"),
            remediation=[str(r) for r in reply.get("remediation") or []] or None,
        )

    def bearer(self) -> str:
        return self._ask({"op": "get"})

    def rejected(self, token: str) -> str:
        return self._ask({"op": "rejected", "token": token})
