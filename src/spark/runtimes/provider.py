"""Hosted-provider backend (OpenRouter, DeepSeek, any OpenAI-compatible API).

Selected for every runtime whose TOML sets ``[server] provider_base_url``, so a
new provider is a config file, not code. The supervised child is spark's own
``shims/provider_proxy.py``: a loopback proxy that injects the API key (from
the Keychain, via ``secret_env``) on the way upstream. Unlike ``relay``, the
consumer never sees the upstream URL or the key.

``prepare`` is the fail-fast gate: a missing or rejected key is an operator
error that no restart can fix, so it fails before anything is spawned.
"""

from __future__ import annotations

import http.client
import sys
from importlib.resources import files
from pathlib import Path

from ..config.schema import ModelEntry
from ..errors import ConfigError, ProviderAuthError, ProviderUnreachableError, SecretNotFoundError
from ..shims.provider_proxy import KEY_ENV, open_upstream
from ..telemetry import get_telemetry
from .base import Backend, register

#: One pre-launch round trip; not a runtime flag, just a bound on a probe.
_AUTH_CHECK_TIMEOUT_S = 15.0


def _shim() -> Path:
    return Path(str(files("spark.shims") / "provider_proxy.py"))


@register("_provider")
class ProviderBackend(Backend):
    @property
    def accepts_reserved_fd(self) -> bool:
        return True  # the shim adopts spark's pre-bound listen socket

    @property
    def secret_name(self) -> str:
        name = self.rt.server.secret_env.get(KEY_ENV, "")
        if not name:
            raise ConfigError(
                f"Runtime '{self.name}' sets provider_base_url but no API key mapping.",
                remediation=[
                    f"Add to config/runtimes/{self.name}.toml: [server.secret_env] "
                    f'{KEY_ENV} = "{self.name}_api_key"',
                ],
                context={"runtime": self.name},
            )
        return name

    def api_model_id(self, entry: ModelEntry) -> str:
        return entry.remote_model or entry.id

    def extra_env(self) -> dict[str, str]:
        # Correlates the proxy's JSONL with this supervisor session.
        session = str(getattr(get_telemetry(), "session_id", "unbound"))
        return {**super().extra_env(), "SPARK_SESSION_ID": session}

    def build_launch_cmd(self, entry: ModelEntry, host: str, port: int) -> list[str]:
        s = self.rt.server
        return [
            sys.executable,
            str(_shim()),
            "--host",
            host,
            "--port",
            str(port),
            "--upstream",
            s.provider_base_url,
            "--timeout-s",
            str(s.provider_timeout_s),
            "--component",
            self.name,
        ]

    def prepare(self, entry: ModelEntry, *, secrets=None) -> None:
        name = self.secret_name
        set_key = f"spark secret set {name}"
        if secrets is None or not secrets.exists(name):
            raise SecretNotFoundError(
                f"No API key stored for '{self.name}' (vault entry '{name}').",
                remediation=[f"Store it (hidden prompt, never argv): {set_key}", "Then retry."],
                context={"runtime": self.name, "vault_entry": name},
            )
        key = secrets.get(name)  # registered with the redactor by the store
        status = self._auth_check(key)
        log = get_telemetry()
        if status in (401, 403):
            log.error("provider", "auth_rejected", runtime=self.name, status=status)
            raise ProviderAuthError(
                f"'{self.name}' rejected the stored API key (HTTP {status}).",
                remediation=[
                    f"Re-store a valid key: {set_key}",
                    "Check the key is active (not revoked, has credit) in the provider dashboard.",
                ],
                context={"runtime": self.name, "status": status},
            )
        if status >= 500:
            raise ProviderUnreachableError(
                f"'{self.name}' answered HTTP {status} to the key check.",
                remediation=["The provider is having trouble; retry shortly."],
                context={"runtime": self.name, "status": status},
            )
        # 2xx: verified. Other 4xx (429 rate limit, 404 on a moved check path)
        # say nothing against the key — launch, and leave a trace.
        level = "info" if 200 <= status < 300 else "warn"
        log.emit(level, "provider", "auth_check", runtime=self.name, status=status)

    def _auth_check(self, key: str) -> int:
        s = self.rt.server
        path = "/" + s.provider_auth_check_path.lstrip("/")
        conn, base_path = open_upstream(s.provider_base_url, _AUTH_CHECK_TIMEOUT_S)
        try:
            conn.request("GET", base_path + path, headers={"Authorization": f"Bearer {key}"})
            resp = conn.getresponse()
            resp.read()
            return resp.status
        except (OSError, http.client.HTTPException) as exc:
            raise ProviderUnreachableError(
                f"Cannot reach '{self.name}' at {s.provider_base_url} ({type(exc).__name__}).",
                remediation=[
                    "Check network connectivity / VPN / proxy, then retry.",
                    f"Probe: curl -sI {s.provider_base_url}",
                ],
                context={"runtime": self.name, "error_type": type(exc).__name__},
                cause=exc,
            ) from exc
        finally:
            conn.close()
