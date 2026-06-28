"""Secret store abstraction.

Threat model (from the operator's working agreement):
- Secrets MUST NOT live in text files inside the repo or data dir.
- Secret values MUST NOT be passed on argv (visible via ``ps``/shell history).
- Secret values MUST NOT be logged (telemetry redaction is the backstop).
- Secret values MUST NOT be reachable by any LLM/research flow — research code
  constructs prompts from config + host profile only and never calls ``get``.

The store is an interface; :class:`~spark.secrets.keychain.KeychainStore` is the
macOS implementation. The only path a secret takes is: vault -> child process env
at spawn. Nothing else may read it.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from ..telemetry import register_secret_value

# Secret names are constrained to a safe charset so they can never be used to
# inject arguments into the backend CLI.
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")

# Well-known secret names spark understands.
HF_TOKEN = "hf_token"


def validate_name(name: str) -> str:
    if not _NAME_RE.match(name or ""):
        from ..errors import SecretError

        raise SecretError(
            f"Invalid secret name: {name!r}",
            code="SEC_ERROR",
            remediation=["Use only letters, digits, '_', '.', '-' (max 128 chars)."],
        )
    return name


class SecretStore(ABC):
    """Backend-agnostic secret vault interface."""

    #: stable identifier for logs/diagnostics
    backend_name: str = "abstract"

    @abstractmethod
    def set(self, name: str, value: str) -> None:
        """Store/overwrite a secret. Value is taken from getpass/stdin upstream."""

    @abstractmethod
    def get(self, name: str) -> str:
        """Retrieve a secret value. Raises SecretNotFoundError if absent.

        Callers MUST NOT log the return value. The value is auto-registered with
        the telemetry redactor as a backstop."""

    @abstractmethod
    def delete(self, name: str) -> None:
        """Remove a secret. Idempotent-ish; raises only on backend failure."""

    @abstractmethod
    def list_names(self) -> list[str]:
        """List secret *names* only — never values."""

    @abstractmethod
    def exists(self, name: str) -> bool:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this backend can be used on this host."""

    # -- shared helper ---------------------------------------------------------
    def _register_for_redaction(self, value: str) -> str:
        register_secret_value(value)
        return value

    def resolve_env(self, mapping: dict[str, str]) -> dict[str, str]:
        """Resolve a {ENV_VAR: secret_name} map to {ENV_VAR: value} for a child
        process. Missing secrets are skipped (not fatal) — a model may not need a
        token. Values are registered with the redactor."""
        out: dict[str, str] = {}
        for env_var, secret_name in mapping.items():
            try:
                if self.exists(secret_name):
                    out[env_var] = self.get(secret_name)
            except Exception:
                # Never let a secret-resolution hiccup crash a launch silently in
                # a way that leaks; just skip and let the runtime report auth need.
                continue
        return out
