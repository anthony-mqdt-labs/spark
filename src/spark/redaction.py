"""Shared secret-redaction registry and scrubbing primitives.

This module owns the one process-wide list of literal secret values that must
never appear in a log, a server log tail, a rendered error, or an outbound
prompt. It deliberately has **no** imports from :mod:`spark.telemetry` or
:mod:`spark.secrets` so either layer can depend on it without a cycle.

Security posture: redaction is a backstop, not the primary control. Secrets are
injected into child environments only; everything here exists for the case where
a value leaks into free-form text despite that.
"""

from __future__ import annotations

import re
import threading
from typing import Any

REDACTED = "***REDACTED***"

# Keys whose values are always masked regardless of content. ``auth`` is matched
# as a whole word segment so ``author`` (a benign metadata key) is not masked.
_SENSITIVE_KEY_HINTS = (
    "token",
    "secret",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "authorization",
    "auth_header",
    "bearer",
    "credential",
)

# "token" as a credential (hf_token, access_token) — not a count or a tokenizer
# (prompt_tokens, completion_tokens, tokenizer_path), which the inference-event
# logging contract requires in the clear.
_TOKEN_NOT_CREDENTIAL = re.compile(r"token(?:s|iz)")

# Known credential shapes, shared with the research prompt guard. A secret that
# was never read through the vault (e.g. one pasted into a model card) still gets
# masked when it looks like a credential.
CREDENTIAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._-]{20,}\b"),
)

_secret_values: set[str] = set()
_secret_lock = threading.Lock()


def register_secret_value(value: str | None) -> None:
    """Register a literal secret so it is masked everywhere.

    Every non-empty value is registered, including short ones. Short values are
    scrubbed only on word boundaries so a two-character token does not blank out
    every substring in a log line.
    """
    if not value:
        return
    with _secret_lock:
        _secret_values.add(value)


def secret_values() -> tuple[str, ...]:
    """Snapshot the registered values (for the prompt guard and diagnostics)."""
    with _secret_lock:
        return tuple(_secret_values)


def reset_secret_values() -> None:
    """Clear the registry. Intended for tests and process-isolation boundaries."""
    with _secret_lock:
        _secret_values.clear()


def is_sensitive_key(key: str) -> bool:
    k = key.lower()
    if re.search(r"(?:^|_)auth(?:_|$)", k):
        return True
    for hint in _SENSITIVE_KEY_HINTS:
        if hint == "token":
            if "token" in _TOKEN_NOT_CREDENTIAL.sub("", k):
                return True
        elif hint in k:
            return True
    return False


def scrub_string(text: str) -> str:
    """Remove registered secrets and credential-shaped substrings from text."""
    values = secret_values()
    if values:
        # Longest first so an overlapping registration cannot leave a fragment.
        for value in sorted(values, key=len, reverse=True):
            if len(value) >= 4:
                if value in text:
                    text = text.replace(value, REDACTED)
            else:
                text = re.sub(
                    rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])",
                    REDACTED,
                    text,
                )
    for pattern in CREDENTIAL_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def redact(obj: Any, *, key: str | None = None) -> Any:
    """Recursively mask sensitive keys and scrub literal secrets from strings."""
    if key is not None and is_sensitive_key(key):
        return REDACTED
    if isinstance(obj, dict):
        return {k: redact(v, key=str(k)) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return scrub_string(obj)
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    # Fallback: stringify unknown types and scrub the representation too.
    return scrub_string(repr(obj))
