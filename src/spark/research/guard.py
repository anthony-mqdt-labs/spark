"""Secrets-guard for outbound research prompts.

The research chain must never leak secret material to an LLM/agent. Defense in
depth:
1. The request is built only from config + host facts + the *public* model card —
   the code path never calls ``secrets.get()``.
2. Before dispatch, this guard scans the final prompt for (a) any literal secret
   value registered with the telemetry redactor, and (b) high-entropy token-shaped
   substrings matching common key prefixes. Any hit aborts the dispatch.
"""

from __future__ import annotations

from ..errors import SecretLeakError as SecretLeakError  # re-export: now in errors.py
from ..redaction import CREDENTIAL_PATTERNS, contains_registered_secret

# Known credential shapes are owned by the shared redaction registry, not by the
# logger: the guard and the log scrubber must agree on what a credential looks
# like (and a telemetry refactor must not silently disable this check).
_TOKEN_PATTERNS = CREDENTIAL_PATTERNS


def assert_no_secrets(prompt: str) -> None:
    """Raise SecretLeakError if the prompt appears to contain secret material."""
    # (a) Exact registered secret values (boundary-aware for short tokens).
    if contains_registered_secret(prompt):
        raise SecretLeakError(
            "Refusing to send a research prompt: it contains a known secret value.",
            remediation=[
                "This is a guard bug — research prompts must never include secrets.",
                "File the prompt-builder path; do not bypass this check.",
            ],
        )
    # (b) Token-shaped substrings.
    for pat in _TOKEN_PATTERNS:
        if pat.search(prompt):
            raise SecretLeakError(
                "Refusing to send a research prompt: it matches a credential pattern.",
                remediation=[
                    f"Matched pattern: {pat.pattern}",
                    "Scrub the offending content from the model card / host summary.",
                ],
                context={"pattern": pat.pattern},
            )
