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

import re

from ..errors import SparkError
from ..telemetry.jsonl import _secret_values, _secret_lock

# Known credential shapes (HF, OpenAI, AWS, generic bearer/sk- tokens, etc.).
_TOKEN_PATTERNS = [
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._-]{20,}\b"),
]


class SecretLeakError(SparkError):
    code = "SEC_PROMPT_LEAK"
    exit_code = 77


def assert_no_secrets(prompt: str) -> None:
    """Raise SecretLeakError if the prompt appears to contain secret material."""
    # (a) Exact registered secret values.
    with _secret_lock:
        registered = tuple(_secret_values)
    for v in registered:
        if v and v in prompt:
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
