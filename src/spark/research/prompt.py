"""Build the (secret-free) research request and the agent prompt."""

from __future__ import annotations

import json
import urllib.request

from ..budget import GIB, usable_memory_bytes
from ..config.schema import SparkConfig
from ..probe.host import HostProfile
from .types import ResearchRequest

_CARD_URL = "https://huggingface.co/{repo}/raw/main/README.md"


def host_summary(profile: HostProfile, config: SparkConfig) -> dict:
    """Non-secret host facts for the prompt."""
    return {
        "chip": profile.chip,
        "arch": profile.arch,
        "cpu_count": profile.cpu_count,
        "total_memory_gib": round(profile.total_memory_bytes / GIB, 1),
        "usable_memory_gib": round(usable_memory_bytes(profile, config.memory) / GIB, 1),
        "platform": profile.os,
    }


def _repo_base(hf_repo: str) -> str:
    # Strip a `-hf` style ':quant' suffix before fetching the card.
    return hf_repo.split(":", 1)[0]


def fetch_model_card(hf_repo: str, max_chars: int) -> str:
    """Fetch the public model card README. No auth header is ever sent."""
    url = _CARD_URL.format(repo=_repo_base(hf_repo))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "spark-research"})
        with urllib.request.urlopen(req, timeout=15) as r:
            if r.status != 200:
                return ""
            data = r.read(max_chars * 4).decode("utf-8", errors="replace")
            return data[:max_chars]
    except Exception:
        return ""


def build_request(
    hf_repo: str, profile: HostProfile, config: SparkConfig, runtime_names: list[str]
) -> ResearchRequest:
    return ResearchRequest(
        hf_repo=_repo_base(hf_repo),
        host_summary=host_summary(profile, config),
        runtime_names=runtime_names,
        model_card_excerpt=fetch_model_card(hf_repo, config.research.model_card_chars),
    )


_SCHEMA_HINT = """{
  "recommended_backend": "<one of the runtime names>",
  "per_runtime": {
    "<runtime_name>": {
      "quant": "<e.g. q4, q8, bf16, or '' if N/A>",
      "launch_overrides": {"<flag_placeholder>": "<value>"},
      "context_length": <int or null>,
      "rationale": "<short why>"
    }
  },
  "notes": "<overall caveats>"
}"""


def build_prompt(request: ResearchRequest) -> str:
    """Compose the agent prompt. Output MUST be a single JSON object (no prose)."""
    return f"""You are configuring a local LLM inference wrapper. Research the optimal
serving configuration for the model `{request.hf_repo}` on THIS specific host, for
each of the available runtimes, and return ONLY a JSON object (no prose, no code
fences).

HOST (do not assume more than stated):
{json.dumps(request.host_summary, indent=2)}

AVAILABLE RUNTIMES (recommend among these only):
{json.dumps(request.runtime_names)}

Guidance:
- Respect the host's usable memory budget; prefer a quant that fits with headroom.
- Recommend the single best backend for this host in "recommended_backend".
- "launch_overrides" keys are extra flags for that runtime (use sparingly; only if
  they materially help, e.g. context length or rope settings).
- If unsure about a field, use "" or null rather than guessing wildly.

Return EXACTLY this JSON shape:
{_SCHEMA_HINT}

MODEL CARD EXCERPT (public, may be empty):
---
{request.model_card_excerpt}
---

Remember: output ONLY the JSON object."""
