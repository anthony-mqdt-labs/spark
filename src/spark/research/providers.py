"""Research providers: CLI agents driven by config (claude, hermes, pi, ...)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from abc import ABC, abstractmethod

from ..config.schema import ResearchProviderDef, SparkConfig
from ..errors import SparkError
from ..telemetry import get_telemetry
from .guard import assert_no_secrets
from .prompt import build_prompt
from .types import ResearchOutput, ResearchRequest

# Raw-output sample size on parse/validation failure (logging mandate: raw bytes
# truncated to 256). Head and tail both logged — truncation shows at the tail.
_SAMPLE_CHARS = 256

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of agent output (tolerant of prose/fences)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty agent output")
    # 1. whole thing is JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # 2. fenced ```json block
    m = _FENCE_RE.search(text)
    if m:
        return json.loads(m.group(1))
    # 3. first balanced {...} (string-aware)
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in agent output")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
    raise ValueError("unterminated JSON object in agent output")


class ResearchProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def research(self, request: ResearchRequest) -> ResearchOutput: ...


def agent_env(scrub: frozenset[str]) -> dict[str, str]:
    """spark's env minus every var that carries a vault secret to a runtime.

    The agent needs its own auth (its env passes through), but a token the
    operator exported for a runtime (``HF_TOKEN``) must not ride along into an
    LLM-driven process — the same rule the prompt guard enforces on the text.
    """
    return {k: v for k, v in os.environ.items() if k not in scrub}


class CliAgentProvider(ResearchProvider):
    def __init__(self, defn: ResearchProviderDef, scrub_env: frozenset[str] = frozenset()) -> None:
        self.defn = defn
        self.name = defn.name
        self.scrub_env = scrub_env | {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}

    def is_available(self) -> bool:
        if not self.defn.enabled:
            return False
        if self.defn.detect_binary:
            return shutil.which(self.defn.detect_binary) is not None
        return True

    def research(self, request: ResearchRequest) -> ResearchOutput:
        prompt = build_prompt(request)
        assert_no_secrets(prompt)  # hard guard before anything leaves the process

        cmd = list(self.defn.command)
        stdin_data = None
        if self.defn.prompt_via == "arg":
            cmd = [tok.replace("{prompt}", prompt) for tok in cmd]
        else:
            stdin_data = prompt

        try:
            proc = subprocess.run(
                cmd, input=stdin_data, capture_output=True, text=True,
                timeout=self.defn.timeout_s, check=False,
                env=agent_env(self.scrub_env),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SparkError(f"{self.name}: failed to run ({exc})", code="NET_RESEARCH") from exc

        if proc.returncode != 0:
            raise SparkError(
                f"{self.name}: exit {proc.returncode}: {(proc.stderr or '').strip()[:200]}",
                code="NET_RESEARCH",
            )

        text = proc.stdout
        try:
            if self.defn.parse == "json_envelope":
                envelope = json.loads(proc.stdout)
                text = envelope.get(self.defn.envelope_field, "")
            obj = _extract_json(text)
            return ResearchOutput.model_validate(obj)
        except Exception as exc:
            # Without a sample the failure is undiagnosable post-hoc (the agent
            # output is gone). Never contains our secrets: the prompt passed the
            # guard, and telemetry redaction still applies on write.
            get_telemetry().warn(
                "research", "provider_output_unparseable",
                provider=self.name,
                error=str(exc)[:300],
                output_len=len(text),
                output_head=text[:_SAMPLE_CHARS],
                output_tail=text[-_SAMPLE_CHARS:] if len(text) > _SAMPLE_CHARS else "",
            )
            raise


def build_providers(config: SparkConfig) -> list[ResearchProvider]:
    scrub = frozenset(
        var for rt in config.runtimes.values() for var in rt.server.secret_env
    )
    return [CliAgentProvider(d, scrub) for d in config.research.providers]
