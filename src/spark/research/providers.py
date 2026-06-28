"""Research providers: CLI agents driven by config (claude, hermes, pi, ...)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from abc import ABC, abstractmethod

from ..config.schema import ResearchProviderDef, SparkConfig
from ..errors import SparkError
from .guard import assert_no_secrets
from .prompt import build_prompt
from .types import ResearchOutput, ResearchRequest

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


class CliAgentProvider(ResearchProvider):
    def __init__(self, defn: ResearchProviderDef) -> None:
        self.defn = defn
        self.name = defn.name

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
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SparkError(f"{self.name}: failed to run ({exc})", code="NET_RESEARCH") from exc

        if proc.returncode != 0:
            raise SparkError(
                f"{self.name}: exit {proc.returncode}: {(proc.stderr or '').strip()[:200]}",
                code="NET_RESEARCH",
            )

        text = proc.stdout
        if self.defn.parse == "json_envelope":
            envelope = json.loads(proc.stdout)
            text = envelope.get(self.defn.envelope_field, "")

        obj = _extract_json(text)
        return ResearchOutput.model_validate(obj)


def build_providers(config: SparkConfig) -> list[ResearchProvider]:
    return [CliAgentProvider(d) for d in config.research.providers]
