"""Research request/response types (pydantic-validated agent output)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="ignore")  # agents may add chatter keys; ignore


class ResearchRequest(BaseModel):
    """Non-secret inputs handed to a research provider."""

    model_config = ConfigDict(extra="forbid")

    hf_repo: str
    host_summary: dict          # chip, memory, cores, arch — NEVER secrets
    runtime_names: list[str]
    model_card_excerpt: str = ""


class RuntimeRec(_Strict):
    """Per-runtime recommendation."""

    quant: str = ""
    launch_overrides: dict[str, str] = Field(default_factory=dict)
    context_length: int | None = None
    rationale: str = ""


class ResearchOutput(_Strict):
    """The schema agents must return (after the secrets-guard, before staging)."""

    recommended_backend: str
    per_runtime: dict[str, RuntimeRec] = Field(default_factory=dict)
    notes: str = ""

    def recommended(self) -> RuntimeRec | None:
        return self.per_runtime.get(self.recommended_backend)
