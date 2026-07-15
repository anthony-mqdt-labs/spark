"""Research request/response types (pydantic-validated agent output)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


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

    @field_validator("launch_overrides", mode="before")
    @classmethod
    def _coerce_scalar_values(cls, v: object) -> object:
        # Agents emit bare JSON numbers/bools for numeric flags (max_tokens: 4096,
        # num_ctx: 32768) despite the prompt's string hint. Values only ever become
        # CLI argv strings, so scalar -> str is lossless; anything else (list,
        # dict, null) still fails validation loudly.
        if not isinstance(v, dict):
            return v
        out: dict[object, object] = {}
        for k, val in v.items():
            if isinstance(val, bool):
                out[k] = "true" if val else "false"
            elif isinstance(val, (int, float)):
                out[k] = str(val)
            else:
                out[k] = val
        return out


class ResearchOutput(_Strict):
    """The schema agents must return (after the secrets-guard, before staging)."""

    recommended_backend: str
    per_runtime: dict[str, RuntimeRec] = Field(default_factory=dict)
    notes: str = ""

    def recommended(self) -> RuntimeRec | None:
        return self.per_runtime.get(self.recommended_backend)
