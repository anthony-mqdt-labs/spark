"""Research request/response types (pydantic-validated agent output)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..errors import SparkError


_DEFAULT_REMEDIATION: dict[str, list[str]] = {
    "NET_RESEARCH": [
        "Check that the research agent CLI is installed and authenticated.",
        "Inspect the research events under the data dir's logs/ for the provider error.",
        "Or import a config manually: spark config import <model> < result.json",
    ],
    "NET_RESEARCH_UNAVAILABLE": [
        "Install/enable an agent CLI (e.g. `claude`).",
        "Or import a config manually: spark config import <model> < result.json",
    ],
    "NET_RESEARCH_EXHAUSTED": [
        "Inspect the research events under the data dir's logs/ for per-provider errors.",
        "Import config manually: spark config import <model> < result.json",
    ],
}


class ResearchError(SparkError):
    """Base for research-chain failures: every one tells the operator what next."""

    code = "NET_RESEARCH"

    def __init__(
        self,
        message: str,
        *,
        remediation: list[str] | None = None,
        context: dict[str, object] | None = None,
        cause: BaseException | None = None,
        code: str | None = None,
    ) -> None:
        effective_code = code or self.code
        super().__init__(
            message,
            remediation=remediation or list(
                _DEFAULT_REMEDIATION.get(effective_code, _DEFAULT_REMEDIATION["NET_RESEARCH"])
            ),
            context=context,
            cause=cause,
            code=code,
        )


class ResearchProviderError(ResearchError):
    """A provider process could not run or exited non-zero."""

    code = "NET_RESEARCH"


class ResearchUnavailableError(ResearchError):
    """No provider in the fallback chain was available."""

    code = "NET_RESEARCH_UNAVAILABLE"


class ResearchExhaustedError(ResearchError):
    """Every available provider failed; staged research is unchanged."""

    code = "NET_RESEARCH_EXHAUSTED"


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
