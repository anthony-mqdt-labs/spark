"""Research fallback chain: try providers in order until one succeeds."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import SparkError
from ..telemetry import get_telemetry
from .providers import ResearchProvider
from .types import ResearchOutput, ResearchRequest


@dataclass
class ChainResult:
    output: ResearchOutput
    provider: str


class ResearchChain:
    def __init__(self, providers: list[ResearchProvider]) -> None:
        self.providers = providers

    def run(self, request: ResearchRequest) -> ChainResult:
        log = get_telemetry()
        errors: list[str] = []
        attempted = 0
        for p in self.providers:
            if not p.is_available():
                log.info("research", "provider_skipped", provider=p.name,
                         reason="unavailable_or_disabled")
                continue
            attempted += 1
            log.info("research", "provider_attempt", provider=p.name, repo=request.hf_repo)
            try:
                out = p.research(request)
                log.info("research", "provider_success", provider=p.name,
                         recommended=out.recommended_backend)
                return ChainResult(output=out, provider=p.name)
            except Exception as exc:
                msg = f"{p.name}: {exc}"
                errors.append(msg)
                log.warn("research", "provider_failed", provider=p.name, error=str(exc)[:300])
                continue

        if attempted == 0:
            raise SparkError(
                "No research providers are available.",
                code="NET_RESEARCH_UNAVAILABLE",
                remediation=[
                    "Install/enable an agent CLI (e.g. `claude`).",
                    "Or import a config manually: spark config import <model> < result.json",
                ],
            )
        raise SparkError(
            f"All {attempted} research provider(s) failed.",
            code="NET_RESEARCH_EXHAUSTED",
            remediation=[
                "Inspect logs for per-provider errors.",
                "Import config manually: spark config import <model> < result.json",
            ],
            context={"errors": errors},
        )
