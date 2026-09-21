"""Research provider chain for `spark download`.

Discovers optimal per-runtime launch config for a model on this host using an
ordered fallback chain of agent CLIs (claude -> hermes -> pi -> ...). Results are
STAGED for operator review, never auto-applied.

HARD CONSTRAINT: providers receive only the public model card + host facts +
runtime names. The secrets-guard (`guard.assert_no_secrets`) runs before any
prompt leaves the process; the code path never reads the secret vault.
"""

from __future__ import annotations

from ..config.schema import SparkConfig
from ..probe.host import HostProfile
from .chain import ChainResult, ResearchChain
from .guard import SecretLeakError, assert_no_secrets
from .prompt import build_request, host_summary
from .providers import CliAgentProvider, ResearchProvider, build_providers
from .staging import apply_output_to_entry, delete_staged, load_staged, stage
from .types import ResearchOutput, ResearchRequest, RuntimeRec
from .validate import known_flags, normalize_quant, validate_accept, validate_flags

__all__ = [
    "ResearchChain",
    "ChainResult",
    "ResearchProvider",
    "CliAgentProvider",
    "build_providers",
    "ResearchRequest",
    "ResearchOutput",
    "RuntimeRec",
    "build_request",
    "host_summary",
    "assert_no_secrets",
    "SecretLeakError",
    "stage",
    "load_staged",
    "delete_staged",
    "apply_output_to_entry",
    "normalize_quant",
    "known_flags",
    "validate_flags",
    "validate_accept",
    "run_research",
]


def run_research(
    hf_repo: str,
    profile: HostProfile,
    config: SparkConfig,
    runtime_names: list[str],
) -> ChainResult:
    """Build a (secret-free) request and run the provider chain."""
    request = build_request(hf_repo, profile, config, runtime_names)
    chain = ResearchChain(build_providers(config))
    return chain.run(request)
