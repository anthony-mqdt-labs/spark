"""Typed errors for spark.

Every failure mode is a ``SparkError`` subclass that carries actionable
remediation. The CLI renders these as ``{what, why, do-this-next}`` instead of a
bare traceback, per the project's hardened-failure rule.

Error codes are stable, machine-greppable strings (used in logs and tests). Keep
them sorted by domain prefix: ``CFG_``, ``SEC_``, ``RT_``, ``MODEL_``, ``HOST_``,
``RUN_``, ``NET_``. Every subclass is defined here — domain modules re-export
their errors, never define new ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class SparkError(Exception):
    """Base class for all spark failures.

    Attributes:
        message: Human-readable summary of *what* went wrong.
        code: Stable machine-readable identifier (e.g. ``MODEL_NOT_FOUND``).
        remediation: Ordered, concrete steps the operator can take next.
        context: Structured, non-secret detail for logs (never put secrets here).
        cause: Optional underlying exception.
    """

    code: str = "SPARK_ERROR"
    exit_code: int = 1
    #: Generic do-this-next fallback; domain subclasses pin specific steps.
    default_remediation: tuple[str, ...] = (
        "Read the error context and correct the input.",
        "Run `spark doctor` to check the host and configuration.",
    )

    def __init__(
        self,
        message: str,
        *,
        remediation: list[str] | None = None,
        context: dict[str, object] | None = None,
        cause: BaseException | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.remediation = remediation or list(self.default_remediation)
        self.context = context or {}
        self.cause = cause
        if code is not None:
            self.code = code

    def as_log(self) -> dict[str, object]:
        """Render for structured logging (no secrets — context is caller's duty)."""
        return {
            "error_type": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "context": self.context,
            "cause": repr(self.cause) if self.cause else None,
        }


# --- configuration -------------------------------------------------------------
class ConfigError(SparkError):
    code = "CFG_INVALID"
    exit_code = 78  # EX_CONFIG


class ConfigNotFoundError(ConfigError):
    code = "CFG_NOT_FOUND"


class LaunchOverrideError(SparkError):
    """A launch override tried to seize a flag spark alone owns."""

    code = "CFG_OVERRIDE_FORBIDDEN"

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(
            f"launch override {key!r} is reserved: bind host, port and model path "
            "are owned by spark, not by research output.",
            remediation=[
                f"Remove {key!r} from launch_overrides.",
                "Set the bind host/port in config/defaults.toml (or a user override).",
                "Register the weights path in the model registry, not via --model.",
                "Discard the staged research: spark config review <model> --reject, "
                "then re-run spark research <model>.",
            ],
            context={"key": key},
        )


# --- secrets -------------------------------------------------------------------
class SecretError(SparkError):
    code = "SEC_ERROR"
    exit_code = 77  # EX_NOPERM


class SecretNotFoundError(SecretError):
    code = "SEC_NOT_FOUND"


class SecretBackendError(SecretError):
    code = "SEC_BACKEND"


class ProviderAuthError(SecretError):
    """A hosted provider rejected the stored API key (401/403)."""

    code = "SEC_REJECTED"
    default_remediation = (
        "Re-store a valid key: spark secret set <name>",
        "Check the key is active (not revoked, out of credit) in the provider dashboard.",
    )


class SecretLeakError(SparkError):
    """The research guard caught secret material in an outbound prompt."""

    code = "SEC_PROMPT_LEAK"
    exit_code = 77


# --- host / probe --------------------------------------------------------------
class HostError(SparkError):
    code = "HOST_ERROR"


class UnsupportedHostError(HostError):
    code = "HOST_UNSUPPORTED"


# --- runtimes / backends -------------------------------------------------------
class RuntimeBackendError(SparkError):
    code = "RT_ERROR"


class BackendUnavailableError(RuntimeBackendError):
    code = "RT_UNAVAILABLE"


# --- models --------------------------------------------------------------------
class ModelError(SparkError):
    code = "MODEL_ERROR"


class ModelNotFoundError(ModelError):
    code = "MODEL_NOT_FOUND"
    exit_code = 69  # EX_UNAVAILABLE


class AmbiguousModelError(ModelError):
    code = "MODEL_AMBIGUOUS"


class AlreadyRegisteredError(ModelError):
    """The weights are already registered — there is nothing to adopt."""

    code = "MODEL_ALREADY_REGISTERED"


class WeightsMissingError(ModelError):
    """The entry asserts weights that are not on this host.

    Launch would otherwise hand the runtime a repo id, which quietly fetches
    several gigabytes at the first request — after the health gate has already
    passed. Fail early, with the two real options.
    """

    code = "MODEL_WEIGHTS_MISSING"
    exit_code = 69  # EX_UNAVAILABLE


# --- resource budgets ----------------------------------------------------------
class ResourceError(SparkError):
    code = "RUN_RESOURCE"
    exit_code = 75  # EX_TEMPFAIL


class InsufficientMemoryError(ResourceError):
    code = "RUN_OOM_RISK"


class InsufficientDiskError(ResourceError):
    code = "RUN_DISK"


class PortInUseError(ResourceError):
    code = "RUN_PORT_IN_USE"


# --- run / supervisor ----------------------------------------------------------
class LaunchError(SparkError):
    code = "RUN_LAUNCH"


class ProviderUnreachableError(SparkError):
    """A hosted provider's API could not be reached (DNS, TLS, timeout, 5xx)."""

    code = "NET_UNREACHABLE"
    exit_code = 75  # EX_TEMPFAIL
    default_remediation = (
        "Check network connectivity, then retry.",
        "Check the provider's status page; a 5xx is theirs, not spark's.",
    )


class HealthCheckError(SparkError):
    code = "RUN_HEALTHCHECK"


class RecoveryExhaustedError(SparkError):
    code = "RUN_RECOVERY_EXHAUSTED"


class CatalogError(SparkError):
    """The live catalog could not be built or published.

    A catalog failure is a filesystem/registry failure, not a bad host probe —
    keep it distinct from ``HostError`` so callers can tell the two apart.
    """

    code = "RUN_CATALOG"


# --- research ------------------------------------------------------------------
_RESEARCH_REMEDIATION: dict[str, tuple[str, ...]] = {
    "NET_RESEARCH": (
        "Check that the research agent CLI is installed and authenticated.",
        "Inspect the research events under the data dir's logs/ for the provider error.",
        "Or import a config manually: spark config import <model> < result.json",
    ),
    "NET_RESEARCH_UNAVAILABLE": (
        "Install/enable an agent CLI (e.g. `claude`).",
        "Or import a config manually: spark config import <model> < result.json",
    ),
    "NET_RESEARCH_EXHAUSTED": (
        "Inspect the research events under the data dir's logs/ for per-provider errors.",
        "Import config manually: spark config import <model> < result.json",
    ),
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
            remediation=remediation
            or list(
                _RESEARCH_REMEDIATION.get(effective_code, _RESEARCH_REMEDIATION["NET_RESEARCH"])
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


@dataclass
class Remediation:
    """Helper to build a tidy, deduplicated remediation list."""

    steps: list[str] = field(default_factory=list)

    def add(self, step: str) -> "Remediation":
        if step and step not in self.steps:
            self.steps.append(step)
        return self

    def build(self) -> list[str]:
        return list(self.steps)
