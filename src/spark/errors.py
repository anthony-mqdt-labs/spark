"""Typed errors for spark.

Every failure mode is a ``SparkError`` subclass that carries actionable
remediation. The CLI renders these as ``{what, why, do-this-next}`` instead of a
bare traceback, per the project's hardened-failure rule.

Error codes are stable, machine-greppable strings (used in logs and tests). Keep
them sorted by domain prefix: ``CFG_``, ``SEC_``, ``RT_``, ``MODEL_``, ``HOST_``,
``RUN_``, ``NET_``.
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
        self.remediation = remediation or []
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


# --- secrets -------------------------------------------------------------------
class SecretError(SparkError):
    code = "SEC_ERROR"
    exit_code = 77  # EX_NOPERM


class SecretNotFoundError(SecretError):
    code = "SEC_NOT_FOUND"


class SecretBackendError(SecretError):
    code = "SEC_BACKEND"


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


class HealthCheckError(SparkError):
    code = "RUN_HEALTHCHECK"


class RecoveryExhaustedError(SparkError):
    code = "RUN_RECOVERY_EXHAUSTED"


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
