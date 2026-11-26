"""Pydantic models for spark configuration.

These are the *only* place config shape is defined. Code consumes validated
instances; it never parses TOML directly. ``extra="forbid"`` everywhere so a typo
or an injected key in a config file is a hard error, not silent behavior.
"""

from __future__ import annotations

import os
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..errors import LaunchOverrideError as LaunchOverrideError  # re-export: was defined here
from ..errors import ModelError

ModelFormat = Literal["mlx", "mlx-vlm", "gguf", "ollama", "safetensors", "prism", "any"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


_SEG = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
#: A registered id: one safe path component (it names the registry TOML).
MODEL_ID_RE = re.compile(rf"^{_SEG}$")
#: Any entry id, including the ephemeral ``owner/name`` ids synthesized for
#: unregistered hub repos. Never ``..``; at most one slash.
ENTRY_ID_RE = re.compile(rf"^{_SEG}(/{_SEG})?$")


def check_model_id(v: str, *, allow_owner: bool = False) -> str:
    rx = ENTRY_ID_RE if allow_owner else MODEL_ID_RE
    if not rx.match(v or "") or ".." in v:
        raise ValueError(
            f"invalid model id {v!r}: letters/digits/_/.- only, start alphanumeric, "
            "<=128 chars, no '..'"
        )
    return v


def id_file_stem(model_id: str) -> str:
    """Filename-safe form of an entry id (``owner/name`` -> ``owner--name``, the
    HF-cache convention), for per-model files outside the registry."""
    return check_model_id(model_id, allow_owner=True).replace("/", "--")


#: Keys that must never come from research output (the card is untrusted text).
#: Spark owns the bind host/port from bundled TOML and the model path from the
#: registry: an appended ``--host``, ``--port`` or ``--model`` would silently
#: rebind the server or load weights from an attacker-chosen location while
#: still passing D11's "is this a real flag?" check.
_RESERVED_OVERRIDE_KEYS = frozenset({"host", "port", "model"})


def check_reserved_override_keys(v: dict[str, str]) -> dict[str, str]:
    """Reject overrides that would rebind the server or redirect its weights."""
    for k in v:
        if isinstance(k, str) and k.strip().lstrip("-").strip().lower() in _RESERVED_OVERRIDE_KEYS:
            raise LaunchOverrideError(k)
    return v


def check_override_keys(v: dict[str, str]) -> dict[str, str]:
    """launch_overrides keys become argv flags (D9). A key that is not a
    ``-``-prefixed flag would land as a stray positional argument."""
    check_reserved_override_keys(v)
    for k in v:
        if not isinstance(k, str) or not re.fullmatch(r"--?[A-Za-z0-9][A-Za-z0-9_.-]*", k):
            raise ValueError(f"launch override key {k!r} is not a CLI flag (expected --name)")
    return v


def _expand_binary(v: str) -> str:
    """Expand ``~`` and ``$VAR``/``${VAR}`` so runtime defs can reference
    machine-specific paths (e.g. a sibling project's venv) without hardcoding
    them in checked-in TOML. Bare command names (resolved on PATH) pass through
    unchanged."""
    return os.path.expandvars(os.path.expanduser(v))


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def check_provider_base_url(v: str) -> str:
    """The key rides on every request to this URL: it must be TLS, unless the
    upstream is this machine (test doubles). No query/fragment/userinfo."""
    if not v:
        return v
    from urllib.parse import urlsplit

    parts = urlsplit(v)
    host = parts.hostname or ""
    if parts.scheme == "https" and host:
        pass
    elif parts.scheme == "http" and host in _LOOPBACK_HOSTS:
        pass
    else:
        raise ValueError(
            f"provider_base_url {v!r} must be https://<host>/... (plain http only for loopback)"
        )
    if parts.query or parts.fragment or parts.username or parts.password:
        raise ValueError(f"provider_base_url {v!r} must not carry a query, fragment, or userinfo")
    return v.rstrip("/")


class DetectSpec(_Strict):
    """How to detect a runtime's presence and version."""

    binary: str = Field(..., description="Entrypoint resolved on PATH.")
    version_args: list[str] = Field(default_factory=lambda: ["--version"])
    # Some tools print version to stderr or need a different probe.
    version_timeout_s: float = 5.0
    # Distribution name to read via importlib.metadata in the runtime's OWN
    # interpreter — for console-script runtimes whose server has no --version
    # flag (mlx_lm.server, mlx_vlm.server). Tried before version_args when set.
    version_package: str = ""
    # Python modules the runtime needs at inference time but a bare install may
    # omit (e.g. mlx-vlm needs torch+torchvision for its image processors).
    # `spark doctor` verifies these against the runtime's OWN interpreter, and
    # `spark doctor --fix` installs them there (uv-tool runtimes only). Entries
    # are 'module' or 'module:pip-package' when the import name differs from the
    # PyPI name (e.g. 'PIL:pillow') — the mapping is always declared, never guessed.
    python_requires: list[str] = Field(default_factory=list)

    @field_validator("binary")
    @classmethod
    def _expand(cls, v: str) -> str:
        return _expand_binary(v)


class ServerSpec(_Strict):
    """Launch template for an OpenAI-compatible (or proxied) inference server.

    ``args_template`` tokens are formatted with a context dict containing at least:
    ``model_path``, ``model_id``, ``host``, ``port``. Unknown placeholders raise.

    For relay backends, ``binary`` and ``args_template`` are ignored; instead
    ``relay_base_url`` and ``relay_health_endpoint`` are used to proxy requests.
    """

    binary: str = ""
    args_template: list[str] = Field(default_factory=list)
    openai_compatible: bool = True
    default_port: int = Field(8080, ge=1, le=65535)
    health_path: str = "/v1/models"
    ready_timeout_s: float = Field(120.0, gt=0)
    # Daemon-style backends (e.g. ollama): if the endpoint is already healthy,
    # attach to it instead of spawning, and never terminate it on shutdown.
    attach_if_running: bool = False
    # Route the server behind shims/openai_filter_proxy.py: the proxy binds the
    # published port, spawns this server on an ephemeral loopback port, and
    # answers GET /v1/models from spark's inventory (registered ∩ present)
    # instead of the runtime's own cache scan. For runtimes whose listing
    # cannot be restricted any other way (mlx_lm has no --model-discovery).
    filter_models: bool = False
    # Extra non-secret env for the child process.
    env: dict[str, str] = Field(default_factory=dict)
    # Map of CHILD_ENV_VAR -> secret name in the vault. Resolved at spawn time and
    # injected into the child env only. Never logged, never written to disk.
    secret_env: dict[str, str] = Field(default_factory=dict)
    # Relay backend fields: external service base URL for proxying requests.
    relay_base_url: str = ""
    # Health endpoint path on the external relay service (default "/health").
    relay_health_endpoint: str = "/health"
    # Hosted-provider backends (OpenRouter, DeepSeek, ...): a non-empty base URL
    # makes spark launch shims/provider_proxy.py, a loopback proxy that injects
    # the key from secret_env SPARK_PROVIDER_API_KEY. https only (loopback http
    # allowed for tests). ``binary``/``args_template`` are ignored.
    provider_base_url: str = ""
    # GET with the key before launch: 401/403 fails fast instead of restarting.
    provider_auth_check_path: str = "/models"
    # Upstream socket timeout; reasoning models can stall long between tokens.
    provider_timeout_s: float = Field(600.0, gt=0)

    @field_validator("binary")
    @classmethod
    def _expand(cls, v: str) -> str:
        return _expand_binary(v)

    @field_validator("provider_base_url")
    @classmethod
    def _provider_url(cls, v: str) -> str:
        return check_provider_base_url(v)


class RuntimeDef(_Strict):
    """A runtime backend definition (loaded from config/runtimes/<name>.toml)."""

    name: str
    description: str = ""
    model_format: ModelFormat = "any"
    detect: DetectSpec
    server: ServerSpec
    # Host gating. Empty list = no constraint.
    requires_os: list[str] = Field(default_factory=list)  # e.g. ["Darwin"]
    requires_arch: list[str] = Field(default_factory=list)  # e.g. ["arm64"]
    # Shown when unavailable, to guide the operator.
    install_hint: str = ""
    unavailable_reason: str = ""
    # Whether this runtime needs spark-visible weights on disk to serve a model.
    # True for inference runtimes; False for runtimes that own their weights
    # (the darkcore router child, ollama's own daemon store, relays to a remote
    # service) — the launch preflight must not demand weights those never use.
    requires_local_weights: bool = True


class MemoryPolicy(_Strict):
    """Unified-memory budgeting (critical on a 16 GB Apple Silicon box)."""

    # Fraction of total unified memory considered safely usable for weights+kv.
    usable_fraction: float = Field(0.65, gt=0, le=1)
    # Refuse to launch if estimated footprint exceeds usable budget unless --force.
    enforce: bool = True
    # Bytes-per-param multipliers for rough footprint estimation by quant tag.
    bytes_per_param: dict[str, float] = Field(
        default_factory=lambda: {
            "f16": 2.0,
            "bf16": 2.0,
            "q8": 1.0,
            "q6": 0.75,
            "q5": 0.65,
            "q4": 0.5,
            "q3": 0.4,
            "q2": 0.3,
        }
    )
    # KV-cache and runtime overhead headroom multiplier on top of weights.
    overhead_multiplier: float = Field(1.20, ge=1)


class DiskPolicy(_Strict):
    # Refuse downloads that would drop free space below this many GiB.
    min_free_gib: float = Field(10.0, ge=0)
    enforce: bool = True


class ProbeConfig(_Strict):
    """Host-probe cache age; owned by bundled TOML, never a Python literal."""

    max_age_s: float = Field(..., gt=0)


class GeneralConfig(_Strict):
    """Host policy owned by ``config/defaults.toml``.

    Every field here is REQUIRED: a missing bundled TOML must fail loudly instead
    of silently binding a different port range (the 8080 default was never the
    consumer-pinned 8095 canonical endpoint).
    """

    host: str  # loopback only; no LAN exposure by default
    port_range: tuple[int, int]
    log_retention_days: int = Field(..., ge=1)
    # Backend preference, highest priority first. Used to pick among capable
    # runtimes for a given model on this host.
    preference: list[str]

    @field_validator("port_range")
    @classmethod
    def _ports(cls, v: tuple[int, int]) -> tuple[int, int]:
        lo, hi = v
        if not (1 <= lo <= hi <= 65535):
            raise ValueError(f"port_range {v} must satisfy 1 <= lo <= hi <= 65535")
        return v


class SupervisorPolicy(_Strict):
    max_restarts: int = Field(3, ge=0)
    backoff_base_s: float = Field(1.0, ge=0)
    backoff_max_s: float = Field(30.0, ge=0)
    health_interval_s: float = Field(5.0, gt=0)
    # Socket timeout for a single health probe (not the interval between them).
    health_probe_timeout_s: float = Field(..., gt=0)
    instance_heartbeat_s: float = Field(..., gt=0)
    rss_sample_ticks: int = Field(..., ge=1)
    server_log_roll_mib: int = Field(..., ge=1)
    shutdown_grace_s: float = Field(10.0, ge=0)
    # A crash after this much healthy uptime gets a fresh restart budget (and
    # backoff): it is a new incident, not the same failure recurring.
    stable_after_s: float = Field(300.0, gt=0)
    # After the health endpoint answers, force one minimal generation before
    # declaring the server ready. Lazy-loading runtimes (mlx_lm/mlx_vlm) start
    # their HTTP surface *before* the model is resident, and their /v1/models
    # answers from an HF-cache scan without loading anything — so a health probe
    # alone cannot tell "serving" from "merely breathing". A warmup generation
    # is the only honest readiness signal for them.
    verify_generation: bool = True


class ResearchProviderDef(_Strict):
    """A research agent in the fallback chain (claude -> hermes -> pi -> ...)."""

    name: str
    type: Literal["cli"] = "cli"
    # Command template; tokens may include {prompt} (only used when prompt_via=arg).
    command: list[str] = Field(default_factory=list)
    prompt_via: Literal["stdin", "arg"] = "stdin"
    # If set, the provider is only "available" when this binary is on PATH.
    detect_binary: str = ""
    enabled: bool = True
    # How to read the agent's stdout: raw text (find JSON) or a JSON envelope
    # (e.g. `claude --output-format json` -> take the .result field, then find JSON).
    parse: Literal["raw_json", "json_envelope"] = "raw_json"
    envelope_field: str = "result"
    timeout_s: float = 240.0


class ResearchConfig(_Strict):
    enabled: bool = True
    model_card_chars: int = 6000
    flag_probe_timeout_s: float = Field(..., gt=0)
    providers: list[ResearchProviderDef] = Field(default_factory=list)


class SparkConfig(_Strict):
    """Top-level resolved configuration handed to business logic."""

    # Required: the whole [general] table comes from bundled TOML, never from a
    # Python literal (config is data, not code).
    general: GeneralConfig
    probe: ProbeConfig
    # Lambdas (not class objects) so a type checker does not mistake them for
    # `Field`'s dict-validating factory overload of the same arity.
    memory: MemoryPolicy = Field(default_factory=lambda: MemoryPolicy())
    disk: DiskPolicy = Field(default_factory=lambda: DiskPolicy())
    supervisor: SupervisorPolicy = Field(default_factory=lambda: SupervisorPolicy())
    research: ResearchConfig = Field(default_factory=lambda: ResearchConfig())
    runtimes: dict[str, RuntimeDef] = Field(default_factory=dict)


# --- model registry ------------------------------------------------------------
class DFlashProfile(_Strict):
    """Opt-in DFlash speculative-decoding profile, activated by
    ``spark run <model> --dflash``.

    Stored separately from the model's *default* (plain) launch so DFlash is never
    the default path: on 16 GB-class Apple Silicon it is a net loss for 4-bit targets
    and OOMs for higher-precision ones (patchwork inference-bench result #002). It
    only pays off with RAM headroom for a high-precision target + drafter, hence
    opt-in until the host warrants it. Mirrors the model's own ``backend`` +
    ``launch_overrides`` shape so it composes through the normal launch path."""

    backend: str = "mlx_vlm"  # runtime implementing DFlash
    launch_overrides: dict[str, str] = Field(
        default_factory=dict
    )  # drafter flags (--draft-model, …)

    @field_validator("launch_overrides")
    @classmethod
    def _keys(cls, v: dict[str, str]) -> dict[str, str]:
        return check_override_keys(v)


class ModelEntry(_Strict):
    """A registered model (one TOML per model under the data dir)."""

    id: str = Field(..., description="Short slug used on the CLI, e.g. 'ornith9b'.")
    hf_repo: str = ""
    path: str = ""  # local snapshot dir or GGUF file
    backend: str = ""  # chosen runtime name
    model_format: ModelFormat = "any"
    quant: str = ""  # e.g. 'q4'
    params_billions: float | None = None
    size_bytes: int | None = None
    context_length: int | None = None
    # Per-model launch overrides merged onto the runtime's args at spawn.
    launch_overrides: dict[str, str] = Field(default_factory=dict)
    # Research provenance.
    research_status: Literal["pending", "staged", "registered", "manual"] = "pending"
    aliases: list[str] = Field(default_factory=list)
    notes: str = ""
    # Hosted-provider entries: the provider's model id published as the
    # endpoint's api_model_id (e.g. "deepseek-chat"). Clients may send any other.
    remote_model: str = ""
    # Opt-in speculative-decoding profile; None = model has no DFlash path (--dflash errors).
    dflash: DFlashProfile | None = None

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        return check_model_id(v, allow_owner=True)

    @field_validator("launch_overrides")
    @classmethod
    def _keys(cls, v: dict[str, str]) -> dict[str, str]:
        return check_override_keys(v)

    def with_dflash(self) -> "ModelEntry":
        """Copy configured for DFlash: the profile's backend, with its drafter
        overrides merged over the model's own. Caller ensures ``dflash`` is set."""
        if self.dflash is None:
            raise ModelError(
                f"'{self.id}' has no [dflash] profile to enable.",
                remediation=[
                    "Add a [dflash] section to the model's registry entry, or "
                    "run it without --dflash.",
                ],
                context={"model": self.id},
            )
        p = self.dflash
        return self.model_copy(
            update={
                "backend": p.backend or self.backend,
                "launch_overrides": {**self.launch_overrides, **p.launch_overrides},
            }
        )
