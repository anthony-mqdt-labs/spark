"""Pydantic models for spark configuration.

These are the *only* place config shape is defined. Code consumes validated
instances; it never parses TOML directly. ``extra="forbid"`` everywhere so a typo
or an injected key in a config file is a hard error, not silent behavior.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ModelFormat = Literal["mlx", "mlx-vlm", "gguf", "ollama", "safetensors", "any"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


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


class ServerSpec(_Strict):
    """Launch template for an OpenAI-compatible (or proxied) inference server.

    ``args_template`` tokens are formatted with a context dict containing at least:
    ``model_path``, ``model_id``, ``host``, ``port``. Unknown placeholders raise.
    """

    binary: str
    args_template: list[str] = Field(default_factory=list)
    openai_compatible: bool = True
    default_port: int = 8080
    health_path: str = "/v1/models"
    ready_timeout_s: float = 120.0
    # Daemon-style backends (e.g. ollama): if the endpoint is already healthy,
    # attach to it instead of spawning, and never terminate it on shutdown.
    attach_if_running: bool = False
    # Extra non-secret env for the child process.
    env: dict[str, str] = Field(default_factory=dict)
    # Map of CHILD_ENV_VAR -> secret name in the vault. Resolved at spawn time and
    # injected into the child env only. Never logged, never written to disk.
    secret_env: dict[str, str] = Field(default_factory=dict)


class RuntimeDef(_Strict):
    """A runtime backend definition (loaded from config/runtimes/<name>.toml)."""

    name: str
    description: str = ""
    model_format: ModelFormat = "any"
    detect: DetectSpec
    server: ServerSpec
    # Host gating. Empty list = no constraint.
    requires_os: list[str] = Field(default_factory=list)   # e.g. ["Darwin"]
    requires_arch: list[str] = Field(default_factory=list)  # e.g. ["arm64"]
    # Shown when unavailable, to guide the operator.
    install_hint: str = ""
    unavailable_reason: str = ""


class MemoryPolicy(_Strict):
    """Unified-memory budgeting (critical on a 16 GB Apple Silicon box)."""

    # Fraction of total unified memory considered safely usable for weights+kv.
    usable_fraction: float = 0.65
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
    overhead_multiplier: float = 1.20


class DiskPolicy(_Strict):
    # Refuse downloads that would drop free space below this many GiB.
    min_free_gib: float = 10.0
    enforce: bool = True


class GeneralConfig(_Strict):
    host: str = "127.0.0.1"            # bind localhost only by default (no LAN exposure)
    port_range: tuple[int, int] = (8080, 8099)
    log_retention_days: int = 7
    # Backend preference, highest priority first. Used to pick among capable
    # runtimes for a given model on this host.
    preference: list[str] = Field(
        default_factory=lambda: ["mlx_lm", "mlx_vlm", "llama_cpp", "omlx", "ollama"]
    )


class SupervisorPolicy(_Strict):
    max_restarts: int = 3
    backoff_base_s: float = 1.0
    backoff_max_s: float = 30.0
    health_interval_s: float = 5.0
    shutdown_grace_s: float = 10.0


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
    providers: list[ResearchProviderDef] = Field(default_factory=list)


class SparkConfig(_Strict):
    """Top-level resolved configuration handed to business logic."""

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    memory: MemoryPolicy = Field(default_factory=MemoryPolicy)
    disk: DiskPolicy = Field(default_factory=DiskPolicy)
    supervisor: SupervisorPolicy = Field(default_factory=SupervisorPolicy)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
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

    backend: str = "mlx_vlm"                                        # runtime implementing DFlash
    launch_overrides: dict[str, str] = Field(default_factory=dict)  # drafter flags (--draft-model, …)


class ModelEntry(_Strict):
    """A registered model (one TOML per model under the data dir)."""

    id: str = Field(..., description="Short slug used on the CLI, e.g. 'ornith9b'.")
    hf_repo: str = ""
    path: str = ""                       # local snapshot dir or GGUF file
    backend: str = ""                    # chosen runtime name
    model_format: ModelFormat = "any"
    quant: str = ""                      # e.g. 'q4'
    params_billions: float | None = None
    size_bytes: int | None = None
    context_length: int | None = None
    # Per-model launch overrides merged onto the runtime's args at spawn.
    launch_overrides: dict[str, str] = Field(default_factory=dict)
    # Research provenance.
    research_status: Literal["pending", "staged", "registered", "manual"] = "pending"
    aliases: list[str] = Field(default_factory=list)
    notes: str = ""
    # Opt-in speculative-decoding profile; None = model has no DFlash path (--dflash errors).
    dflash: DFlashProfile | None = None

    def with_dflash(self) -> "ModelEntry":
        """Copy configured for DFlash: the profile's backend, with its drafter
        overrides merged over the model's own. Caller ensures ``dflash`` is set."""
        assert self.dflash is not None, "with_dflash() requires a dflash profile"
        p = self.dflash
        return self.model_copy(
            update={
                "backend": p.backend or self.backend,
                "launch_overrides": {**self.launch_overrides, **p.launch_overrides},
            }
        )
