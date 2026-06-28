"""Runtime backend abstraction + registry.

All supported backends are server-launchers driven by a :class:`RuntimeDef`. The
generic implementation fills the TOML ``args_template`` and reports a health URL;
per-runtime subclasses override only the bits that differ (e.g. model-path
resolution). This keeps launch *flags* in config (data) and launch *logic* minimal.
"""

from __future__ import annotations

from typing import Type

from ..config.schema import ModelEntry, RuntimeDef, SparkConfig
from ..errors import ConfigError
from ..probe.host import RuntimeStatus


class _SafeMap(dict):
    """format_map mapping that raises a clear error on unknown placeholders."""

    def __missing__(self, key: str):  # noqa: D401
        raise KeyError(key)


def _fill(token: str, ctx: dict[str, object]) -> str:
    if "{" not in token:
        return token
    try:
        return token.format_map(_SafeMap(ctx))
    except KeyError as exc:
        raise ConfigError(
            f"Unknown placeholder {exc} in launch template token {token!r}.",
            code="CFG_INVALID",
            remediation=[
                "Valid placeholders: {model_path}, {model_dir}, {model_id}, "
                "{host}, {port}.",
            ],
        ) from exc


def override_flags(overrides: dict[str, str]) -> list[str]:
    """Render per-model launch_overrides as extra CLI flags.

    ``{"--max-tokens": "2048"}`` -> ``["--max-tokens", "2048"]``;
    a value of "" yields a bare boolean flag ``["--trust-remote-code"]``."""
    out: list[str] = []
    for k, v in overrides.items():
        out.append(k)
        if v not in ("", None):
            out.append(str(v))
    return out


class Backend:
    """Generic server backend driven entirely by its RuntimeDef."""

    def __init__(self, rt: RuntimeDef, config: SparkConfig) -> None:
        self.rt = rt
        self.config = config

    # -- identity --------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.rt.name

    @property
    def default_port(self) -> int:
        return self.rt.server.default_port

    @property
    def openai_compatible(self) -> bool:
        return self.rt.server.openai_compatible

    # -- model resolution ------------------------------------------------------
    def resolve_model_ref(self, entry: ModelEntry) -> str:
        """The string handed to the runtime as the model. Local path preferred;
        otherwise the HF repo id (most runtimes can fetch it)."""
        ref = entry.path or entry.hf_repo or entry.id
        return ref

    def resolve_model_dir(self, entry: ModelEntry) -> str:
        """Directory handed to runtimes that serve a *directory* of models (only
        ``{model_dir}`` templates use this). Default empty; staged by
        :meth:`prepare` in backends that need it (e.g. oMLX)."""
        return ""

    # -- launch ----------------------------------------------------------------
    def build_launch_cmd(
        self, entry: ModelEntry, host: str, port: int
    ) -> list[str]:
        ctx: dict[str, object] = {
            "model_path": self.resolve_model_ref(entry),
            "model_dir": self.resolve_model_dir(entry),
            "model_id": entry.id,
            "host": host,
            "port": port,
        }
        args = [_fill(tok, ctx) for tok in self.rt.server.args_template]
        return [self.rt.server.binary, *args, *override_flags(entry.launch_overrides)]

    def secret_env_map(self) -> dict[str, str]:
        return dict(self.rt.server.secret_env)

    def extra_env(self) -> dict[str, str]:
        return dict(self.rt.server.env)

    # -- preparation -----------------------------------------------------------
    def prepare(self, entry: ModelEntry, *, secrets=None) -> None:
        """Hook run before launch (e.g. pull a model, verify weights). Default
        no-op. Raise a SparkError with remediation to abort the launch."""
        return None

    @property
    def attach_if_running(self) -> bool:
        return self.rt.server.attach_if_running

    # -- health ----------------------------------------------------------------
    def health_url(self, host: str, port: int) -> str:
        path = self.rt.server.health_path
        if not path.startswith("/"):
            path = "/" + path
        return f"http://{host}:{port}{path}"

    def openai_base_url(self, host: str, port: int) -> str:
        """The base URL clients should point at."""
        if self.openai_compatible:
            return f"http://{host}:{port}/v1"
        return f"http://{host}:{port}"

    def ready_timeout_s(self) -> float:
        return self.rt.server.ready_timeout_s


# --- registry ------------------------------------------------------------------
_REGISTRY: dict[str, Type[Backend]] = {}


def register(name: str):
    def deco(cls: Type[Backend]) -> Type[Backend]:
        _REGISTRY[name] = cls
        return cls

    return deco


def get_backend(name: str, config: SparkConfig) -> Backend:
    if name not in config.runtimes:
        raise ConfigError(
            f"Unknown runtime '{name}'.",
            code="CFG_NOT_FOUND",
            remediation=["See available runtimes with `spark doctor`."],
        )
    rt = config.runtimes[name]
    cls = _REGISTRY.get(name, Backend)
    return cls(rt, config)


def select_backend_for(
    entry: ModelEntry,
    config: SparkConfig,
    available: list[str],
) -> str:
    """Pick the backend for a model: its registered choice if available, else the
    highest-preference available runtime whose model_format is compatible."""
    if entry.backend and entry.backend in available:
        return entry.backend

    def compatible(rt_name: str) -> bool:
        rt = config.runtimes.get(rt_name)
        if not rt:
            return False
        if entry.model_format in ("any", ""):
            return True
        return rt.model_format in ("any", entry.model_format)

    for cand in config.general.preference:
        if cand in available and compatible(cand):
            return cand
    # last resort: any available
    for cand in available:
        if compatible(cand):
            return cand
    from ..errors import BackendUnavailableError

    raise BackendUnavailableError(
        f"No available runtime can serve model '{entry.id}'.",
        remediation=[
            "Install a compatible runtime (see `spark doctor` for hints).",
            f"Model format is '{entry.model_format}'.",
        ],
        context={"model": entry.id, "available": available},
    )
