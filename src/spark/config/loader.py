"""Layered configuration loader.

Precedence (low -> high):
    1. bundled ``config/defaults.toml``
    2. bundled ``config/runtimes/*.toml``  (each file = one RuntimeDef)
    3. user ``$config_dir/config.toml``     (overrides + extra runtimes)

The merged mapping is validated into a :class:`SparkConfig`. TOML is read with
the stdlib ``tomllib``; no third-party TOML parser is pulled in.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from ..errors import ConfigError
from .paths import SparkPaths, bundled_config_dir, resolve_paths
from .schema import RuntimeDef, SparkConfig


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"Malformed TOML in {path.name}",
            code="CFG_INVALID",
            remediation=[
                f"Fix the syntax error in {path}",
                "Run `spark config validate` after editing.",
            ],
            context={"path": str(path)},
            cause=exc,
        ) from exc


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_runtime_defs(runtimes_dir: Path) -> dict[str, dict[str, Any]]:
    """Each ``*.toml`` in the runtimes dir is one RuntimeDef table."""
    defs: dict[str, dict[str, Any]] = {}
    if not runtimes_dir.is_dir():
        return defs
    for f in sorted(runtimes_dir.glob("*.toml")):
        raw = _load_toml(f)
        name = raw.get("name") or f.stem
        raw.setdefault("name", name)
        defs[name] = raw
    return defs


def load_config(paths: SparkPaths | None = None) -> SparkConfig:
    """Load and validate the resolved SparkConfig."""
    paths = paths or resolve_paths()
    bundled = bundled_config_dir()

    # 1. defaults
    merged: dict[str, Any] = _load_toml(bundled / "defaults.toml")

    # 2. bundled runtimes
    runtime_defs = _load_runtime_defs(bundled / "runtimes")

    # 3. user overrides
    user = _load_toml(paths.user_config_file)
    # User-supplied runtimes (inline table or [runtimes.<name>]) override bundled.
    user_runtimes = user.pop("runtimes", {})
    if isinstance(user_runtimes, dict):
        for name, body in user_runtimes.items():
            if isinstance(body, dict):
                body.setdefault("name", name)
                runtime_defs[name] = _deep_merge(runtime_defs.get(name, {}), body)
    merged = _deep_merge(merged, user)

    merged["runtimes"] = runtime_defs

    try:
        return SparkConfig.model_validate(merged)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError(
            "Configuration failed validation.",
            code="CFG_INVALID",
            remediation=[
                "Run `spark config validate` to see the offending fields.",
                f"Check user overrides in {paths.user_config_file}",
                f"Check bundled defaults in {bundled}",
            ],
            cause=exc,
        ) from exc


def load_runtime_def(name: str, paths: SparkPaths | None = None) -> RuntimeDef:
    cfg = load_config(paths)
    if name not in cfg.runtimes:
        raise ConfigError(
            f"Unknown runtime '{name}'.",
            code="CFG_NOT_FOUND",
            remediation=["List runtimes with `spark doctor`."],
        )
    return cfg.runtimes[name]
