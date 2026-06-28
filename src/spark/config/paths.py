"""Filesystem path resolution for spark.

Precedence for the spark root:
1. ``$SPARK_HOME`` (explicit override — useful for tests and isolation)
2. XDG base dirs (``$XDG_CONFIG_HOME`` / ``$XDG_DATA_HOME``) when set
3. ``~/.config/spark`` and ``~/.local/share/spark`` (operator's expected layout)

platformdirs is used only as the cross-platform cache fallback. Paths are
computed lazily and never created as a side effect of import.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import platformdirs

_APP = "spark"


def _xdg(env: str, default: Path) -> Path:
    val = os.environ.get(env)
    if val:
        return Path(val).expanduser()
    return default


@dataclass(frozen=True)
class SparkPaths:
    """Resolved spark directories. Call :meth:`ensure` to create them."""

    config_dir: Path
    data_dir: Path
    cache_dir: Path

    # --- derived locations ----------------------------------------------------
    @property
    def user_config_file(self) -> Path:
        return self.config_dir / "config.toml"

    @property
    def host_profile_file(self) -> Path:
        return self.data_dir / "host.toml"

    @property
    def models_dir(self) -> Path:
        """Per-model registry entries (one TOML per model)."""
        return self.data_dir / "models"

    @property
    def model_store_dir(self) -> Path:
        """Where downloaded model weights live (HF snapshots)."""
        return self.data_dir / "store"

    @property
    def staging_dir(self) -> Path:
        """Research output awaiting operator review before registration."""
        return self.data_dir / "staging"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def run_dir(self) -> Path:
        """PID/socket/state for running servers."""
        return self.data_dir / "run"

    def ensure(self) -> "SparkPaths":
        for d in (
            self.config_dir,
            self.data_dir,
            self.models_dir,
            self.model_store_dir,
            self.staging_dir,
            self.log_dir,
            self.run_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        # Lock down dirs that may hold sensitive material (run/) to user-only.
        try:
            os.chmod(self.run_dir, 0o700)
            os.chmod(self.data_dir, 0o700)
        except OSError:
            pass
        return self


def resolve_paths() -> SparkPaths:
    """Resolve spark paths from the environment without creating anything."""
    override = os.environ.get("SPARK_HOME")
    if override:
        root = Path(override).expanduser()
        return SparkPaths(
            config_dir=root / "config",
            data_dir=root / "data",
            cache_dir=root / "cache",
        )

    config_dir = _xdg("XDG_CONFIG_HOME", Path.home() / ".config") / _APP
    data_dir = _xdg("XDG_DATA_HOME", Path.home() / ".local" / "share") / _APP
    cache_dir = Path(platformdirs.user_cache_dir(_APP))
    return SparkPaths(config_dir=config_dir, data_dir=data_dir, cache_dir=cache_dir)


def bundled_config_dir() -> Path:
    """Locate the shipped, read-only config tree.

    Source checkout: ``<repo>/config``. Installed wheel: ``spark/_bundled_config``
    (see pyproject force-include)."""
    here = Path(__file__).resolve()
    # src/spark/config/paths.py -> repo root is parents[3]
    repo_config = here.parents[3] / "config"
    if repo_config.is_dir():
        return repo_config
    bundled = here.parents[1] / "_bundled_config"
    if bundled.is_dir():
        return bundled
    # Last resort: package-adjacent.
    return here.parents[1] / "config"
