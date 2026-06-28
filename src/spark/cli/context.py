"""Shared CLI context: resolves paths, config, secrets, and telemetry once."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

from ..config.loader import load_config
from ..config.paths import SparkPaths, resolve_paths
from ..config.schema import SparkConfig
from ..secrets import get_secret_store
from ..secrets.store import SecretStore
from ..telemetry import Telemetry, init_telemetry


@dataclass
class SparkCtx:
    paths: SparkPaths
    config: SparkConfig
    secrets: SecretStore
    telemetry: Telemetry
    session_id: str


def build_context() -> SparkCtx:
    paths = resolve_paths().ensure()
    session_id = f"{uuid.uuid4().hex[:12]}"
    telemetry = init_telemetry(
        paths.log_dir,
        session_id,
        retention_days=7,
        echo_errors=bool(os.environ.get("SPARK_DEBUG")),
    )
    config = load_config(paths)
    telemetry.info(
        "cli", "session_start",
        version=_version(), argv_count=len(os.sys.argv),
    )
    secrets = get_secret_store(paths)
    return SparkCtx(
        paths=paths,
        config=config,
        secrets=secrets,
        telemetry=telemetry,
        session_id=session_id,
    )


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("spark")
    except Exception:
        return "0.0.0"
