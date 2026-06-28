"""Secret vault: backend-agnostic interface + macOS Keychain implementation."""

from __future__ import annotations

from ..config.paths import SparkPaths, resolve_paths
from .keychain import KeychainStore
from .store import HF_TOKEN, SecretStore, validate_name

__all__ = [
    "SecretStore",
    "KeychainStore",
    "HF_TOKEN",
    "validate_name",
    "get_secret_store",
]


def get_secret_store(paths: SparkPaths | None = None) -> SecretStore:
    """Return the default secret store for this host (macOS Keychain)."""
    paths = paths or resolve_paths()
    index = paths.data_dir / "secrets-index.json"
    return KeychainStore(index_path=index)
