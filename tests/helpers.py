"""Shared test doubles."""

from __future__ import annotations

from spark.errors import SecretNotFoundError
from spark.secrets.store import SecretStore, validate_name


class FakeStore(SecretStore):
    """In-memory secret store for tests (no Keychain dependency)."""

    backend_name = "fake"

    def __init__(self):
        self._d: dict[str, str] = {}

    def set(self, name, value):
        validate_name(name)
        self._d[name] = value

    def get(self, name):
        if name not in self._d:
            raise SecretNotFoundError(f"{name} missing")
        return self._register_for_redaction(self._d[name])

    def delete(self, name):
        self._d.pop(name, None)

    def list_names(self):
        return sorted(self._d)

    def exists(self, name):
        return name in self._d

    def is_available(self):
        return True
