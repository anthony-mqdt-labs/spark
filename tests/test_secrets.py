from __future__ import annotations

import uuid

import pytest
from helpers import FakeStore

from spark.errors import SecretError
from spark.secrets.store import validate_name


def test_validate_name_rejects_injection():
    for bad in ["", "a b", "a;b", "a/b", "-x" * 100 + "z" * 100]:
        with pytest.raises(SecretError):
            validate_name(bad)


def test_validate_name_accepts_good():
    for good in ["hf_token", "openai.key", "my-secret_1"]:
        assert validate_name(good) == good


def test_resolve_env_skips_missing_and_includes_present():
    s = FakeStore()
    s.set("hf_token", "tok")
    out = s.resolve_env({"HF_TOKEN": "hf_token", "OTHER": "does_not_exist"})
    assert out == {"HF_TOKEN": "tok"}


# --- real Keychain integration (macOS only) ------------------------------------
def test_keychain_roundtrip(tmp_path):
    from spark.secrets.keychain import KeychainStore

    store = KeychainStore(index_path=tmp_path / "idx.json")
    if not store.is_available():
        pytest.skip("Keychain not available on this host")

    name = f"spark_pytest_{uuid.uuid4().hex[:8]}"
    secret = "round-trip-value-123"
    try:
        store.set(name, secret)
        assert store.exists(name)
        assert store.get(name) == secret
        assert name in store.list_names()
        # overwrite
        store.set(name, "new-value-456")
        assert store.get(name) == "new-value-456"
    finally:
        store.delete(name)
    assert not store.exists(name)


def test_keychain_rejects_empty_value(tmp_path):
    from spark.secrets.keychain import KeychainStore

    store = KeychainStore(index_path=tmp_path / "idx.json")
    if not store.is_available():
        pytest.skip("Keychain not available")
    with pytest.raises(SecretError):
        store.set("spark_pytest_empty", "")
