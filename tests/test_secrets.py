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
        try:
            store.delete(name)
        finally:
            # Independent of the code under test: a broken delete() must not
            # leak a test item into the real login keychain.
            import subprocess

            subprocess.run(
                ["security", "delete-generic-password", "-s", "spark", "-a", name],
                capture_output=True,
                check=False,
            )
    assert not store.exists(name)


def test_keychain_rejects_empty_value(tmp_path):
    from spark.secrets.keychain import KeychainStore

    store = KeychainStore(index_path=tmp_path / "idx.json")
    if not store.is_available():
        pytest.skip("Keychain not available")
    with pytest.raises(SecretError):
        store.set("spark_pytest_empty", "")


# --- CLI presentation (FakeStore only: never touch the real Keychain) ---------
def _cli_ctx(paths):
    from spark.cli.context import SparkCtx
    from spark.config.loader import load_config
    from spark.telemetry import Telemetry
    from helpers import FakeStore

    return SparkCtx(
        paths=paths,
        config=load_config(),
        secrets=FakeStore(),
        telemetry=Telemetry(paths.log_dir, session_id="test"),
        session_id="test",
    )


def test_secret_empty_state_hints_at_set(paths, monkeypatch):
    from click.testing import CliRunner

    from spark.cli.secret import secret_ls

    monkeypatch.setattr("spark.cli.secret.build_context", lambda: _cli_ctx(paths))
    out = CliRunner().invoke(secret_ls, []).output
    assert "no secrets stored" in out
    assert "→" in out and "spark secret set <name>" in out


def test_secret_set_and_remove_use_success_glyph(paths, monkeypatch):
    from click.testing import CliRunner

    from spark.cli.secret import secret_ls, secret_rm, secret_set

    ctx = _cli_ctx(paths)
    monkeypatch.setattr("spark.cli.secret.build_context", lambda: ctx)
    runner = CliRunner()
    stored = runner.invoke(secret_set, ["hf_token", "--stdin"], input="tok\n")
    assert stored.exit_code == 0, stored.output
    assert "✓" in stored.output and "hf_token" in stored.output
    assert "hf_token" in runner.invoke(secret_ls, []).output
    removed = runner.invoke(secret_rm, ["hf_token"])
    assert removed.exit_code == 0, removed.output
    assert "✓" in removed.output and "hf_token" in removed.output
