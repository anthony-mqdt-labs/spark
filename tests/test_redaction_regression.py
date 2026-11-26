"""Regression tests for the shared secret-redaction registry.

The registry used to live as private state inside telemetry/jsonl.py, which meant
the research prompt guard depended on a logger's internals and the per-server log
file had no redactor at all. These tests pin the public seam instead.
"""

from __future__ import annotations

import pytest

from spark.redaction import (
    REDACTED,
    is_sensitive_key,
    redact,
    register_secret_value,
    reset_secret_values,
    scrub_string,
    secret_values,
)
from spark.research.guard import SecretLeakError, assert_no_secrets


@pytest.fixture(autouse=True)
def isolated_registry():
    """Keep registration state from leaking between tests (and restore it)."""
    before = secret_values()
    reset_secret_values()
    yield
    reset_secret_values()
    for value in before:
        register_secret_value(value)


def test_registered_value_is_scrubbed_from_free_text():
    register_secret_value("hf_longlivedtoken123")
    assert scrub_string("Authorization: Bearer hf_longlivedtoken123") == (
        f"Authorization: Bearer {REDACTED}"
    )


def test_short_secret_is_also_scrubbed():
    # The old logger skipped values shorter than four characters entirely.
    register_secret_value("xy")
    assert scrub_string("hf_token=xy&other=1") == f"hf_token={REDACTED}&other=1"


def test_short_secret_does_not_break_an_ordinary_research_prompt():
    # A one-character token must not make every prompt containing "max" or
    # "context" fail the guard.
    register_secret_value("x")
    assert_no_secrets("Suggest flags for max tokens and context length")


def test_short_secret_is_still_detected_when_standalone():
    register_secret_value("x")
    with pytest.raises(SecretLeakError):
        assert_no_secrets("the token is x here")


def test_short_secret_does_not_mangle_ordinary_log_text():
    register_secret_value("1")
    assert scrub_string("Llama-3.1-8B q4") == "Llama-3.1-8B q4"


def test_short_secret_is_scrubbed_in_a_credential_context():
    register_secret_value("1")
    assert scrub_string("hf_token=1 failed") == f"hf_token={REDACTED} failed"


def test_credential_pattern_is_scrubbed_without_prior_registration():
    text = "error: hf_abcdefghijklmnopqrstuvwxyz was rejected"
    assert "hf_abcdefghijklmnopqrstuvwxyz" not in scrub_string(text)
    assert REDACTED in scrub_string(text)


def test_auth_key_is_sensitive_but_author_and_count_keys_are_not():
    assert is_sensitive_key("auth")
    assert is_sensitive_key("x_auth_token")
    assert is_sensitive_key("hf_token")
    assert not is_sensitive_key("author")
    assert not is_sensitive_key("prompt_tokens")


def test_redact_masks_nested_sensitive_keys_and_scrubs_strings():
    register_secret_value("supersecretvalue")
    result = redact({"auth": "opaque", "items": [{"hf_token": "opaque"}, "use supersecretvalue"]})
    assert result == {
        "auth": REDACTED,
        "items": [{"hf_token": REDACTED}, f"use {REDACTED}"],
    }


def test_prompt_guard_reads_the_shared_registry():
    register_secret_value("leakme123456")
    with pytest.raises(SecretLeakError):
        assert_no_secrets("model card says: leakme123456")


def test_telemetry_reexports_register_for_compatibility():
    from spark.telemetry import register_secret_value as telemetry_register

    telemetry_register("compatvalue123")
    assert "compatvalue123" in secret_values()
