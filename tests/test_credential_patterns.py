"""Every credential shape must be caught by both consumers of the pattern table.

The research guard and the log scrubber share ``CREDENTIAL_PATTERNS``; a pattern
added to one side only would let a pasted key reach an LLM prompt or a log line.
"""

from __future__ import annotations

import pytest

from spark.redaction import CREDENTIAL_PATTERNS, REDACTED, scrub_string
from spark.research.guard import SecretLeakError, assert_no_secrets

# One concrete sample per pattern, in table order: hf_, sk-, AKIA, ghp_, xox*,
# bearer. Keeping an explicit count assertion guards against a new pattern
# landing in the table without a corresponding regression case.
_SAMPLES = (
    "hf_" + "a" * 24,
    "sk-" + "b" * 24,
    "AKIA" + "1234567890ABCDEF",
    "ghp_" + "c" * 24,
    "xoxb-" + "d" * 12,
    "Bearer " + "e" * 24,
)


def test_pattern_table_has_one_sample_per_entry():
    assert len(CREDENTIAL_PATTERNS) == len(_SAMPLES)
    for pattern, sample in zip(CREDENTIAL_PATTERNS, _SAMPLES, strict=True):
        assert pattern.search(sample), f"{pattern.pattern} does not match {sample!r}"


@pytest.mark.parametrize("sample", _SAMPLES)
def test_guard_rejects_every_credential_shape(sample):
    with pytest.raises(SecretLeakError) as exc:
        assert_no_secrets(f"research prompt says: {sample}")
    assert exc.value.context["pattern"]  # the named pattern is reported


@pytest.mark.parametrize("sample", _SAMPLES)
def test_scrubber_redacts_every_credential_shape(sample):
    out = scrub_string(f"research prompt says: {sample} tail")
    assert REDACTED in out
    assert sample not in out


@pytest.mark.parametrize(
    "text",
    [
        "prompt_tokens=12",
        "model v2.1.0 q4_k_m on mlx_lm, 16 GiB host",
    ],
)
def test_benign_numbers_and_versions_survive(text):
    assert_no_secrets(text)
    assert scrub_string(text) == text
