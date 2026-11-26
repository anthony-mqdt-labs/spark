"""One fuzzy rule must serve registry, run and adopt."""

from __future__ import annotations

from spark.registry.models import fuzzy_match


def test_fuzzy_match_prefers_exact_then_unique_prefix_then_substring():
    candidates = [("org/model-8b", "a"), ("org/other-8b", "b")]
    assert fuzzy_match("MODEL-8B", candidates)[0] == "a"
    assert fuzzy_match("model", candidates)[0] == "a"
    assert fuzzy_match("other", candidates)[0] == "b"
    assert fuzzy_match("8b", candidates)[1] == ["a", "b"]


def test_fuzzy_match_repo_tail_disambiguates_when_requested():
    # One name: the substring rule alone already finds the repo tail.
    single = [("prism-ml/Ternary-Bonsai-8B-mlx-2bit", "pack")]
    assert fuzzy_match("bonsai-8b", single)[0] == "pack"

    # Two substring hits, but only one tail starts with the query: with tails
    # enabled the unique tail wins instead of leaving an ambiguity.
    candidates = [("prism-ml/Bonsai-8B", "pack"), ("other-8b-bonsai", "noise")]
    assert fuzzy_match("bonsai", candidates)[1] == ["pack", "noise"]
    assert fuzzy_match("bonsai", candidates, tails=True)[0] == "pack"
