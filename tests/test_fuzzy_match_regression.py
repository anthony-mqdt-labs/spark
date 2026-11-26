"""One fuzzy rule must serve registry, run and adopt."""

from __future__ import annotations

from spark.registry.models import fuzzy_match


def test_fuzzy_match_prefers_exact_then_unique_prefix_then_substring():
    candidates = [("org/model-8b", "a"), ("org/other-8b", "b")]
    assert fuzzy_match("MODEL-8B", candidates)[0] == "a"
    assert fuzzy_match("model", candidates)[0] == "a"
    assert fuzzy_match("other", candidates)[0] == "b"
    assert fuzzy_match("8b", candidates)[1] == ["a", "b"]


def test_fuzzy_match_repo_tail_when_requested():
    candidates = [("prism-ml/Ternary-Bonsai-8B-mlx-2bit", "pack")]
    assert fuzzy_match("bonsai-8b", candidates, tails=True)[0] == "pack"
    # The tail is a substring of the full repo id, so the base rule already
    # covers it; `tails` just makes that intent explicit at the call site.
    assert fuzzy_match("bonsai-8b", candidates)[0] == "pack"
