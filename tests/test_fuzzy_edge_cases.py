"""Fuzzy resolution edge cases: no query, duplicate names, ambiguous tails.

Resolution is the single seam shared by run/adopt/completion, so a duplicate
alias must be an explicit ambiguity here rather than an arbitrary first hit.
"""

from __future__ import annotations

import inspect

import pytest

from spark.config.schema import ModelEntry
from spark.errors import AmbiguousModelError, ModelNotFoundError
from spark.registry import save_model
from spark.registry.models import fuzzy_match, resolve_model


def test_fuzzy_match_exposes_optional_tail_disambiguation():
    tails = inspect.signature(fuzzy_match).parameters["tails"]
    assert tails.kind is inspect.Parameter.KEYWORD_ONLY
    assert tails.default is False


def test_empty_query_matches_nothing():
    pair = ("a-model", 1)
    assert fuzzy_match("", [pair]) == (None, [])
    assert fuzzy_match("   ", [pair]) == (None, [])
    assert fuzzy_match("", []) == (None, [])


def test_duplicate_alias_is_ambiguous_not_first_hit(paths):
    save_model(ModelEntry(id="alpha", aliases=["shared"], model_format="mlx"), paths)
    save_model(ModelEntry(id="beta", aliases=["shared"], model_format="mlx"), paths)

    with pytest.raises(AmbiguousModelError) as exc:
        resolve_model("shared", paths)

    assert exc.value.code == "MODEL_AMBIGUOUS"
    assert set(exc.value.context["matches"]) == {"alpha", "beta"}
    advice = " ".join(exc.value.remediation)
    assert "alpha" in advice and "beta" in advice


def test_duplicate_exact_ids_across_files_are_ambiguous(paths, tmp_path):
    from spark.telemetry import init_telemetry

    init_telemetry(tmp_path / "logs", "fuzzy-duplicate-ids")
    first = save_model(ModelEntry(id="twin-one", hf_repo="org/A"), paths)
    # list_models deliberately keeps a mismatched file and warns about it;
    # two files claiming the same id must still surface as an ambiguity.
    (paths.models_dir / "twin-two.toml").write_text(first.read_text(encoding="utf-8"))

    with pytest.raises(AmbiguousModelError) as exc:
        resolve_model("twin-one", paths)
    assert exc.value.context["matches"].count("twin-one") == 2


def test_ambiguous_tail_returns_all_candidates():
    first, second = object(), object()
    names = [("large-8b-one", first), ("large-8b-two", second)]
    assert fuzzy_match("8b", names, tails=True) == (None, [first, second])
    assert fuzzy_match("8b", names, tails=False) == (None, [first, second])


def test_unique_tail_disambiguates_ambiguous_substring():
    first, second = object(), object()
    names = [("large/8b-one", first), ("large-8b-two", second)]
    # Both names contain "8b" (ambiguous substring), but only one *tail* starts
    # with it. Tail prefix is a subset of substring matches, yet its uniqueness
    # differs across candidates — so the tail rule is a real disambiguator.
    assert fuzzy_match("8b", names, tails=False) == (None, [first, second])
    assert fuzzy_match("8b", names, tails=True) == (first, [])


def test_empty_model_query_does_not_resolve_to_every_model(paths):
    save_model(ModelEntry(id="alpha", model_format="mlx"), paths)
    save_model(ModelEntry(id="beta", model_format="mlx"), paths)
    with pytest.raises(ModelNotFoundError) as exc:
        resolve_model("", paths)
    assert exc.value.context["query"] == ""
