"""Regression tests: hub-cache refs are data, never unchecked path components."""

from __future__ import annotations

import pytest

import spark.hf_cache as hf_cache


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_cache, "hf_cache_root", lambda: tmp_path)
    repo = tmp_path / "models--org--name"
    snaps = repo / "snapshots"
    refs = repo / "refs"
    snaps.mkdir(parents=True)
    refs.mkdir(parents=True)
    return tmp_path, repo, snaps, refs


def test_snapshot_ref_cannot_escape_the_cache(cache_root):
    _, _, snaps, refs = cache_root
    outside = snaps.parent.parent / "evil"
    outside.mkdir()
    (outside / "config.json").write_text("{}")
    (refs / "main").write_text("../../evil\n")

    result = hf_cache.snapshot_dir("org/name")
    assert result is None or result.resolve().is_relative_to(snaps.resolve())


def test_snapshot_ref_accepts_a_commit_hash(cache_root):
    _, _, snaps, refs = cache_root
    commit = "a" * 40
    (snaps / commit).mkdir()
    (snaps / commit / "config.json").write_text("{}")
    (refs / "main").write_text(commit + "\n")

    assert hf_cache.snapshot_dir("org/name") == snaps / commit


def test_snapshot_ref_rejects_non_commit_tokens(cache_root):
    _, _, snaps, refs = cache_root
    commit = "b" * 40
    (snaps / commit).mkdir()
    (snaps / commit / "config.json").write_text("{}")
    # A non-commit token is ignored; the fallback still picks a real snapshot.
    (refs / "main").write_text("latest")
    assert hf_cache.snapshot_dir("org/name") == snaps / commit


def test_repo_id_rejects_trailing_newline_and_unicode():
    assert not hf_cache.is_valid_repo_id("owner/name\n")
    assert not hf_cache.is_valid_repo_id("owner/näme")
    assert hf_cache.is_valid_repo_id("owner/name-1.2")
