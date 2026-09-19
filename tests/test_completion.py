"""Tab completion must offer what this host can actually do.

The 2026-09-18 follow-up this pins down: `spark <TAB>` listed registry entries
whose weights had been deleted, so the menu advertised models that cannot
launch, and the MISSING marker rode along in a description the fuzzy popup
renders as an afterthought.

Availability still governs the menu — but *visible* is not *offered*. An entry
whose weights are provably absent is never a `run` candidate, while it stays
nameable where naming it is the whole point: `spark forget` is how such an
entry is removed, and `spark research`/`spark download` address it by name.
"""

from __future__ import annotations

from click.testing import CliRunner

from spark.cli.completion import (
    MISSING_MARK,
    complete_command,
    completion_command,
    completion_models,
)
from spark.config.schema import ModelEntry
from spark.registry import save_model

GOOD = "live-9b"
GONE = "gone-7b"
UNKNOWN = "opaque-3b"


def _seed(paths) -> None:
    """One runnable entry, one whose weights are provably gone, one unverifiable."""
    weights = paths.model_store_dir / GOOD
    weights.mkdir(parents=True, exist_ok=True)
    (weights / "model.safetensors").write_bytes(b"x" * 32)
    save_model(
        ModelEntry(
            id=GOOD,
            path=str(weights),
            backend="mlx_lm",
            quant="q4",
            model_format="mlx",
        ),
        paths,
    )
    # The reported failure: the registry entry outlived the weights directory.
    save_model(
        ModelEntry(
            id=GONE,
            path=str(paths.model_store_dir / GONE),
            hf_repo="org/Gone-7B",
            backend="mlx_lm",
            model_format="mlx",
        ),
        paths,
    )
    save_model(ModelEntry(id=UNKNOWN, model_format="mlx"), paths)


def _ids(candidates) -> list[str]:
    return [model_id for model_id, _desc in candidates]


def _desc(candidates, model_id: str) -> str:
    return dict(candidates)[model_id]


# --- the data source ----------------------------------------------------------
def test_run_context_hides_the_entry_whose_weights_are_gone(paths):
    _seed(paths)
    assert _ids(completion_models(paths, "run")) == [GOOD, UNKNOWN]


def test_default_context_is_the_runnable_list(paths):
    """A bare `spark <TAB>` must not offer something that cannot launch."""
    _seed(paths)
    assert _ids(completion_models(paths, "")) == _ids(completion_models(paths, "run"))


def test_forget_still_names_the_entry_that_must_be_removed(paths):
    """`spark forget` is how a stale entry is cleaned, so it must be nameable."""
    _seed(paths)
    candidates = completion_models(paths, "forget")
    assert set(_ids(candidates)) == {GOOD, GONE, UNKNOWN}
    assert MISSING_MARK in _desc(candidates, GONE)
    assert MISSING_MARK not in _desc(candidates, GOOD)


def test_administrative_contexts_keep_naming_the_missing_entry(paths):
    _seed(paths)
    for context in ("research", "download"):
        assert GONE in _ids(completion_models(paths, context))


def test_the_marker_is_the_only_thing_added_to_a_missing_description(paths):
    _seed(paths)
    run_desc = _desc(completion_models(paths, "run"), GOOD)
    forget_desc = _desc(completion_models(paths, "forget"), GONE)
    assert run_desc == "mlx_lm · q4 · mlx"
    assert forget_desc == f"mlx_lm · mlx · {MISSING_MARK}"


def test_an_empty_registry_is_not_an_error(paths):
    assert completion_models(paths, "run") == []
    assert completion_models(paths, "forget") == []


# --- the emitted script -------------------------------------------------------
def test_script_passes_the_subcommand_as_context_past_position_two():
    """`spark run <TAB>` must ask for the launch list; bare `spark <TAB>` too."""
    script = CliRunner().invoke(completion_command, ["zsh"]).output
    assert "__complete models" in script
    # The context is the subcommand, but only once one has been typed — at
    # position 2 the word being completed is the subcommand itself.
    assert "CURRENT > 2" in script
    assert '"${words[2]}"' in script


# --- end to end through the hidden CLI command --------------------------------
def test_cli_offers_only_launchable_models_for_run(paths, monkeypatch):
    _seed(paths)
    monkeypatch.setattr("spark.cli.completion.resolve_paths", lambda: paths)
    result = CliRunner().invoke(complete_command, ["models", "run"])
    assert result.exit_code == 0
    listed = [line.split("\t")[0] for line in result.output.splitlines() if line]
    assert listed == [GOOD, UNKNOWN]


def test_cli_keeps_the_missing_entry_for_forget_with_the_marker(paths, monkeypatch):
    _seed(paths)
    monkeypatch.setattr("spark.cli.completion.resolve_paths", lambda: paths)
    result = CliRunner().invoke(complete_command, ["models", "forget"])
    rows = dict(
        line.split("\t", 1) for line in result.output.splitlines() if "\t" in line
    )
    assert GONE in rows
    assert MISSING_MARK in rows[GONE]


def test_cli_stays_silent_when_paths_cannot_be_resolved(monkeypatch):
    """Completion must never surface an error into the shell."""

    def _boom():
        raise RuntimeError("no paths for you")

    monkeypatch.setattr("spark.cli.completion.resolve_paths", _boom)
    result = CliRunner().invoke(complete_command, ["models", "run"])
    assert result.exit_code == 0
    assert result.output == ""
