"""Regression tests for the CLI's shared visual language."""

from __future__ import annotations

from click.testing import CliRunner

from spark.cli.app import cli
from spark.cli.render import (
    _research_label,
    fmt_size,
    hint,
    render_store_issues,
    short,
    success,
    warning,
)
from spark.registry.availability import LOCAL, Availability
from spark.registry.store import StoreIssue


def test_fmt_size_never_reports_small_files_as_zero_mib():
    assert fmt_size(0) == "—"
    assert fmt_size(512) == "512 B"
    assert fmt_size(1024) == "1 KiB"
    assert fmt_size(1536) == "2 KiB"
    assert fmt_size(3 * 2**20) == "3 MiB"
    assert fmt_size(int(1.5 * 2**30)) == "1.5 GiB"


def test_core_availability_labels_use_the_same_byte_format():
    avail = Availability(state=LOCAL, detail="in the spark model store", bytes_present=1536)
    assert avail.label == "local 2 KiB"


def test_short_truncates_with_ellipsis_and_keeps_short_text():
    assert short("short", 10) == "short"
    assert short("abcdefghijklmnop", 8) == "abcdefg…"


def test_root_help_has_a_quick_start_and_short_help_flag():
    runner = CliRunner()
    long = runner.invoke(cli, ["--help"])
    short = runner.invoke(cli, ["-h"])
    assert long.exit_code == short.exit_code == 0
    assert "Quick start:" in long.output
    assert "spark doctor" in long.output
    assert "staged, never applied" in long.output
    assert long.output == short.output


def test_working_uses_a_spinner_in_a_tty_and_a_plain_line_when_piped(monkeypatch, capsys):
    import spark.cli.render as render

    entered: list[str] = []

    class _FakeStatus:
        def __init__(self, message, **kwargs):
            entered.append(message)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class _TTY:
        is_terminal = True

        @staticmethod
        def status(message, **kwargs):
            return _FakeStatus(message, **kwargs)

    monkeypatch.setattr(render, "console", _TTY())
    with render.working("researching …"):
        pass
    assert entered == ["researching …"]

    monkeypatch.setattr(render, "console", __import__("rich").console.Console())
    with render.working("researching …"):
        pass
    assert "… researching …" in capsys.readouterr().out


def test_semantic_helpers_use_consistent_glyphs(capsys):
    success("adopted bonsai-8b")
    warning("1 model has no weights")
    hint("spark list", "see everything")
    out = capsys.readouterr().out
    assert "✓ adopted bonsai-8b" in out
    assert "! 1 model has no weights" in out
    assert "→ see everything: spark list" in out


def test_research_labels_are_compact_and_unambiguous():
    assert _research_label("registered") == "tuned"
    assert _research_label("manual") == "manual"
    assert _research_label("pending") == "pending"
    assert _research_label("staged") == "staged"  # awaiting explicit review, not pending research
    assert _research_label("") == "pending"


def test_store_issues_are_one_line_each_with_a_single_fix_hint(capsys, tmp_path):
    render_store_issues(
        [
            StoreIssue(
                path=tmp_path / "partial-id",
                model_id="partial-id",
                registered=False,
                hf_repo="org/Partial",
                incomplete_files=2,
                bytes_present=2**16,
            ),
            StoreIssue(
                path=tmp_path / "orphan",
                model_id="orphan",
                registered=False,
                hf_repo="",
                incomplete_files=0,
                bytes_present=0,
            ),
        ]
    )
    out = capsys.readouterr().out
    assert len([ln for ln in out.splitlines() if "•" in ln]) == 2
    assert "64 KiB on disk" in out
    assert "— on disk" in out
    assert out.count("→ fix:") == 1
    flat = " ".join(out.split())  # Rich may soft-wrap the one long hint line
    assert "spark download org/Partial" in flat
    assert "spark download <hf-repo> --id orphan" in flat
    assert "rm -rf" in flat
