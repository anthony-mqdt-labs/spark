"""Ready banner and stop messaging: compact, actionable, glyph-consistent.

No servers are launched here — only the presentation helpers are exercised.
"""

from __future__ import annotations

from types import SimpleNamespace

from spark.cli.run import _print_ready, _print_stopped


def _info(**overrides):
    info = dict(
        base_url="http://127.0.0.1:8095/v1",
        backend="mlx_lm",
        port=8095,
        elapsed_s=1.2,
        log_path="/logs/servers/m.log",
        note="",
        api_model_id="",
    )
    info.update(overrides)
    return SimpleNamespace(**info)


def test_ready_banner_shows_endpoint_model_and_api_id(capsys):
    _print_ready("bonsai-8b", "mlx_lm", _info(api_model_id="/models/bonsai"))
    out = capsys.readouterr().out
    assert "http://127.0.0.1:8095/v1" in out
    assert "bonsai-8b" in out and "mlx_lm" in out
    assert "API id:" in out and "/models/bonsai" in out


def test_ready_banner_is_compact_and_ends_with_a_stop_hint(capsys):
    _print_ready("m", "mlx_lm", _info())
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == ""  # one blank line of breathing room above the banner
    body = [line for line in lines[1:] if line.strip()]
    # version header, Local, Model, Logs, stop hint — no blank lines between rows
    assert len(body) == 5
    assert all(line.strip() for line in lines[1:-1])
    assert "→ stop" in body[-1] and "Ctrl+C" in body[-1]


def test_ready_banner_hides_api_id_when_it_equals_the_alias(capsys):
    _print_ready("m", "mlx_lm", _info(api_model_id="m"))
    out = capsys.readouterr().out
    assert "API id:" not in out


def test_stop_message_reports_a_clean_shutdown_with_next_step(capsys):
    _print_stopped("bonsai-8b", restarts=1)
    out = capsys.readouterr().out
    assert "✓" in out and "bonsai-8b stopped" in out
    assert "no spark-owned process left behind" in out
    assert "restarts: 1" in out
    assert "→" in out and "spark run bonsai-8b" in out
