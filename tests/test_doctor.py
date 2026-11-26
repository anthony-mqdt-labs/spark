"""Rendering tests for `spark doctor`: compact, scannable diagnostics.

The complaint: long install hints in the table's `detail` column produced
6-line rows, and the trailing "Ready. Available: a, b, c…" line was a dense
comma-joined list. These tests pin the new shape without touching the underlying
profile or the --fix repair path.
"""

from __future__ import annotations

import time

import pytest
from click.testing import CliRunner

from spark.cli.doctor import doctor_command
from spark.probe.host import HostProfile, RuntimeStatus

LONG_HINT = (
    "uv tool install mlx-vlm --with torch --with torchvision --with pillow "
    "--with transformers --with sentencepiece --with accelerate --with huggingface-hub"
)

ALL_NAMES = ("mlx_lm", "ollama", "mlx_vlm", "vllm")


def _profile() -> HostProfile:
    return HostProfile(
        os="Darwin",
        arch="arm64",
        chip="Apple M2",
        total_memory_bytes=16 * 2**30,
        cpu_count=8,
        probed_at=time.time(),
        runtimes={
            "mlx_lm": RuntimeStatus(
                name="mlx_lm",
                available=True,
                path="/opt/homebrew/bin/mlx_lm.server",
                version="0.5.1",
            ),
            "ollama": RuntimeStatus(
                name="ollama", available=True, path="/opt/homebrew/bin/ollama", version="0.5.6"
            ),
            "mlx_vlm": RuntimeStatus(
                name="mlx_vlm",
                available=False,
                reason="'mlx_vlm.server' not found on PATH",
                install_hint=LONG_HINT,
            ),
            "vllm": RuntimeStatus(
                name="vllm",
                available=False,
                reason="vLLM has no performant Apple Silicon path; use mlx_lm/llama_cpp here.",
                install_hint="On a CUDA host: uv pip install vllm",
            ),
        },
    )


@pytest.fixture
def invoke_doctor(tmp_path, monkeypatch):
    """Isolated SPARK_HOME, stubbed probe and no real dep subprocesses."""
    monkeypatch.setenv("SPARK_HOME", str(tmp_path))
    monkeypatch.setattr("spark.cli.doctor.missing_python_deps", lambda *a, **k: [])

    def _invoke(profile: HostProfile, *args: str):
        monkeypatch.setattr("spark.cli.doctor.load_or_probe", lambda *a, **k: profile)
        return CliRunner().invoke(
            doctor_command,
            list(args),
            env={"SPARK_HOME": str(tmp_path)},
            catch_exceptions=False,
        )

    return _invoke


def _row_heights(output: str, names=ALL_NAMES) -> dict[str, int]:
    """Physical table lines occupied by each runtime's row.

    A row starts at the line naming its runtime; wrapped continuation lines also
    begin with ``│``, so each row is bounded by the next runtime's line (or the
    bottom border ``└`` for the last row).
    """
    lines = output.splitlines()
    starts: dict[str, int] = {}
    for i, line in enumerate(lines):
        for name in names:
            if name in line and name not in starts:
                starts[name] = i
    ordered = sorted(starts.items(), key=lambda kv: kv[1])
    heights: dict[str, int] = {}
    for idx, (name, start) in enumerate(ordered):
        end = ordered[idx + 1][1] if idx + 1 < len(ordered) else len(lines)
        height = 0
        for line in lines[start:end]:
            if line.lstrip().startswith("│"):
                height += 1
            elif height:
                break
        heights[name] = height
    return heights


def test_runtime_rows_are_compact(invoke_doctor):
    output = invoke_doctor(_profile()).output
    heights = _row_heights(output)
    assert set(heights) == set(ALL_NAMES)
    assert all(1 <= h <= 2 for h in heights.values()), heights


def test_next_steps_names_unavailable_runtimes_with_actions(invoke_doctor):
    output = invoke_doctor(_profile()).output
    assert "Next steps" in output
    after = output.split("Next steps", 1)[1]
    assert "mlx_vlm" in after and "vllm" in after  # unavailable runtimes named
    assert "uv tool install mlx-vlm" in after  # command hint, actionable
    assert "install mlx_lm" not in after  # available runtime not listed


def test_next_steps_uses_advisory_line_for_prose_hints(invoke_doctor):
    output = invoke_doctor(_profile()).output
    after = output.split("Next steps", 1)[1]
    advisory = [ln for ln in after.splitlines() if "vllm: On a CUDA host" in ln]
    assert advisory and advisory[0].lstrip().startswith("•")


def test_next_steps_caps_at_five(invoke_doctor):
    profile = _profile()
    for i in range(5):
        profile.runtimes[f"rt{i}"] = RuntimeStatus(
            name=f"rt{i}", available=False, reason="not on PATH", install_hint="pip install example"
        )
    after = invoke_doctor(profile).output.split("Next steps", 1)[1]
    assert "... and 2 more" in after  # 7 unavailable -> 5 shown + 2 more


def test_summary_states_counts_and_next_action(invoke_doctor):
    output = invoke_doctor(_profile()).output
    assert "2/4 runtimes available" in output  # success() line with counts
    assert "spark list" in output
    assert any(line.lstrip().startswith("→") for line in output.splitlines())


def test_host_budget_uses_human_units_and_warns_on_low_disk(invoke_doctor, monkeypatch):
    monkeypatch.setattr("spark.cli.doctor.free_disk_bytes", lambda *a: 8 * 2**30)
    output = invoke_doctor(_profile()).output
    assert "16.0 GiB" in output  # fmt_size-style unified memory
    assert "8.0 GiB" in output  # free disk in human units
    assert "low disk" in output
    assert any(line.lstrip().startswith("!") for line in output.splitlines())
