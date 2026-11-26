"""Tests for `spark doctor --fix` — installing missing runtime Python deps."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from click.testing import CliRunner

from spark.cli.doctor import _fix_python_deps, doctor_command
from spark.config.schema import DetectSpec, RuntimeDef, ServerSpec
from spark.errors import RuntimeBackendError
from spark.probe.host import HostProfile, RuntimeStatus
from spark.telemetry import init_telemetry


@pytest.fixture(autouse=True)
def _telemetry(tmp_path):
    init_telemetry(tmp_path / "logs", session_id="test-doctor-fix")


def _rt() -> RuntimeDef:
    return RuntimeDef(
        name="mlx_vlm",
        detect=DetectSpec(binary="mlx_vlm.server", python_requires=["torch", "torchvision"]),
        server=ServerSpec(binary="mlx_vlm.server"),
        install_hint="uv tool install mlx-vlm",
    )


ARGV = [
    "uv",
    "tool",
    "install",
    "mlx-vlm",
    "--with",
    "torch",
    "--with",
    "torchvision",
]


def test_fix_success_when_installer_ok_and_deps_appear(monkeypatch):
    calls = []

    def fake_run(argv, check=False):
        calls.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("spark.cli.doctor.missing_python_deps", lambda binary, reqs: [])
    assert _fix_python_deps(_rt(), ["torch", "torchvision"], list(ARGV)) is True
    assert calls == [ARGV]


def test_fix_fails_on_installer_nonzero_exit(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda argv, check=False: SimpleNamespace(returncode=2))
    monkeypatch.setattr(
        "spark.cli.doctor.missing_python_deps",
        lambda *a: pytest.fail("must not re-verify after a failed install"),
    )
    assert _fix_python_deps(_rt(), ["torch"], list(ARGV)) is False


def test_fix_fails_when_deps_still_missing_after_install(monkeypatch):
    # e.g. a wrong module:pip mapping — installer succeeds, import still fails.
    monkeypatch.setattr(subprocess, "run", lambda argv, check=False: SimpleNamespace(returncode=0))
    monkeypatch.setattr(
        "spark.cli.doctor.missing_python_deps", lambda binary, reqs: ["torchvision"]
    )
    assert _fix_python_deps(_rt(), ["torch", "torchvision"], list(ARGV)) is False


def test_fix_fails_when_installer_binary_absent(monkeypatch):
    def fake_run(argv, check=False):
        raise FileNotFoundError("uv")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _fix_python_deps(_rt(), ["torch"], list(ARGV)) is False


def _profile_with_mlx_vlm() -> HostProfile:
    return HostProfile(
        os="Darwin",
        arch="arm64",
        chip="Apple M2",
        total_memory_bytes=16 * 2**30,
        cpu_count=8,
        probed_at=0.0,
        runtimes={
            "mlx_vlm": RuntimeStatus(
                name="mlx_vlm", available=True, path="/x/mlx_vlm.server", version="0.1"
            )
        },
    )


def test_doctor_fix_calls_same_repair_path(tmp_path, monkeypatch):
    """`doctor --fix` still routes through _fix_python_deps with the derived argv."""
    calls = []
    monkeypatch.setenv("SPARK_HOME", str(tmp_path))
    monkeypatch.setattr("spark.cli.doctor.load_or_probe", lambda *a, **k: _profile_with_mlx_vlm())
    monkeypatch.setattr(
        "spark.cli.doctor.missing_python_deps",
        lambda binary, reqs: ["torch", "torchvision"],
    )

    def fake_fix(rt, missing, argv):
        calls.append((rt.name, missing, argv))
        return True

    monkeypatch.setattr("spark.cli.doctor._fix_python_deps", fake_fix)
    result = CliRunner().invoke(
        doctor_command,
        ["--fix"],
        env={"SPARK_HOME": str(tmp_path)},
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert calls == [("mlx_vlm", ["torch", "torchvision"], ARGV)]


def test_doctor_fix_keeps_error_semantics(tmp_path, monkeypatch):
    """A failed repair still raises RuntimeBackendError with the same code/remediation."""
    monkeypatch.setenv("SPARK_HOME", str(tmp_path))
    monkeypatch.setattr("spark.cli.doctor.load_or_probe", lambda *a, **k: _profile_with_mlx_vlm())
    monkeypatch.setattr("spark.cli.doctor.missing_python_deps", lambda binary, reqs: ["torch"])
    monkeypatch.setattr("spark.cli.doctor._fix_python_deps", lambda *a, **k: False)

    result = CliRunner().invoke(
        doctor_command,
        ["--fix"],
        env={"SPARK_HOME": str(tmp_path)},
        catch_exceptions=True,
    )
    err = result.exception
    assert isinstance(err, RuntimeBackendError)
    assert err.code == "RT_DEPS_UNFIXED"
    assert err.remediation == [
        "scroll up for the installer output",
        "run the printed `uv tool install ... --with ...` command by hand",
        "re-check with `spark doctor`",
    ]
