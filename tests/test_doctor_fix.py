"""Tests for `spark doctor --fix` — installing missing runtime Python deps."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from spark.cli.doctor import _fix_python_deps
from spark.config.schema import DetectSpec, RuntimeDef, ServerSpec
from spark.telemetry import init_telemetry


@pytest.fixture(autouse=True)
def _telemetry(tmp_path):
    init_telemetry(tmp_path / "logs", session_id="test-doctor-fix")


def _rt() -> RuntimeDef:
    return RuntimeDef(
        name="mlx_vlm",
        detect=DetectSpec(
            binary="mlx_vlm.server", python_requires=["torch", "torchvision"]
        ),
        server=ServerSpec(binary="mlx_vlm.server"),
        install_hint="uv tool install mlx-vlm",
    )


ARGV = [
    "uv", "tool", "install", "mlx-vlm",
    "--with", "torch", "--with", "torchvision",
]


def test_fix_success_when_installer_ok_and_deps_appear(monkeypatch):
    calls = []

    def fake_run(argv, check=False):
        calls.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        "spark.cli.doctor.missing_python_deps", lambda binary, reqs: []
    )
    assert _fix_python_deps(_rt(), ["torch", "torchvision"], list(ARGV)) is True
    assert calls == [ARGV]


def test_fix_fails_on_installer_nonzero_exit(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run", lambda argv, check=False: SimpleNamespace(returncode=2)
    )
    monkeypatch.setattr(
        "spark.cli.doctor.missing_python_deps",
        lambda *a: pytest.fail("must not re-verify after a failed install"),
    )
    assert _fix_python_deps(_rt(), ["torch"], list(ARGV)) is False


def test_fix_fails_when_deps_still_missing_after_install(monkeypatch):
    # e.g. a wrong module:pip mapping — installer succeeds, import still fails.
    monkeypatch.setattr(
        subprocess, "run", lambda argv, check=False: SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(
        "spark.cli.doctor.missing_python_deps", lambda binary, reqs: ["torchvision"]
    )
    assert _fix_python_deps(_rt(), ["torch", "torchvision"], list(ARGV)) is False


def test_fix_fails_when_installer_binary_absent(monkeypatch):
    def fake_run(argv, check=False):
        raise FileNotFoundError("uv")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _fix_python_deps(_rt(), ["torch"], list(ARGV)) is False
