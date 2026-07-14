from __future__ import annotations

import os
import sys

from spark.config.schema import DetectSpec, RuntimeDef, ServerSpec
from spark.probe.host import (
    _detect_version,
    _interpreter_for,
    dep_fix_command,
    detect_runtime,
    missing_python_deps,
    split_python_req,
)


def _rt(**kw):
    base = dict(
        name="x",
        detect=DetectSpec(binary="x"),
        server=ServerSpec(binary="x"),
    )
    base.update(kw)
    return RuntimeDef(**base)


def test_version_extracted_from_clean_output():
    assert _detect_version("/bin/echo", ["1.2.3"], 3.0) == "1.2.3"


def test_usage_banner_yields_no_version():
    assert _detect_version("/bin/echo", ["usage: foo --bar"], 3.0) == ""


def test_detect_os_gating():
    rt = _rt(requires_os=["Linux"], unavailable_reason="linux only")
    st = detect_runtime(rt, os_name="Darwin", arch="arm64")
    assert not st.available
    assert "linux only" in st.reason


def test_detect_arch_gating():
    rt = _rt(requires_arch=["x86_64"])
    st = detect_runtime(rt, os_name="Darwin", arch="arm64")
    assert not st.available


def test_detect_missing_binary():
    rt = _rt(detect=DetectSpec(binary="definitely-not-a-real-binary-xyz"))
    st = detect_runtime(rt, os_name="Darwin", arch="arm64")
    assert not st.available
    assert "not found" in st.reason


def test_detect_present_binary():
    rt = _rt(detect=DetectSpec(binary="echo", version_args=["1.0.0"]))
    st = detect_runtime(rt, os_name="Darwin", arch="arm64")
    assert st.available
    assert st.path


def _fake_cli(tmp_path, monkeypatch, interp=sys.executable):
    """A console-script-style launcher whose shebang points at `interp`."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "fakecli"
    script.write_text(f"#!{interp}\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])
    return script


def test_interpreter_for_reads_shebang(tmp_path, monkeypatch):
    _fake_cli(tmp_path, monkeypatch)
    assert _interpreter_for("fakecli") == sys.executable


def test_missing_python_deps_all_present(tmp_path, monkeypatch):
    _fake_cli(tmp_path, monkeypatch)
    assert missing_python_deps("fakecli", ["sys", "os", "json"]) == []


def test_missing_python_deps_reports_absent(tmp_path, monkeypatch):
    _fake_cli(tmp_path, monkeypatch)
    missing = missing_python_deps("fakecli", ["os", "totally_not_real_mod_xyz"])
    assert missing == ["totally_not_real_mod_xyz"]


def test_missing_python_deps_unknown_is_not_missing(monkeypatch):
    # Unresolvable interpreter or nothing to check -> [] (unknown, never a warning).
    assert missing_python_deps("definitely-not-a-real-binary-xyz", ["os"]) == []
    assert missing_python_deps("python3", []) == []


def test_split_python_req():
    assert split_python_req("torch") == ("torch", "torch")
    assert split_python_req("PIL:pillow") == ("PIL", "pillow")
    assert split_python_req(" PIL : pillow ") == ("PIL", "pillow")


def test_missing_python_deps_returns_raw_req_with_pip_mapping(tmp_path, monkeypatch):
    _fake_cli(tmp_path, monkeypatch)
    missing = missing_python_deps(
        "fakecli", ["os", "totally_not_real_mod_xyz:some-pip-name"]
    )
    assert missing == ["totally_not_real_mod_xyz:some-pip-name"]


def test_dep_fix_command_extends_uv_tool_install():
    argv = dep_fix_command("uv tool install mlx-vlm", ["torch", "torchvision"])
    assert argv == [
        "uv", "tool", "install", "mlx-vlm",
        "--with", "torch", "--with", "torchvision",
    ]


def test_dep_fix_command_uses_pip_name_from_mapping():
    argv = dep_fix_command("uv tool install some-tool", ["PIL:pillow"])
    assert argv == ["uv", "tool", "install", "some-tool", "--with", "pillow"]


def test_dep_fix_command_refuses_non_uv_installers():
    # Never derive a runnable fix for installers whose semantics we don't own.
    assert dep_fix_command("brew install llama.cpp", ["torch"]) is None
    assert dep_fix_command("curl -fsSL https://ollama.com/install.sh | sh", ["x"]) is None
    assert dep_fix_command("uv tool install", ["torch"]) is None  # no package
    assert dep_fix_command("uv tool install mlx-vlm", []) is None  # nothing missing
