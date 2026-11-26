from __future__ import annotations

import builtins
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from spark.config.schema import DetectSpec, RuntimeDef, ServerSpec
from spark.probe import host
from spark.probe.host import (
    HostProfile,
    RuntimeStatus,
    _detect_version,
    _interpreter_for,
    _version_from_package,
    dep_fix_command,
    detect_runtime,
    load_or_probe,
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


def test_numeric_version_recovered_from_warning_banner():
    # ollama with its daemon down prints only warning lines; the dotted version
    # must still be extracted from them.
    out = _detect_version("/bin/echo", ["Warning: client version is 0.32.0"], 3.0)
    assert out == "0.32.0"


def test_loose_version_token_not_taken_from_banner():
    # The 'version: <word>' pattern is too loose to trust inside banner lines.
    assert _detect_version("/bin/echo", ["error: version: unknown"], 3.0) == ""


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


def test_version_from_package(tmp_path, monkeypatch):
    _fake_cli(tmp_path, monkeypatch)
    # click is a spark dependency, so it's installed in the fake cli's
    # interpreter (sys.executable = this venv).
    v = _version_from_package("fakecli", "click")
    assert v and v[0].isdigit()
    assert _version_from_package("fakecli", "not-a-real-dist-xyz") == ""
    assert _version_from_package("definitely-not-a-real-binary-xyz", "click") == ""


def test_detect_runtime_prefers_version_package(tmp_path, monkeypatch):
    _fake_cli(tmp_path, monkeypatch)
    rt = _rt(
        name="faketool",
        detect=DetectSpec(
            binary="fakecli",
            version_args=["usage: no version here"],
            version_package="click",
        ),
    )
    st = detect_runtime(rt, os_name="Darwin", arch="arm64")
    assert st.available
    assert st.version and st.version[0].isdigit()


def test_split_python_req():
    assert split_python_req("torch") == ("torch", "torch")
    assert split_python_req("PIL:pillow") == ("PIL", "pillow")
    assert split_python_req(" PIL : pillow ") == ("PIL", "pillow")


def test_missing_python_deps_returns_raw_req_with_pip_mapping(tmp_path, monkeypatch):
    _fake_cli(tmp_path, monkeypatch)
    missing = missing_python_deps("fakecli", ["os", "totally_not_real_mod_xyz:some-pip-name"])
    assert missing == ["totally_not_real_mod_xyz:some-pip-name"]


def test_dep_fix_command_extends_uv_tool_install():
    argv = dep_fix_command("uv tool install mlx-vlm", ["torch", "torchvision"])
    assert argv == [
        "uv",
        "tool",
        "install",
        "mlx-vlm",
        "--with",
        "torch",
        "--with",
        "torchvision",
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


# --- cache TTL (config-owned, never a magic number in probe code) ---------------
def _count_probes(monkeypatch):
    calls: list[object] = []

    def fake_probe(config):
        calls.append(config)
        return HostProfile(
            os="Darwin",
            arch="arm64",
            chip="M2",
            total_memory_bytes=16 * 2**30,
            cpu_count=8,
            probed_at=time.time(),
        )

    monkeypatch.setattr("spark.probe.host.probe_host", fake_probe)
    return calls


def test_load_or_probe_uses_cache_within_ttl(config, tmp_path, monkeypatch):
    calls = _count_probes(monkeypatch)
    path = tmp_path / "host.toml"
    load_or_probe(config, path, max_age_s=config.probe.max_age_s)
    load_or_probe(config, path, max_age_s=config.probe.max_age_s)
    assert len(calls) == 1


def test_load_or_probe_bypasses_cache_past_ttl(config, tmp_path, monkeypatch):
    calls = _count_probes(monkeypatch)
    path = tmp_path / "host.toml"
    load_or_probe(config, path, max_age_s=1.0)
    now = time.time()
    monkeypatch.setattr("spark.probe.host.time.time", lambda: now + 2)
    load_or_probe(config, path, max_age_s=1.0)
    assert len(calls) == 2


def test_load_or_probe_honors_config_max_age(config, tmp_path, monkeypatch):
    calls = _count_probes(monkeypatch)
    cfg = config.model_copy(update={"probe": config.probe.model_copy(update={"max_age_s": 1.0})})
    path = tmp_path / "host.toml"
    load_or_probe(cfg, path, max_age_s=cfg.probe.max_age_s)
    now = time.time()
    monkeypatch.setattr("spark.probe.host.time.time", lambda: now + 2)
    load_or_probe(cfg, path, max_age_s=cfg.probe.max_age_s)
    assert len(calls) == 2  # the config's 1s TTL expired; a hardcoded day had not


def test_load_or_probe_requires_max_age_s(config, tmp_path):
    with pytest.raises(TypeError):
        load_or_probe(config, tmp_path / "host.toml")


# --- remaining detection / cache branches --------------------------------------
def test_profile_to_dict_and_available_runtimes():
    profile = HostProfile(
        os="Darwin",
        arch="arm64",
        chip="M2",
        total_memory_bytes=16 * 2**30,
        cpu_count=8,
        probed_at=1.0,
        runtimes={
            "on": RuntimeStatus(name="on", available=True),
            "off": RuntimeStatus(name="off", available=False, reason="gated"),
        },
    )
    assert profile.available_runtimes == ["on"]
    assert profile.to_dict()["runtimes"]["off"]["reason"] == "gated"


def test_darwin_facts_come_from_sysctl(monkeypatch):
    values = {
        "hw.memsize": "17179869184\n",
        "machdep.cpu.brand_string": "Apple M2 Max\n",
    }
    monkeypatch.setattr(host.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        host.subprocess, "run", lambda cmd, **kw: SimpleNamespace(stdout=values[cmd[-1]])
    )
    assert host._sysctl("hw.memsize") == "17179869184"
    assert host._total_memory_bytes() == 17179869184
    assert host._chip() == "Apple M2 Max"


def test_facts_fall_back_when_platform_is_not_darwin(monkeypatch):
    monkeypatch.setattr(host.platform, "system", lambda: "Linux")
    monkeypatch.setattr(host.platform, "processor", lambda: "")
    monkeypatch.setattr(host.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(host.os, "sysconf", lambda key: 4096 if key == "SC_PAGE_SIZE" else 100)
    assert host._total_memory_bytes() == 409600
    assert host._chip() == "x86_64"
    monkeypatch.setattr(
        host.os, "sysconf", lambda key: (_ for _ in ()).throw(OSError("no sysconf"))
    )
    assert host._total_memory_bytes() == 0


def test_sysctl_failure_yields_empty_string(monkeypatch):
    def boom(*_a, **_k):
        raise OSError("sysctl missing")

    monkeypatch.setattr(host.subprocess, "run", boom)
    assert host._sysctl("hw.memsize") == ""


def test_version_detection_survives_subprocess_failures(monkeypatch):
    def boom(*_a, **_k):
        raise subprocess.TimeoutExpired("/bin/echo", 1)

    monkeypatch.setattr(host.subprocess, "run", boom)
    assert _detect_version("/bin/echo", [], 0.1) == ""
    assert _version_from_package("fakecli", "click") == ""  # interpreter unresolvable


def test_version_from_package_survives_subprocess_error(tmp_path, monkeypatch):
    _fake_cli(tmp_path, monkeypatch)

    def boom(*_a, **_k):
        raise OSError("spawn failed")

    monkeypatch.setattr(host.subprocess, "run", boom)
    assert _version_from_package("fakecli", "click") == ""


def _script(tmp_path, monkeypatch, name, first_line, mode=0o755):
    bindir = tmp_path / "bin-branches"
    bindir.mkdir(exist_ok=True)
    script = bindir / name
    script.write_text(first_line + "\n")
    script.chmod(mode)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])
    return script


def test_interpreter_for_unreadable_and_missing_shebang(tmp_path, monkeypatch):
    _script(tmp_path, monkeypatch, "no-shebang", "not a launcher")
    assert _interpreter_for("no-shebang") is None
    _script(tmp_path, monkeypatch, "empty-shebang", "#!")
    assert _interpreter_for("empty-shebang") is None
    _script(tmp_path, monkeypatch, "unreadable", "#!/usr/bin/python3")
    real_open = builtins.open

    def boom_open(file, *args, **kwargs):
        if str(file).endswith("unreadable"):
            raise OSError("permission denied")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", boom_open)
    assert _interpreter_for("unreadable") is None
    assert _interpreter_for("definitely-not-a-real-binary-xyz") is None


def test_interpreter_for_resolves_env_shebang(tmp_path, monkeypatch):
    _script(tmp_path, monkeypatch, "env-cli", "#!/usr/bin/env python3")
    assert _interpreter_for("env-cli") == host.shutil.which("python3")


def test_missing_python_deps_survives_subprocess_error(tmp_path, monkeypatch):
    _fake_cli(tmp_path, monkeypatch)

    def boom(*_a, **_k):
        raise OSError("spawn failed")

    monkeypatch.setattr(host.subprocess, "run", boom)
    assert missing_python_deps("fakecli", ["os"]) == []


def test_dep_fix_command_bad_shell_quote_is_none():
    assert dep_fix_command("uv tool install 'unclosed", ["torch"]) is None


def test_probe_host_builds_profile_and_logs(config, monkeypatch):
    events = []

    class SpyTelemetry:
        def info(self, *args, **kwargs):
            events.append((args, kwargs))

    monkeypatch.setattr(host, "get_telemetry", lambda: SpyTelemetry())
    monkeypatch.setattr(host.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(host.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(host, "_chip", lambda: "Apple M2")
    monkeypatch.setattr(host, "_total_memory_bytes", lambda: 16 * 2**30)
    monkeypatch.setattr(host, "_cpu_count", lambda: 8)
    monkeypatch.setattr(host.time, "time", lambda: 123.0)
    monkeypatch.setattr(
        host,
        "detect_runtime",
        lambda rt, *, os_name, arch: RuntimeStatus(name=rt.name, available=True),
    )
    profile = host.probe_host(config)
    assert profile.probed_at == 123.0
    assert profile.chip == "Apple M2" and profile.cpu_count == 8
    assert set(profile.runtimes) == set(config.runtimes)
    assert any(args[1] == "host_profiled" for args, _ in events)
    assert any(args[1] == "runtime_detected" for args, _ in events)


def test_cpu_count_none_is_zero(monkeypatch):
    monkeypatch.setattr(host.os, "cpu_count", lambda: None)
    assert host._cpu_count() == 0


def test_load_or_probe_reprobes_on_corrupt_cache(config, tmp_path, monkeypatch):
    calls = _count_probes(monkeypatch)
    path = tmp_path / "host.toml"
    path.write_text("this is not = valid toml")
    load_or_probe(config, path, max_age_s=3600)
    assert len(calls) == 1


def test_load_or_probe_force_bypasses_fresh_cache(config, tmp_path, monkeypatch):
    calls = _count_probes(monkeypatch)
    path = tmp_path / "host.toml"
    load_or_probe(config, path, max_age_s=3600)
    load_or_probe(config, path, max_age_s=3600, force=True)
    assert len(calls) == 2


def test_load_or_probe_reprobes_when_runtime_config_changes(config, tmp_path, monkeypatch):
    calls = _count_probes(monkeypatch)
    path = tmp_path / "host.toml"
    load_or_probe(config, path, max_age_s=3600)
    name = next(iter(config.runtimes))
    rt = config.runtimes[name]
    changed_rt = rt.model_copy(
        update={"detect": rt.detect.model_copy(update={"version_args": ["--different"]})}
    )
    changed = config.model_copy(update={"runtimes": {**config.runtimes, name: changed_rt}})
    load_or_probe(changed, path, max_age_s=3600)
    assert len(calls) == 2


def test_load_or_probe_round_trips_runtime_statuses(config, tmp_path, monkeypatch):
    def fake_probe(_config):
        return HostProfile(
            os="Darwin",
            arch="arm64",
            chip="M2",
            total_memory_bytes=16 * 2**30,
            cpu_count=8,
            probed_at=time.time(),
            runtimes={
                "mlx": RuntimeStatus(
                    name="mlx",
                    available=True,
                    path="/bin/mlx",
                    version="0.1.0",
                    install_hint="uv tool install mlx-lm",
                ),
                "off": RuntimeStatus(name="off", available=False, reason="gated"),
            },
        )

    monkeypatch.setattr("spark.probe.host.probe_host", fake_probe)
    path = tmp_path / "nested" / "host.toml"
    load_or_probe(config, path, max_age_s=3600)
    cached = load_or_probe(config, path, max_age_s=3600)
    assert cached.runtimes["mlx"].version == "0.1.0"
    assert cached.runtimes["mlx"].install_hint == "uv tool install mlx-lm"
    assert cached.runtimes["off"].reason == "gated"
