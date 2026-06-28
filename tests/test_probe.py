from __future__ import annotations

from spark.config.schema import DetectSpec, RuntimeDef, ServerSpec
from spark.probe.host import _detect_version, detect_runtime


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
