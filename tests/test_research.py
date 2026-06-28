from __future__ import annotations

import sys

import pytest

from spark.config.loader import load_config
from spark.config.schema import ModelEntry, ResearchProviderDef
from spark.errors import SparkError
from spark.research.chain import ResearchChain
from spark.research.guard import SecretLeakError, assert_no_secrets
from spark.research.providers import CliAgentProvider, ResearchProvider, _extract_json
from spark.research.staging import apply_output_to_entry, load_staged, stage
from spark.research.types import ResearchOutput, ResearchRequest
from spark.telemetry import register_secret_value


# --- guard ---------------------------------------------------------------------
def test_guard_passes_clean_text():
    assert_no_secrets("serve qwen with q4 on mlx_lm, 16 GiB host")  # no raise


def test_guard_blocks_hf_token_pattern():
    with pytest.raises(SecretLeakError):
        assert_no_secrets("use token hf_abcdefghijklmnopqrstuvwxyz012345")


def test_guard_blocks_registered_secret_value():
    register_secret_value("zzz-registered-secret-zzz")
    with pytest.raises(SecretLeakError):
        assert_no_secrets("prompt with zzz-registered-secret-zzz embedded")


# --- json extraction -----------------------------------------------------------
def test_extract_raw_json():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_fenced_json():
    assert _extract_json('text\n```json\n{"a": 2}\n```\nmore') == {"a": 2}


def test_extract_embedded_json_with_nested_braces():
    out = _extract_json('prose {"a": {"b": 1}, "c": "}"} trailing')
    assert out == {"a": {"b": 1}, "c": "}"}


def test_extract_raises_without_json():
    with pytest.raises(ValueError):
        _extract_json("no json here")


# --- chain fallback ------------------------------------------------------------
class _Fake(ResearchProvider):
    def __init__(self, name, *, available=True, output=None, boom=False):
        self.name = name
        self._a = available
        self._o = output
        self._boom = boom

    def is_available(self):
        return self._a

    def research(self, request):
        if self._boom:
            raise RuntimeError(f"{self.name} failed")
        return self._o


def _req():
    return ResearchRequest(hf_repo="org/m", host_summary={"chip": "M2"},
                           runtime_names=["mlx_lm"])


def test_chain_returns_first_success():
    good = ResearchOutput(recommended_backend="mlx_lm")
    chain = ResearchChain([_Fake("a", boom=True), _Fake("b", output=good)])
    res = chain.run(_req())
    assert res.provider == "b" and res.output.recommended_backend == "mlx_lm"


def test_chain_skips_unavailable():
    good = ResearchOutput(recommended_backend="mlx_lm")
    chain = ResearchChain([_Fake("a", available=False), _Fake("b", output=good)])
    assert chain.run(_req()).provider == "b"


def test_chain_all_unavailable_raises():
    chain = ResearchChain([_Fake("a", available=False)])
    with pytest.raises(SparkError) as e:
        chain.run(_req())
    assert e.value.code == "NET_RESEARCH_UNAVAILABLE"


def test_chain_all_fail_raises_exhausted():
    chain = ResearchChain([_Fake("a", boom=True), _Fake("b", boom=True)])
    with pytest.raises(SparkError) as e:
        chain.run(_req())
    assert e.value.code == "NET_RESEARCH_EXHAUSTED"


# --- CLI provider over a real subprocess (no LLM) ------------------------------
def test_cli_provider_parses_subprocess_json():
    script = ("import sys; sys.stdin.read(); "
              "print('{\"recommended_backend\":\"mlx_lm\","
              "\"per_runtime\":{\"mlx_lm\":{\"quant\":\"q4\"}}}')")
    defn = ResearchProviderDef(name="fake", command=[sys.executable, "-c", script],
                               prompt_via="stdin", parse="raw_json")
    out = CliAgentProvider(defn).research(_req())
    assert out.recommended_backend == "mlx_lm"
    assert out.per_runtime["mlx_lm"].quant == "q4"


def test_cli_provider_unavailable_when_binary_missing():
    defn = ResearchProviderDef(name="x", detect_binary="definitely-not-real-xyz",
                               command=["definitely-not-real-xyz"])
    assert CliAgentProvider(defn).is_available() is False


# --- staging -------------------------------------------------------------------
def test_stage_load_and_apply(paths):
    out = ResearchOutput(
        recommended_backend="llama_cpp",
        per_runtime={"llama_cpp": {"quant": "q4", "context_length": 8192,
                                   "launch_overrides": {"ctx": "8192"}}},
    )
    stage(paths, "m", out, provider="claude", hf_repo="org/m", host_summary={"chip": "M2"})
    doc = load_staged(paths, "m")
    assert doc["provider"] == "claude"

    entry = ModelEntry(id="m", hf_repo="org/m")
    apply_output_to_entry(entry, out)
    assert entry.backend == "llama_cpp"
    assert entry.quant == "q4"
    assert entry.context_length == 8192
    assert entry.launch_overrides == {"ctx": "8192"}
    assert entry.research_status == "registered"


# --- prompt is secret-free -----------------------------------------------------
def test_built_prompt_has_no_secrets(monkeypatch):
    from spark.research import prompt as P
    from spark.probe.host import HostProfile

    monkeypatch.setattr(P, "fetch_model_card", lambda repo, n: "A small instruct model.")
    cfg = load_config()
    prof = HostProfile(os="Darwin", arch="arm64", chip="Apple M2",
                       total_memory_bytes=16 * 2**30, cpu_count=8, probed_at=0.0)
    req = P.build_request("org/Model", prof, cfg, ["mlx_lm", "llama_cpp"])
    text = P.build_prompt(req)
    assert "org/Model" in text and "mlx_lm" in text
    assert_no_secrets(text)  # must not raise
