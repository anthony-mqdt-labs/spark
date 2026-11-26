"""Regression tests for the config/research hardening slice.

Pins four accept-time holes:
  * GeneralConfig silently defaulting a non-canonical port range when the
    bundled TOML is missing;
  * prompt-injected `launch_overrides` steering `--host`, `--port` or `--model`
    past D11 flag validation;
  * bare SparkError failures in the research chain/provider with no remediation;
  * broad `except Exception` reporting real bugs as config typos, plus the
    duplicated accept/import pipeline drifting apart.
"""

from __future__ import annotations

import json
import sys

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from spark.cli.config_cmd import config_group
from spark.config.loader import load_config
from spark.config.paths import resolve_paths
from spark.config.schema import DFlashProfile, GeneralConfig, ModelEntry, ResearchProviderDef
from spark.errors import ConfigError, SparkError
from spark.registry import save_model
from spark.research.chain import ResearchChain
from spark.research.providers import CliAgentProvider, ResearchProvider
from spark.research.staging import stage
from spark.research.types import (
    ResearchExhaustedError,
    ResearchOutput,
    ResearchProviderError,
    ResearchRequest,
    ResearchUnavailableError,
    RuntimeRec,
)
from spark.research.validate import validate_accept, validate_flags


# --- 1. TOML-owned general fields are required ----------------------------------
def test_general_config_rejects_empty_mapping():
    with pytest.raises(ValidationError):
        GeneralConfig.model_validate({})


def test_bundled_defaults_pin_canonical_port_range():
    assert load_config().general.port_range == (8095, 8099)


def test_missing_bundled_defaults_is_a_config_error(tmp_path, monkeypatch):
    """A missing/empty defaults.toml must never silently bind 8080."""
    monkeypatch.setenv("SPARK_HOME", str(tmp_path / "home"))
    empty = tmp_path / "empty-config"
    empty.mkdir()
    monkeypatch.setattr("spark.config.loader.bundled_config_dir", lambda: empty)
    with pytest.raises(ConfigError) as e:
        load_config()
    assert e.value.remediation


# --- 2. launch_overrides cannot steer host/port/model ---------------------------
@pytest.mark.parametrize(
    "key",
    ["--host", "--port", "--model", "--HOST", "-host", "--Model"],
)
def test_forbidden_override_keys_are_typed_errors(key):
    with pytest.raises(SparkError) as e:
        ModelEntry(id="m", launch_overrides={key: "0.0.0.0"})
    assert e.value.code == "CFG_OVERRIDE_FORBIDDEN"
    assert e.value.remediation


def test_forbidden_override_keys_rejected_in_dflash_profile():
    with pytest.raises(SparkError) as e:
        DFlashProfile(launch_overrides={"--model": "/attacker/weights"})
    assert e.value.remediation


def test_benign_override_still_passes_accept_time(monkeypatch):
    monkeypatch.setattr("spark.research.validate.known_flags", lambda *a, **k: {"--max-tokens"})
    errors, warnings = validate_flags({"--max-tokens": "2048"}, "mlx_lm", load_config())
    assert errors == [] and warnings == []


@pytest.mark.parametrize(
    ("key", "value"),
    [("--host", "0.0.0.0"), ("--port", "9999"), ("--model", "/attacker/path")],
)
def test_validate_flags_refuses_injected_binding_and_model_keys(monkeypatch, key, value):
    monkeypatch.setattr(
        "spark.research.validate.known_flags",
        lambda *a, **k: {"--host", "--port", "--model"},
    )
    with pytest.raises(SparkError) as e:
        validate_flags({key: value}, "mlx_lm", load_config())
    assert key in str(e.value)
    assert e.value.remediation


def test_validate_accept_refuses_injected_override_without_touching_entry(monkeypatch):
    monkeypatch.setattr("spark.research.validate.known_flags", lambda *a, **k: {"--host"})
    entry = ModelEntry(id="m", backend="mlx_lm")
    out = ResearchOutput(
        recommended_backend="mlx_lm",
        per_runtime={"mlx_lm": RuntimeRec(launch_overrides={"--host": "0.0.0.0"})},
    )
    with pytest.raises(SparkError) as e:
        validate_accept(entry, out, load_config())
    assert "--host" in str(e.value) and e.value.remediation
    assert entry.launch_overrides == {} and entry.backend == "mlx_lm"


# --- 3. typed research errors carry remediation ---------------------------------
def test_provider_spawn_failure_is_typed_with_remediation():
    defn = _missing_binary_provider()
    with pytest.raises(ResearchProviderError) as e:
        defn.research(_request())
    assert e.value.code == "NET_RESEARCH"
    assert e.value.remediation


def test_provider_nonzero_exit_is_typed_with_remediation():
    defn = ResearchProviderDef(
        name="fake", command=[sys.executable, "-c", "import sys; sys.exit(7)"],
        prompt_via="stdin", parse="raw_json",
    )
    with pytest.raises(ResearchProviderError) as e:
        CliAgentProvider(defn).research(_request())
    assert e.value.code == "NET_RESEARCH"
    assert e.value.remediation


class _FakeProvider(ResearchProvider):
    def __init__(self, name, *, available=True, boom=False):
        self.name = name
        self._available = available
        self._boom = boom

    def is_available(self):
        return self._available

    def research(self, request):
        if self._boom:
            raise RuntimeError(f"{self.name} failed")
        return ResearchOutput(recommended_backend="mlx_lm")


def _request():
    return ResearchRequest(
        hf_repo="org/m", host_summary={"chip": "M2"}, runtime_names=["mlx_lm"]
    )


def _missing_binary_provider():
    return CliAgentProvider(
        ResearchProviderDef(
            name="missing", command=["/definitely/not/a/real/provider-binary"],
            prompt_via="stdin", parse="raw_json",
        )
    )


def test_chain_unavailable_is_typed_with_remediation():
    with pytest.raises(ResearchUnavailableError) as e:
        ResearchChain([_FakeProvider("a", available=False)]).run(_request())
    assert e.value.code == "NET_RESEARCH_UNAVAILABLE"
    assert e.value.remediation


def test_chain_exhausted_is_typed_with_remediation():
    with pytest.raises(ResearchExhaustedError) as e:
        ResearchChain([_FakeProvider("a", boom=True)]).run(_request())
    assert e.value.code == "NET_RESEARCH_EXHAUSTED"
    assert e.value.remediation


# --- 4. narrow exception handling -----------------------------------------------
def test_unknown_config_key_is_typed_with_remediation(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("SPARK_HOME", str(home))
    paths = resolve_paths().ensure()
    paths.user_config_file.write_text('[general]\nbogus_key = 1\n', encoding="utf-8")
    with pytest.raises(ConfigError) as e:
        load_config(paths)
    assert e.value.remediation


def test_loader_does_not_swallow_unexpected_exceptions(monkeypatch):
    from spark.config import loader as L

    def boom(_):
        raise RuntimeError("loader bug")

    monkeypatch.setattr(L.SparkConfig, "model_validate", staticmethod(boom))
    with pytest.raises(RuntimeError, match="loader bug"):
        load_config()


# --- 5. one accept/import pipeline ----------------------------------------------
@pytest.fixture
def cli_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("SPARK_HOME", str(home))
    paths = resolve_paths().ensure()
    monkeypatch.setattr(
        "spark.research.validate.known_flags", lambda *a, **k: {"--max-tokens"}
    )
    save_model(
        ModelEntry(id="m", hf_repo="org/m", backend="mlx_lm", model_format="mlx"),
        paths,
    )
    return paths


def _staged_output(**flags):
    return ResearchOutput(
        recommended_backend="mlx_lm",
        per_runtime={
            "mlx_lm": RuntimeRec(quant="q4", launch_overrides=flags or {"--max-tokens": "2048"})
        },
    )


@pytest.mark.parametrize("route", ["review", "import"])
def test_accept_and_import_share_pipeline(cli_home, route):
    output = _staged_output()
    if route == "review":
        stage(cli_home, "m", output, provider="fake", hf_repo="org/m", host_summary={})
        args, kwargs = ["review", "m", "--accept"], {}
    else:
        args = ["import", "m"]
        kwargs = {"input": json.dumps({"output": output.model_dump()})}
    result = CliRunner().invoke(config_group, args, **kwargs)
    assert result.exit_code == 0, result.output
    entry_file = cli_home.models_dir / "m.toml"
    saved = entry_file.read_text(encoding="utf-8")
    assert "max-tokens" in saved or "max_tokens" in saved
    assert "applied" in result.output or "imported" in result.output


@pytest.mark.parametrize("route", ["review", "import"])
def test_accept_and_import_report_unexpected_exceptions(cli_home, monkeypatch, route):
    def boom(*a, **k):
        raise RuntimeError("accept pipeline bug")

    monkeypatch.setattr("spark.research.validate.validate_accept", boom)
    if route == "review":
        stage(cli_home, "m", _staged_output(), provider="fake", hf_repo="org/m", host_summary={})
        args, kwargs = ["review", "m", "--accept"], {}
    else:
        args = ["import", "m"]
        kwargs = {"input": json.dumps({"output": _staged_output().model_dump()})}
    result = CliRunner().invoke(config_group, args, **kwargs)
    assert isinstance(result.exception, RuntimeError), result.output


@pytest.mark.parametrize("route", ["review", "import"])
def test_accept_and_import_refuse_injected_override(cli_home, route):
    output = _staged_output(**{"--host": "0.0.0.0"})
    if route == "review":
        stage(cli_home, "m", output, provider="fake", hf_repo="org/m", host_summary={})
        args, kwargs = ["review", "m", "--accept"], {}
    else:
        args = ["import", "m"]
        kwargs = {"input": json.dumps({"output": output.model_dump()})}
    result = CliRunner().invoke(config_group, args, **kwargs)
    assert isinstance(result.exception, (ConfigError, SparkError)), result.output
    assert result.exception.remediation
    assert (cli_home.models_dir / "m.toml").read_text(encoding="utf-8").find("0.0.0.0") == -1
