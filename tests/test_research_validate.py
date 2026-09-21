"""Accept-time research validation: staged flags must exist on the installed server.

Pins the 2026-09-21 incident: research recommended ``--max-kv-size`` for the
``mlx_lm`` backend (a flag that exists only on ``mlx_vlm``) plus a non-canonical
``2bit`` quant tag, and ``review --accept`` wrote both blindly — the server
died on argparse, three pointless restarts, terminal failure.
"""

from __future__ import annotations

import pytest

from spark.config.schema import ModelEntry, RuntimeDef
from spark.research.types import ResearchOutput, RuntimeRec
from spark.research.validate import (
    known_flags,
    normalize_quant,
    validate_accept,
    validate_flags,
)


def _output(**overrides) -> ResearchOutput:
    return ResearchOutput(
        recommended_backend="mlx_lm",
        per_runtime={
            "mlx_lm": RuntimeRec(
                quant="q8", launch_overrides=overrides, context_length=8192,
            )
        },
    )


def _config_with_fake_binary(tmp_path) -> tuple:
    """A SparkConfig whose mlx_lm binary is a stub echoing a fixed --help."""
    from spark.config.loader import load_config

    script = tmp_path / "fake-server"
    script.write_text(
        "#!/bin/sh\n"
        "echo 'usage: fake [--model MODEL] [--host HOST] [--port PORT] [--max-tokens MAX_TOKENS]'\n"
    )
    script.chmod(0o755)
    config = load_config()
    rt = config.runtimes["mlx_lm"]
    config.runtimes["mlx_lm"] = RuntimeDef.model_validate(
        {**rt.model_dump(), "server": {**rt.server.model_dump(), "binary": str(script)}}
    )
    return config, str(script)


# --- quant --------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "fixed"),
    [("2bit", "q2"), ("4-BIT", "q4"), ("8 bit", "q8"), ("q4", "q4"),
     ("Q8_0", "Q8_0"), ("Q4_K_M", "Q4_K_M"), ("f16", "f16"), ("", "")],
)
def test_normalize_quant_only_rewrites_bare_bit_tags(raw, fixed):
    assert normalize_quant(raw) == fixed


# --- flag surface ---------------------------------------------------------------
def test_known_flags_parses_help_surface(tmp_path):
    _, script = _config_with_fake_binary(tmp_path)
    flags = known_flags(script)
    assert flags is not None
    assert {"--model", "--host", "--port", "--max-tokens"} <= flags


def test_known_flags_is_none_for_missing_binary():
    assert known_flags("/nonexistent/binary-xyz") is None


def test_valid_flags_pass(tmp_path):
    config, _ = _config_with_fake_binary(tmp_path)
    errors, warnings = validate_flags({"--max-tokens": "100"}, "mlx_lm", config)
    assert errors == [] and warnings == []


def test_hallucinated_flag_is_an_error(tmp_path):
    """The incident: --max-kv-size recommended for mlx_lm must refuse accept."""
    config, _ = _config_with_fake_binary(tmp_path)
    errors, _ = validate_flags({"--max-kv-size": "16384"}, "mlx_lm", config)
    assert len(errors) == 1 and "--max-kv-size" in errors[0]


def test_unprobable_surface_warns_open(tmp_path, monkeypatch):
    """llama.cpp 0.4.1 precedent: a hanging --help must not block --accept."""
    from spark.config.loader import load_config

    config = load_config()
    monkeypatch.setattr(
        "spark.research.validate.known_flags", lambda *a, **k: None
    )
    errors, warnings = validate_flags({"--anything": "1"}, "mlx_lm", config)
    assert errors == []
    assert any("skipped" in w for w in warnings)


def test_unknown_backend_is_an_error(tmp_path):
    config, _ = _config_with_fake_binary(tmp_path)
    errors, _ = validate_flags({}, "nope", config)
    assert errors and "nope" in errors[0]


# --- end to end -----------------------------------------------------------------
def test_validate_accept_applies_and_normalizes(tmp_path):
    config, _ = _config_with_fake_binary(tmp_path)
    entry = ModelEntry(id="m", model_format="mlx")
    out = _output(**{"--max-tokens": "512"})
    out.per_runtime["mlx_lm"].quant = "8bit"
    candidate, errors, _ = validate_accept(entry, out, config)
    assert errors == []
    assert candidate.backend == "mlx_lm"
    assert candidate.quant == "q8"  # normalized, not stored as 8bit
    assert candidate.launch_overrides == {"--max-tokens": "512"}
    assert candidate.research_status == "registered"
    # The caller's entry is untouched on the way through.
    assert entry.backend == "" and entry.quant == ""


def test_validate_accept_refuses_without_touching_entry(tmp_path):
    config, _ = _config_with_fake_binary(tmp_path)
    entry = ModelEntry(id="m", backend="mlx_lm", model_format="mlx")
    candidate, errors, _ = validate_accept(entry, _output(**{"--max-kv-size": "1"}), config)
    assert errors
    assert entry.launch_overrides == {}
    assert entry.research_status == "pending"


def test_validate_accept_catches_stale_carryover_flags(tmp_path):
    """A backend switch must not smuggle the old server's flags along."""
    config, _ = _config_with_fake_binary(tmp_path)
    entry = ModelEntry(
        id="m", backend="mlx_vlm",
        launch_overrides={"--max-kv-size": "16384"}, model_format="mlx",
    )
    _, errors, _ = validate_accept(entry, _output(), config)
    assert any("--max-kv-size" in e for e in errors)
