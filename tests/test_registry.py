from __future__ import annotations

import pytest

from spark.config.schema import DFlashProfile, ModelEntry
from spark.errors import AmbiguousModelError, ModelNotFoundError
from spark.registry import list_models, resolve_model, save_model


def _seed(paths, *ids):
    for i in ids:
        save_model(ModelEntry(id=i, model_format="mlx", quant="q4"), paths)


def test_save_and_list_roundtrip(paths):
    _seed(paths, "alpha-9b", "beta-7b")
    ids = {m.id for m in list_models(paths)}
    assert ids == {"alpha-9b", "beta-7b"}


def test_resolve_exact(paths):
    _seed(paths, "alpha-9b", "beta-7b")
    assert resolve_model("alpha-9b", paths).id == "alpha-9b"


def test_resolve_alias(paths):
    save_model(ModelEntry(id="alpha-9b", aliases=["ornith9b"]), paths)
    assert resolve_model("ornith9b", paths).id == "alpha-9b"


def test_resolve_unique_prefix(paths):
    _seed(paths, "alpha-9b", "beta-7b")
    assert resolve_model("alph", paths).id == "alpha-9b"


def test_resolve_ambiguous_raises(paths):
    _seed(paths, "qwen-7b", "qwen-14b")
    with pytest.raises(AmbiguousModelError):
        resolve_model("qwen", paths)


def test_resolve_miss_raises(paths):
    _seed(paths, "alpha-9b")
    with pytest.raises(ModelNotFoundError):
        resolve_model("zzz", paths)


def test_resolve_empty_registry_raises(paths):
    with pytest.raises(ModelNotFoundError):
        resolve_model("anything", paths)


def test_entry_persists_fields(paths):
    save_model(
        ModelEntry(
            id="m", hf_repo="org/M", quant="q4", params_billions=9.0,
            launch_overrides={"ctx": "8192"},
        ),
        paths,
    )
    m = resolve_model("m", paths)
    assert m.hf_repo == "org/M"
    assert m.params_billions == 9.0
    assert m.launch_overrides == {"ctx": "8192"}


def test_dflash_profile_roundtrips(paths):
    """A [dflash] profile survives save -> load through the custom TOML emitter."""
    save_model(
        ModelEntry(
            id="ornith", backend="mlx_lm", model_format="mlx",
            dflash=DFlashProfile(
                backend="mlx_vlm",
                launch_overrides={"--draft-model": "org/Draft", "--draft-kind": "dflash"},
            ),
        ),
        paths,
    )
    m = resolve_model("ornith", paths)
    assert m.backend == "mlx_lm"  # default path unchanged
    assert m.dflash is not None
    assert m.dflash.backend == "mlx_vlm"
    assert m.dflash.launch_overrides["--draft-model"] == "org/Draft"


def test_with_dflash_swaps_backend_and_merges_overrides():
    entry = ModelEntry(
        id="ornith", backend="mlx_lm", launch_overrides={"--max-tokens": "2048"},
        dflash=DFlashProfile(
            backend="mlx_vlm",
            launch_overrides={"--draft-model": "org/Draft", "--draft-kind": "dflash"},
        ),
    )
    d = entry.with_dflash()
    assert d.backend == "mlx_vlm"                       # profile backend wins
    assert d.launch_overrides["--draft-model"] == "org/Draft"
    assert d.launch_overrides["--max-tokens"] == "2048"  # base overrides preserved
    assert entry.backend == "mlx_lm"                    # original untouched (copy)


def test_no_dflash_profile_by_default():
    assert ModelEntry(id="m").dflash is None
