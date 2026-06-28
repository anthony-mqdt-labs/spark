from __future__ import annotations

import pytest

from spark.config.schema import ModelEntry
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
