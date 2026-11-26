"""Regression tests: fast paths must stay fast, and must share one scan."""

from __future__ import annotations

import pytest

from spark.config.loader import load_config
from spark.config.schema import ModelEntry
from spark.inventory import build_inventory, resolve_for_run
from spark.registry.models import save_model
import spark.registry.availability as availability


@pytest.fixture
def one_weights_model(paths):
    weights = paths.model_store_dir / "fast-model"
    weights.mkdir(parents=True, exist_ok=True)
    (weights / "weights.bin").write_bytes(b"x" * 1024)
    save_model(ModelEntry(id="fast-model", path=str(weights)), paths)
    return weights


def test_with_bytes_false_does_not_walk_weights(paths, config, one_weights_model, monkeypatch):
    def _no_walk(*args, **kwargs):
        raise AssertionError("byte walk on a with_bytes=False fast path")

    monkeypatch.setattr(availability, "bytes_in", _no_walk)

    inv = build_inventory(paths, config, with_bytes=False)
    assert [e.id for e, _ in inv.runnable_registered] == ["fast-model"]
    # A dir without a byte walk carries a 1-byte presence marker, not a size.
    assert all(0 <= a.bytes_present <= 1 for _, a in inv.runnable_registered)


def test_with_bytes_true_still_reports_size(paths, config, one_weights_model):
    inv = build_inventory(paths, config, with_bytes=True)
    assert [(e.id, a.bytes_present) for e, a in inv.runnable_registered] == [
        ("fast-model", 1024)
    ]


def test_resolve_for_run_exposes_the_inventory_it_built(paths, config, one_weights_model):
    """A failed run must not rebuild the whole inventory just for suggestions."""
    hit = resolve_for_run("fast-model", paths, config)
    assert hit is not None
    assert hit.inventory is not None
    assert "fast-model" in hit.inventory.runnable_ids
