from __future__ import annotations

import pytest

from spark.config.loader import load_config
from spark.config.paths import SparkPaths


@pytest.fixture
def paths(tmp_path) -> SparkPaths:
    p = SparkPaths(
        config_dir=tmp_path / "config",
        data_dir=tmp_path / "data",
        cache_dir=tmp_path / "cache",
    )
    p.ensure()
    return p


@pytest.fixture
def config():
    return load_config()
