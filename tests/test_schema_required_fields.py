"""Host policy fields must be required, not silently defaulted in Python.

Config files are not owned by this test: start from the bundled defaults via
``load_config()`` and drop exactly one field from a copied dict before
re-validating. A removed TOML key must then name the field in the error.
"""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from spark.config.loader import load_config
from spark.config.schema import SparkConfig

_REQUIRED = [
    ("general", "host"),
    ("general", "port_range"),
    ("general", "log_retention_days"),
    ("general", "preference"),
    ("probe", "max_age_s"),
    ("supervisor", "health_probe_timeout_s"),
    ("research", "flag_probe_timeout_s"),
]


@pytest.mark.parametrize("section, field", _REQUIRED)
def test_each_host_policy_field_is_independently_required(section, field):
    data = copy.deepcopy(load_config().model_dump())
    assert field in data[section]  # the bundled defaults supply it
    del data[section][field]

    with pytest.raises(ValidationError) as exc:
        SparkConfig.model_validate(data)

    missing = [
        err for err in exc.value.errors() if err["type"] == "missing" and err["loc"][-1] == field
    ]
    assert missing, f"missing {section}.{field} was not reported: {exc.value}"
    assert missing[0]["loc"] == (section, field)
