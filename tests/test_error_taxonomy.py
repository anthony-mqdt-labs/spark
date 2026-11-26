"""Every typed failure lives in errors.py and tells the operator what to do next."""

from __future__ import annotations

import inspect

import pytest

from spark import errors
from spark.errors import SparkError


def _subclasses(base: type) -> list[type]:
    seen: set[type] = set()
    stack = list(base.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
    return sorted(seen, key=lambda cls: cls.__name__)


def test_all_spark_error_subclasses_are_defined_in_errors_module():
    subclasses = _subclasses(SparkError)
    assert subclasses, "the taxonomy must contain typed failures"
    assert [cls.__name__ for cls in subclasses if cls.__module__ != errors.__name__] == []

    declared = {
        name
        for name, obj in vars(errors).items()
        if inspect.isclass(obj) and issubclass(obj, SparkError)
    }
    assert {cls.__name__ for cls in subclasses} <= declared


@pytest.mark.parametrize("cls", _subclasses(SparkError), ids=lambda cls: cls.__name__)
def test_every_spark_error_carries_actionable_remediation(cls):
    params = inspect.signature(cls).parameters
    # The only subclass with a domain-specific constructor takes the reserved key.
    err = cls("host") if "key" in params else cls("test failure")
    assert err.remediation, f"{cls.__name__} has no do-this-next steps"
    assert all(isinstance(step, str) and step.strip() for step in err.remediation)


def test_moved_errors_are_reexported_from_former_modules_with_identical_contract():
    from spark.config.schema import LaunchOverrideError as SchemaLaunchOverrideError
    from spark.research.guard import SecretLeakError as GuardSecretLeakError
    from spark.research.types import ResearchError as TypesResearchError
    from spark.research.types import ResearchExhaustedError as TypesResearchExhaustedError
    from spark.research.types import ResearchProviderError as TypesResearchProviderError
    from spark.research.types import ResearchUnavailableError as TypesResearchUnavailableError

    assert SchemaLaunchOverrideError is errors.LaunchOverrideError
    assert GuardSecretLeakError is errors.SecretLeakError
    assert TypesResearchError is errors.ResearchError
    assert TypesResearchProviderError is errors.ResearchProviderError
    assert TypesResearchUnavailableError is errors.ResearchUnavailableError
    assert TypesResearchExhaustedError is errors.ResearchExhaustedError

    override = errors.LaunchOverrideError("--host")
    assert override.code == "CFG_OVERRIDE_FORBIDDEN"
    assert override.context == {"key": "--host"}
    assert errors.SecretLeakError("boom").code == "SEC_PROMPT_LEAK"
    assert errors.SecretLeakError("boom").exit_code == 77
    assert errors.ResearchError("boom").code == "NET_RESEARCH"
    assert errors.ResearchProviderError("boom").code == "NET_RESEARCH"
    assert errors.ResearchUnavailableError("boom").code == "NET_RESEARCH_UNAVAILABLE"
    assert errors.ResearchExhaustedError("boom").code == "NET_RESEARCH_EXHAUSTED"
