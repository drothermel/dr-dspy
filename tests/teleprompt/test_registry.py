from pydantic import BaseModel, ConfigDict

from dspy.teleprompt.registry import (
    compile_params_type,
    register_teleprompter,
    registered_teleprompters,
    validate_compile_params,
)


class _DummyParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: int = 1


@register_teleprompter(params=_DummyParams)
class _RegisteredOptimizer:
    async def compile(self, student, *, params, run):
        return student


def test_register_teleprompter_records_params_type():
    assert compile_params_type(_RegisteredOptimizer) is _DummyParams
    assert _RegisteredOptimizer in registered_teleprompters()


def test_validate_compile_params_accepts_matching_instance():
    optimizer = _RegisteredOptimizer()
    validate_compile_params(optimizer, _DummyParams())


def test_validate_compile_params_rejects_mismatch():
    class OtherParams(BaseModel):
        model_config = ConfigDict(extra="forbid")

    optimizer = _RegisteredOptimizer()
    try:
        validate_compile_params(optimizer, OtherParams())
    except TypeError as exc:
        assert "_DummyParams" in str(exc)
    else:
        raise AssertionError("expected TypeError")


def test_compile_params_type_raises_for_unregistered():
    class Unregistered:
        pass

    try:
        compile_params_type(Unregistered())
    except TypeError as exc:
        assert "not registered" in str(exc)
    else:
        raise AssertionError("expected TypeError")
