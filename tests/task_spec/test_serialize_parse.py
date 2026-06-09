import pytest
from pydantic import ValidationError

from dspy.adapters.types.tool import ToolCallResults, ToolCalls
from dspy.task_spec import TaskSpec, make_task_spec
from dspy.task_spec.type_registry import (
    BUILTIN_TYPES_BY_NAME,
    DSPY_TYPE_ALIASES,
    DSPY_TYPE_MODULES,
    type_from_str,
    type_to_str,
)
from dspy.task_spec.wire import TASK_SPEC_VERSION, field_spec_from_dict, field_spec_to_dict


def _dspy_types_for_round_trip() -> list[tuple[str, type]]:
    types: list[tuple[str, type]] = [
        (name, getattr(__import__(module_path, fromlist=[name]), name))
        for name, module_path in DSPY_TYPE_MODULES.items()
    ]
    types.extend((alias, ToolCalls if alias == "ToolCalls" else ToolCallResults) for alias in DSPY_TYPE_ALIASES)
    return types


@pytest.mark.parametrize(("type_name", "type_obj"), list(BUILTIN_TYPES_BY_NAME.items()))
def test_builtin_type_round_trip(type_name, type_obj):
    assert type_from_str(type_to_str(type_obj)) is type_obj


@pytest.mark.parametrize(("type_name", "type_obj"), _dspy_types_for_round_trip())
def test_dspy_type_round_trip(type_name, type_obj):
    assert type_from_str(type_to_str(type_obj)) is type_obj


@pytest.mark.parametrize(
    "generic",
    [
        list[str],
        dict[str, int],
        str | None,
        tuple[int, ...],
    ],
)
def test_generic_type_round_trip(generic):
    assert type_from_str(type_to_str(generic)) == generic


@pytest.mark.parametrize(
    "spec",
    [
        "input: Image -> output: str",
        "input: Tool -> output: str",
        "input: ToolCalls -> output: str",
        "input: ToolCallResults -> output: str",
        "input: TurnLog -> output: str",
        "input: REPLHistory -> output: str",
        "input: Reasoning -> output: str",
        "input: Code -> output: str",
        "input: File -> output: str",
        "input: Audio -> output: str",
    ],
)
def test_parse_dspy_type_spec_round_trips_through_serialization(spec):
    original = make_task_spec(spec, instructions="Test.")
    restored = TaskSpec.from_dict(original.to_dict())
    for name in original.input_fields:
        assert restored.input_fields[name].type_ == original.input_fields[name].type_
    for name in original.output_fields:
        assert restored.output_fields[name].type_ == original.output_fields[name].type_


def test_type_from_str_raises_on_unknown_type():
    data = field_spec_to_dict(make_task_spec("q -> a", instructions="Answer.").input_fields["q"])
    data["type"] = "unknown.module.MissingType"
    with pytest.raises(ValueError, match="Unknown serialized field type"):
        field_spec_from_dict(data)


def test_type_from_str_round_trips_nested_generics():
    spec = make_task_spec("context: list[str] -> answer", instructions="Answer using context.")
    restored = TaskSpec.from_dict(spec.to_dict())
    assert restored.input_fields["context"].type_ == list[str]


def test_parse_rejects_unknown_spec_string_type():
    with pytest.raises(ValueError, match="Unknown type name"):
        make_task_spec("input: NotARegisteredType -> output", instructions="Test.")


def test_custom_types_override_registry():
    class CustomType(type):
        pass

    custom_types: dict[str, type] = {"custom.module.CustomType": CustomType}
    assert type_from_str("custom.module.CustomType", custom_types=custom_types) is CustomType


def test_from_dict_rejects_old_task_spec_version():
    payload = make_task_spec("q -> a", instructions="Answer.").to_dict()
    payload["task_spec_version"] = TASK_SPEC_VERSION - 1
    with pytest.raises(ValueError, match="Unsupported task_spec_version"):
        TaskSpec.from_dict(payload)


@pytest.mark.parametrize("missing_key", ["type", "desc", "prefix", "role", "name"])
def test_field_spec_from_dict_raises_on_missing_required_key(missing_key):
    data = field_spec_to_dict(make_task_spec("q -> a", instructions="Answer.").input_fields["q"])
    data.pop(missing_key)
    with pytest.raises(ValidationError):
        field_spec_from_dict(data)


def test_field_spec_from_dict_raises_when_has_default_without_default_key():
    data = field_spec_to_dict(make_task_spec("q -> a", instructions="Answer.").input_fields["q"])
    data["has_default"] = True
    data.pop("default", None)
    with pytest.raises(ValueError, match="has_default=true but missing key 'default'"):
        field_spec_from_dict(data)


def test_field_spec_from_dict_raises_on_invalid_role():
    data = field_spec_to_dict(make_task_spec("q -> a", instructions="Answer.").input_fields["q"])
    data["role"] = "sidecar"
    with pytest.raises(ValidationError):
        field_spec_from_dict(data)


def test_field_spec_wire_rejects_extra_keys():
    data = field_spec_to_dict(make_task_spec("q -> a", instructions="Answer.").input_fields["q"])
    data["unexpected"] = True
    with pytest.raises(ValidationError):
        field_spec_from_dict(data)
