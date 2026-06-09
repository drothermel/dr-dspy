import pytest

from dspy.task_spec import TaskSpec, make_task_spec
from dspy.task_spec.serialize import TASK_SPEC_VERSION, field_spec_from_dict, field_spec_to_dict


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


def test_from_dict_rejects_old_task_spec_version():
    payload = make_task_spec("q -> a", instructions="Answer.").to_dict()
    payload["task_spec_version"] = TASK_SPEC_VERSION - 1
    with pytest.raises(ValueError, match="Unsupported task_spec_version"):
        TaskSpec.from_dict(payload)
