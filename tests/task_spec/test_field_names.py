import pytest

from dspy.task_spec import TaskSpec, input_field, output_field, validate_field_name


def test_validate_field_name_accepts_hyphen_and_dot():
    validate_field_name("my-field")
    validate_field_name("foo.bar")
    validate_field_name("turn_log")


def test_validate_field_name_rejects_invalid_characters():
    with pytest.raises(ValueError, match="Invalid field name"):
        validate_field_name("bad name")


def test_task_spec_rejects_invalid_field_name_at_construction():
    with pytest.raises(ValueError, match="Invalid field name"):
        TaskSpec(
            name="bad",
            instructions="test",
            inputs=(input_field("bad name", str, desc="invalid"),),
            outputs=(),
        )


def test_task_spec_accepts_hyphenated_field_names():
    spec = TaskSpec(
        name="ok",
        instructions="test",
        inputs=(input_field("my-field", str, desc="ok"),),
        outputs=(output_field("result.value", str, desc="ok"),),
    )
    assert "my-field" in spec.input_fields
    assert "result.value" in spec.output_fields
