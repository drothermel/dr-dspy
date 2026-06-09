import pytest

from dspy.task_spec import (
    FieldBinding,
    field_bindings,
    format_field_value,
    input_field,
    make_task_spec,
    output_field,
    translate_field_type,
    validate_task_inputs_from_spec,
)
from dspy.task_spec.field_spec import FieldRole


def test_field_bindings_returns_named_specs():
    spec = make_task_spec("q -> a", instructions="Answer.")
    bindings = field_bindings(spec, role=FieldRole.INPUT)
    assert bindings == (FieldBinding(name="q", field=spec.input_fields["q"]),)


def test_format_field_value_serializes_dict():
    field = input_field("payload", dict, desc="Payload map.")
    rendered = format_field_value(field, {"a": 1})
    assert rendered == '{"a": 1}'


def test_translate_field_type_for_bool_output():
    field = output_field("flag", bool, desc="Boolean flag.")
    assert "True or False" in translate_field_type(field)


def test_validate_task_inputs_from_spec_applies_defaults():
    spec = make_task_spec(
        inputs=[input_field("q", desc="Question.", default="default-q")],
        outputs=[output_field("a", desc="Answer.")],
        instructions="Answer.",
    )
    validated = validate_task_inputs_from_spec(spec, {})
    assert validated == {"q": "default-q"}


def test_validate_task_inputs_from_spec_rejects_unknown_keys():
    spec = make_task_spec("q -> a", instructions="Answer.")
    with pytest.raises(ValueError, match="Unknown task input field"):
        validate_task_inputs_from_spec(spec, {"extra": 1})


def test_validate_task_inputs_from_spec_rejects_none_for_required_str():
    spec = make_task_spec(
        inputs=[input_field("q", str, desc="The question.")],
        outputs=[output_field("a", desc="The answer.")],
        instructions="Answer.",
    )
    with pytest.raises(ValueError, match="got incompatible value None"):
        validate_task_inputs_from_spec(spec, {"q": None})


def test_validate_task_inputs_from_spec_accepts_none_for_optional_str():
    spec = make_task_spec(
        inputs=[input_field("q", str | None, desc="Optional question.")],
        outputs=[output_field("a", desc="The answer.")],
        instructions="Answer.",
    )
    validated = validate_task_inputs_from_spec(spec, {"q": None})
    assert validated == {"q": None}


def test_validate_task_inputs_from_spec_rejects_default_none_on_plain_str():
    spec = make_task_spec(
        inputs=[input_field("q", desc="Question with null default.", default=None)],
        outputs=[output_field("a", desc="The answer.")],
        instructions="Answer.",
    )
    with pytest.raises(ValueError, match="got incompatible value None"):
        validate_task_inputs_from_spec(spec, {})
    with pytest.raises(ValueError, match="got incompatible value None"):
        validate_task_inputs_from_spec(spec, {"q": None})


def test_validate_task_inputs_from_spec_accepts_default_none_on_optional_str():
    spec = make_task_spec(
        inputs=[input_field("q", str | None, desc="Optional question.", default=None)],
        outputs=[output_field("a", desc="The answer.")],
        instructions="Answer.",
    )
    assert validate_task_inputs_from_spec(spec, {}) == {"q": None}
    assert validate_task_inputs_from_spec(spec, {"q": None}) == {"q": None}
