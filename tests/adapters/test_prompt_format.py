from dspy.adapters.prompt_format import format_field_value, translate_field_type
from dspy.task_spec import input_field, output_field


def test_format_field_value_serializes_dict():
    field = input_field("payload", dict, desc="Payload map.")
    rendered = format_field_value(field, {"a": 1})
    assert rendered == '{"a": 1}'


def test_translate_field_type_for_bool_output():
    field = output_field("flag", bool, desc="Boolean flag.")
    assert "True or False" in translate_field_type(field)
