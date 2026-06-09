from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.format_field_structure import build_role_field_sections
from dspy.adapters.format_shared import output_field_type_hint
from dspy.adapters.types.tool import ToolCalls
from dspy.task_spec.field_spec import FieldRole
from tests.adapters.scenarios.simple_qa import SIMPLE_QA_SIGNATURE


def test_output_field_type_hint_for_tool_calls():
    assert "tool_calls" in output_field_type_hint(ToolCalls)


def test_output_field_type_hint_for_non_string():
    assert "int" in output_field_type_hint(int)


def test_output_field_type_hint_for_string_is_empty():
    assert output_field_type_hint(str) == ""


def test_build_role_field_sections_uses_adapter_formatter():
    adapter = ChatAdapter()
    section = build_role_field_sections(adapter, SIMPLE_QA_SIGNATURE, FieldRole.INPUT)
    assert "[[ ## question ## ]]" in section
    assert "{question}" in section
