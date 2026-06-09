from dspy.clients.openai_format.tool_calls import tool_call_part_to_openai
from dspy.core.types import LMToolCallPart


def test_tool_call_part_to_openai_preserves_canonical_fields_over_provider_data():
    call = LMToolCallPart(
        id="call_1",
        name="search",
        args={"query": "dspy"},
        provider_data={"type": "custom", "id": "override", "function": {"name": "wrong", "arguments": "{}"}},
    )
    payload = tool_call_part_to_openai(call)
    assert payload["type"] == "function"
    assert payload["id"] == "call_1"
    assert payload["function"] == {"name": "search", "arguments": '{"query": "dspy"}'}
