from dspy.adapters.types.tool import ToolCallResults, ToolCalls
from dspy.clients.openai_format import to_openai_chat_request
from dspy.core.types import LMMessage, LMRequest, LMToolResultPart


def test_tool_call_results_can_round_trip_as_native_tool_result_message():
    tool_call = ToolCalls.ToolCall(id="call_1", name="search", args={"query": "cats"})
    results = ToolCallResults.from_tool_calls_and_values([tool_call], ['{"items": ["cat"]}'])
    result = results.tool_call_results[0]

    message = LMMessage(role="tool", tool_call_id=result.call_id, name=result.name, content=result.value)  # ty:ignore[missing-argument, unknown-argument]

    assert len(message.parts) == 1
    assert isinstance(message.parts[0], LMToolResultPart)
    assert message.parts[0].call_id == "call_1"
    assert message.parts[0].name == "search"
    assert message.parts[0].content[0].text == '{"items": ["cat"]}'  # ty:ignore[unresolved-attribute]

    request = LMRequest(model="test-model", messages=[message])
    assert to_openai_chat_request(request)["messages"] == [
        {"role": "tool", "content": '{"items": ["cat"]}', "tool_call_id": "call_1", "name": "search"}
    ]
