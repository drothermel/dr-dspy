import pytest

from dspy.adapters.types.tool import ToolCallResults, ToolCalls
from dspy.history import TurnEvent, TurnLog


def test_turn_log_rejects_invalid_list_items():
    with pytest.raises(TypeError, match="turns\\[1\\]"):
        TurnLog.model_validate({"turns": [{"thought": "ok"}, "bad-item"]})


def test_turn_log_serialization_round_trip_with_nested_tool_calls():
    tool_call = ToolCalls.ToolCall(id="call_1", name="search", args={"query": "hello"})
    results = ToolCallResults.from_tool_calls_and_values([tool_call], [{"answer": "world"}])
    tool_calls = ToolCalls(tool_calls=[tool_call], tool_call_results=results)
    history = TurnLog(turns=(TurnEvent(tool_calls=tool_calls),))
    dumped = history.model_dump(mode="json")
    restored = TurnLog.model_validate(dumped)
    assert dumped == {
        "turns": [
            {
                "tool_calls": {
                    "tool_calls": [{"id": "call_1", "name": "search", "args": {"query": "hello"}}],
                    "tool_call_results": {
                        "tool_call_results": [
                            {"call_id": "call_1", "name": "search", "value": {"answer": "world"}, "is_error": False}
                        ]
                    },
                }
            }
        ]
    }
    assert isinstance(restored.turns[0], TurnEvent)
    assert restored.turns[0].tool_calls is not None
