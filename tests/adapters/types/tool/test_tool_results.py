from dspy.history import TurnLog
from dspy.adapters.types.tool import ToolCallResults, ToolCalls


def test_tool_call_results_from_tool_calls_and_values():
    tool_calls = [
        ToolCalls.ToolCall(id="call_1", name="search", args={"query": "hello"}),
        ToolCalls.ToolCall(id="call_2", name="fetch", args={"url": "https://example.com"}),
    ]
    results = ToolCallResults.from_tool_calls_and_values(
        tool_calls, [{"items": [1, 2]}, "failed"], is_errors=[False, True]
    )
    assert results.tool_call_results[0].call_id == "call_1"
    assert results.tool_call_results[0].name == "search"
    assert results.tool_call_results[0].value == {"items": [1, 2]}
    assert results.tool_call_results[0].is_error is False
    assert results.tool_call_results[1].call_id == "call_2"
    assert results.tool_call_results[1].name == "fetch"
    assert results.tool_call_results[1].value == "failed"
    assert results.tool_call_results[1].is_error is True


def test_tool_call_results_history_serialization_round_trip():
    tool_call = ToolCalls.ToolCall(id="call_1", name="search", args={"query": "hello"})
    results = ToolCallResults.from_tool_calls_and_values([tool_call], [{"answer": "world"}])
    tool_calls = ToolCalls(tool_calls=[tool_call], tool_call_results=results)
    history = TurnLog(turns=({"tool_calls": tool_calls}))
    dumped = history.model_dump(mode="json")
    restored = TurnLog.model_validate(dumped)
    assert dumped == {
        "turns": [
            {
                "tool_calls": {
                    "tool_calls": [{"name": "search", "args": {"query": "hello"}}],
                    "tool_call_results": {
                        "tool_call_results": [
                            {"call_id": "call_1", "name": "search", "value": {"answer": "world"}, "is_error": False}
                        ]
                    },
                }
            }
        ]
    }
    assert restored.turns == tuple(dumped["turns"])
