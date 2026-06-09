import pytest
from pydantic import TypeAdapter

from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from tests.adapters.types.tool.conftest import requires_jsonschema

TOOL_CALL_TEST_CASES = [
    ([], {"tool_calls": []}),
    (
        [{"name": "search", "args": {"query": "hello"}}],
        {"tool_calls": [{"name": "search", "args": {"query": "hello"}}]},
    ),
    (
        [
            {"name": "search", "args": {"query": "hello"}},
            {"name": "translate", "args": {"text": "world", "lang": "fr"}},
        ],
        {
            "tool_calls": [
                {"name": "search", "args": {"query": "hello"}},
                {"name": "translate", "args": {"text": "world", "lang": "fr"}},
            ]
        },
    ),
    ([{"name": "get_time", "args": {}}], {"tool_calls": [{"name": "get_time", "args": {}}]}),
]


@pytest.mark.parametrize(("tool_calls_data", "expected"), TOOL_CALL_TEST_CASES)
def test_tool_calls_format_basic(tool_calls_data, expected):
    tool_calls_list = [ToolCalls.ToolCall(**data) for data in tool_calls_data]
    tool_calls = ToolCalls(tool_calls=tool_calls_list)
    result = tool_calls.format()
    assert result == expected


def test_tool_calls_format_from_dict_list():
    tool_calls_dicts = [
        {"name": "search", "args": {"query": "hello"}},
        {"name": "translate", "args": {"text": "world", "lang": "fr"}},
    ]
    tool_calls = ToolCalls.from_dict_list(tool_calls_dicts)
    result = tool_calls.format()
    assert len(result["tool_calls"]) == 2
    assert result["tool_calls"][0]["name"] == "search"
    assert result["tool_calls"][1]["name"] == "translate"


def test_tool_calls_preserves_ids_from_dict_list_and_format_includes_set_ids():
    tool_calls = ToolCalls.from_dict_list(
        [
            {"id": "call_1", "name": "search", "args": {"query": "hello"}},
            {"name": "translate", "args": {"text": "world", "lang": "fr"}},
        ]
    )
    assert tool_calls.tool_calls[0].id == "call_1"
    assert tool_calls.tool_calls[1].id is None
    formatted = tool_calls.format()["tool_calls"]
    assert formatted[0]["id"] == "call_1"
    assert "id" not in formatted[1]


def test_tool_calls_json_schema_omits_internal_id_field():
    schema = TypeAdapter(ToolCalls).json_schema()
    assert "tool_call_results" not in schema["properties"]
    assert "id" not in schema["$defs"]["ToolCall"]["properties"]
    assert schema["$defs"]["ToolCall"]["required"] == ["name", "args"]


def test_tool_calls_can_carry_results_without_formatting_them_for_lm():
    tool_call = ToolCalls.ToolCall(id="call_1", name="search", args={"query": "hello"})
    results = ToolCallResults.from_tool_calls_and_values([tool_call], [{"answer": "world"}])
    tool_calls = ToolCalls(tool_calls=[tool_call], tool_call_results=results)
    assert "tool_call_results" not in tool_calls.format()
    assert tool_calls.model_dump(mode="json")["tool_call_results"] == {
        "tool_call_results": [{"call_id": "call_1", "name": "search", "value": {"answer": "world"}, "is_error": False}]
    }


def test_toolcalls_vague_match():
    data_single = {"name": "search", "args": {"query": "hello"}}
    tc = ToolCalls.model_validate(data_single)
    assert isinstance(tc, ToolCalls)
    assert len(tc.tool_calls) == 1
    assert tc.tool_calls[0].name == "search"
    assert tc.tool_calls[0].args == {"query": "hello"}
    data_list = [
        {"name": "search", "args": {"query": "hello"}},
        {"name": "translate", "args": {"text": "world", "lang": "fr"}},
    ]
    tc = ToolCalls.model_validate(data_list)
    assert isinstance(tc, ToolCalls)
    assert len(tc.tool_calls) == 2
    assert tc.tool_calls[0].name == "search"
    assert tc.tool_calls[1].name == "translate"
    data_tool_calls = {"tool_calls": [{"name": "search", "args": {"query": "hello"}}, {"name": "get_time", "args": {}}]}
    tc = ToolCalls.model_validate(data_tool_calls)
    assert isinstance(tc, ToolCalls)
    assert len(tc.tool_calls) == 2
    assert tc.tool_calls[0].name == "search"
    assert tc.tool_calls[1].name == "get_time"
    tc = ToolCalls.model_validate({"name": "search", "arguments": {"query": "hello"}})
    assert len(tc.tool_calls) == 1
    assert tc.tool_calls[0].name == "search"
    assert tc.tool_calls[0].args == {"query": "hello"}
    tc = ToolCalls.model_validate(
        [{"name": "search", "arguments": {"query": "hello"}}, {"name": "get_time", "arguments": {}}]
    )
    assert len(tc.tool_calls) == 2
    assert tc.tool_calls[0].args == {"query": "hello"}
    assert tc.tool_calls[1].args == {}
    tc = ToolCalls.model_validate({"tool_calls": [{"name": "search", "arguments": {"query": "hello"}}]})
    assert len(tc.tool_calls) == 1
    assert tc.tool_calls[0].args == {"query": "hello"}
    tc = ToolCalls.model_validate({"type": "function", "function": {"name": "search", "arguments": {"query": "hello"}}})
    assert len(tc.tool_calls) == 1
    assert tc.tool_calls[0].args == {"query": "hello"}
    tc = ToolCalls.model_validate(
        {"type": "function", "function": {"name": "search", "arguments": '{"query":"hello"}'}}
    )
    assert len(tc.tool_calls) == 1
    assert tc.tool_calls[0].args == {"query": "hello"}
    with pytest.raises(ValueError, match=r"Received invalid value"):
        ToolCalls.model_validate({"foo": "bar"})
    with pytest.raises(ValueError, match=r"Received invalid value"):
        ToolCalls.model_validate([{"foo": "bar"}])
    with pytest.raises(ValueError, match="function value"):
        ToolCalls.from_dict_list([{"function": "bad"}])


@requires_jsonschema
def test_tool_call_execute():

    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny"

    def add_numbers(a: int, b: int) -> int:
        return a + b

    tools = [
        Tool(get_weather, description="Get the weather for a city"),
        Tool(add_numbers, description="Add two numbers."),
    ]
    tool_call = ToolCalls.ToolCall(name="get_weather", args={"city": "Berlin"})
    result = tool_call.execute(functions=tools)
    assert result == "The weather in Berlin is sunny"
    tool_call2 = ToolCalls.ToolCall(name="add_numbers", args={"a": 7, "b": 13})
    result2 = tool_call2.execute(functions={"add_numbers": add_numbers})
    assert result2 == 20

    def get_pi():
        return 3.14159

    tool_call3 = ToolCalls.ToolCall(name="get_pi", args={})
    result3 = tool_call3.execute(functions={"get_pi": get_pi})
    assert result3 == 3.14159
    tool_call4 = ToolCalls.ToolCall(name="nonexistent", args={})
    with pytest.raises(ValueError, match=r"not found") as exc_info:
        tool_call4.execute(functions=tools)
    assert "not found" in str(exc_info.value)


@requires_jsonschema
def test_tool_call_execute_requires_explicit_functions():
    tool_call = ToolCalls.ToolCall(name="local_add", args={"a": 1, "b": 2})
    with pytest.raises(TypeError, match="required positional argument"):
        tool_call.execute()  # ty: ignore[missing-argument]
