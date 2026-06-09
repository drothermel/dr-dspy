from dspy.adapters.base.tool_calls import _provider_tool_call_to_tool_call_dict
from dspy.adapters.types.tool.tool_calls import normalize_tool_call_dict
from dspy.core.types import LMToolCallPart


def test_normalize_tool_call_dict_openai_function_shape():
    normalized = normalize_tool_call_dict(
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": '{"query": "cats"}'},
        }
    )
    assert normalized == {"id": "call_1", "name": "search", "args": {"query": "cats"}}


def test_normalize_tool_call_dict_repairs_invalid_json_arguments():
    normalized = normalize_tool_call_dict(
        {"name": "search", "args": "{'query': 'cats'}"},
        repair=True,
    )
    assert normalized["name"] == "search"
    assert normalized["args"] == {"query": "cats"}


def test_provider_tool_call_to_tool_call_dict_from_lm_tool_call_part():
    tool_call = LMToolCallPart(id="tc_1", name="search", args={"query": "dogs"})
    normalized = _provider_tool_call_to_tool_call_dict(tool_call)
    assert normalized == {"id": "tc_1", "name": "search", "args": {"query": "dogs"}}


def test_provider_tool_call_to_tool_call_dict_parses_raw_arguments_with_repair():
    tool_call = LMToolCallPart(
        id="tc_2",
        name="search",
        args={},
        provider_data={"raw_arguments": "{'query': 'birds'}"},
    )
    normalized = _provider_tool_call_to_tool_call_dict(tool_call)
    assert normalized["args"] == {"query": "birds"}
