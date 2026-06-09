from typing_extensions import override

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.core.types.config import LMConfig, LMToolChoice
from dspy.task_spec import input_field, make_task_spec, output_field
from dspy.testing import DummyLM


def _native_tool_task_spec():
    return make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `tool_calls`.",
    )


class FunctionCallingLM(DummyLM):
    @property
    @override
    def supports_function_calling(self):
        return True


def _preprocess_config(adapter: ChatAdapter, *, config: LMConfig | dict) -> LMConfig:
    def search(query: str) -> str:
        return query

    _, _, resolved = adapter._call_preprocess(
        lm=FunctionCallingLM([{}]),
        config=config,
        task_spec=_native_tool_task_spec(),
        inputs={"question": "Q?", "tools": [Tool(search, description="Search for documents.")]},
    )
    return resolved


def test_parallel_tool_calls_sets_auto_tool_choice_when_missing():
    adapter = ChatAdapter(use_native_function_calling=True, parallel_tool_calls=True)
    config = _preprocess_config(adapter, config=LMConfig())
    assert config.tool_choice == LMToolChoice(mode="auto", parallel=True)


def test_parallel_tool_calls_merges_when_tool_choice_has_no_parallel():
    adapter = ChatAdapter(use_native_function_calling=True, parallel_tool_calls=True)
    config = _preprocess_config(
        adapter,
        config=LMConfig(tool_choice=LMToolChoice(mode="required", allowed=["search"])),
    )
    assert config.tool_choice == LMToolChoice(mode="required", allowed=["search"], parallel=True)


def test_parallel_tool_calls_respects_explicit_parallel_on_tool_choice():
    adapter = ChatAdapter(use_native_function_calling=True, parallel_tool_calls=True)
    config = _preprocess_config(
        adapter,
        config=LMConfig(tool_choice=LMToolChoice(mode="auto", parallel=False)),
    )
    assert config.tool_choice == LMToolChoice(mode="auto", parallel=False)
