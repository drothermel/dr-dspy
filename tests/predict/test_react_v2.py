import asyncio
import json

from typing_extensions import override

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.clients.base_lm import BaseLM
from dspy.clients.openai_format import message_to_openai_chat
from dspy.core.types import LMOutput, LMRequest, LMResponse, LMToolCallPart
from dspy.predict.react_v2 import ReActV2
from dspy.utils.dotdict import dotdict
from dspy.utils.dummies import DummyLM
from tests.adapters.conftest import captured_lm_kwargs
from tests.task_spec.helpers import ts


class ReasoningDummyLM(DummyLM):
    @property
    @override
    def supports_reasoning(self):
        return True


def test_react_v2_submit_tool_returns_original_output_fields(make_run):
    react = ReActV2(ts("question -> answer"), tools=[])
    assert react.tools["submit"](answer="Paris") == {"answer": "Paris"}
    assert "tool_call_results" not in react.react.task_spec.input_fields


def test_react_v2_text_mock_lm_loop_records_inputs_once(make_run):

    def lookup(query: str) -> str:
        return f"found {query}"

    lm = DummyLM(
        [
            {
                "next_thought": "I should look this up.",
                "tool_calls": ToolCalls.from_dict_list([{"name": "lookup", "args": {"query": "cats"}}]),
            },
            {
                "next_thought": "I can answer now.",
                "tool_calls": ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "found cats"}}]),
            },
        ]
    )
    run = make_run(lm=lm, adapter=ChatAdapter())
    pred = asyncio.run(
        ReActV2(ts("question -> answer"), tools=[Tool(lookup, description="Look up a query.")])(
            question="cats", run=run
        )
    )
    assert pred.answer == "found cats"
    assert pred.termination_reason == "submit"
    assert sum("question" in event for event in pred.turn_log.turns) == 1
    assert pred.turn_log.turns[0]["tool_calls"].tool_calls[0].id == "call_0_0"
    assert "tool_call_results" not in pred.turn_log.turns[0]
    assert pred.turn_log.turns[0]["tool_calls"].tool_call_results.tool_call_results[0].call_id == "call_0_0"


def test_react_v2_continuation_omits_missing_original_inputs(make_run):

    def lookup(query: str) -> str:
        return f"found {query}"

    lm = DummyLM(
        [
            {
                "next_thought": "I should look this up.",
                "tool_calls": ToolCalls.from_dict_list([{"name": "lookup", "args": {"query": "cats"}}]),
            },
            {
                "next_thought": "I can answer now.",
                "tool_calls": ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "found cats"}}]),
            },
        ]
    )
    run = make_run(lm=lm, adapter=ChatAdapter())
    pred = asyncio.run(
        ReActV2(ts("question -> answer"), tools=[Tool(lookup, description="Look up a query.")])(
            question="cats", run=run
        )
    )
    assert pred.answer == "found cats"
    second_call_messages = lm.call_log[1].messages_as_openai
    second_current_user_message = second_call_messages[-1]["content"]
    assert "[[ ## question ## ]]\nNone" not in second_current_user_message
    assert "[[ ## question ## ]]" not in second_current_user_message
    assert any("[[ ## question ## ]]\ncats" in message["content"] for message in second_call_messages)


def test_react_v2_text_mode_accepts_top_level_tool_arguments(make_run):

    def lookup(query: str) -> str:
        return f"found {query}"

    lm = DummyLM(
        [
            {
                "next_thought": "I should look this up.",
                "tool_calls": {"name": "lookup", "arguments": {"query": "cats"}},
            },
            {
                "next_thought": "I can answer now.",
                "tool_calls": ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "found cats"}}]),
            },
        ]
    )
    run = make_run(lm=lm, adapter=ChatAdapter(use_native_function_calling=False))
    pred = asyncio.run(
        ReActV2(ts("question -> answer"), tools=[Tool(lookup, description="Look up a query.")])(
            question="cats", run=run
        )
    )
    assert pred.answer == "found cats"
    assert pred.termination_reason == "submit"
    assert pred.turn_log.turns[0]["tool_calls"].tool_calls[0].args == {"query": "cats"}


def test_react_v2_text_mode_accepts_wrapped_submit_arguments(make_run):
    lm = DummyLM(
        [
            {
                "next_thought": "I can answer now.",
                "tool_calls": {"tool_calls": [{"name": "submit", "arguments": {"answer": "done"}}]},
            }
        ]
    )
    run = make_run(lm=lm, adapter=ChatAdapter(use_native_function_calling=False))
    pred = asyncio.run(ReActV2(ts("question -> answer"), tools=[])(question="cats", run=run))
    assert pred.answer == "done"
    assert pred.termination_reason == "submit"


def test_react_v2_unknown_tool_observation_can_continue(make_run):
    lm = DummyLM(
        [
            {
                "next_thought": "Try a missing tool.",
                "tool_calls": ToolCalls.from_dict_list([{"name": "missing_tool", "args": {"query": "cats"}}]),
            },
            {
                "next_thought": "Recover with a final answer.",
                "tool_calls": ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "done"}}]),
            },
        ]
    )
    run = make_run(lm=lm, adapter=ChatAdapter())
    pred = asyncio.run(ReActV2(ts("question -> answer"), tools=[])(question="cats", run=run))
    first_result = pred.turn_log.turns[0]["tool_calls"].tool_call_results.tool_call_results[0]
    assert first_result.is_error is True
    assert first_result.call_id == "call_0_0"
    assert "Unknown tool" in first_result.value
    assert pred.answer == "done"


def test_react_v2_accepts_serialized_history_input(make_run):
    lm = DummyLM(
        [
            {
                "next_thought": "I can answer.",
                "tool_calls": ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "done"}}]),
            }
        ]
    )
    run = make_run(lm=lm, adapter=ChatAdapter())
    pred = asyncio.run(
        ReActV2(ts("question -> answer"), tools=[])(turn_log={"turns": [{"question": "old"}]}, run=run)
    )
    assert pred.answer == "done"
    assert pred.turn_log.turns[0] == {"question": "old"}
    assert all(event for event in pred.turn_log.turns)


def test_react_v2_forced_submit_on_empty_tool_calls(make_run):
    lm = ReasoningDummyLM(
        [
            {"next_thought": "No action.", "tool_calls": ToolCalls(tool_calls=[])},
            {
                "next_thought": "Forced final.",
                "tool_calls": ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "forced"}}]),
            },
        ]
    )
    lm.kwargs["reasoning_effort"] = "low"
    run = make_run(lm=lm, adapter=ChatAdapter())
    pred = asyncio.run(ReActV2(ts("question -> answer"), tools=[])(question="cats", run=run))
    assert pred.answer == "forced"
    assert pred.termination_reason == "forced_submit"
    reasoning = lm.call_log[0].request.config.reasoning
    assert reasoning is not None
    assert reasoning.effort == "low"
    assert lm.call_log[1].request.config.reasoning is None
    assert lm.call_log[1].request.config.tool_choice is None


class NativeToolLM(BaseLM):
    def __init__(self):
        super().__init__("native-tool-lm", "chat", temperature=0.0, max_tokens=1000)
        self.calls = []

    @property
    @override
    def supports_function_calling(self):
        return True

    @override
    async def aforward(self, request: LMRequest) -> LMResponse:
        self.calls.append({"messages": request.messages, "kwargs": captured_lm_kwargs(request)})
        if len(self.calls) == 1:
            tool_call = dotdict(
                id="call_provider_1", type="function", function=dotdict(name="lookup", arguments='{"query":"cats"}')
            )
        else:
            tool_call = dotdict(
                id="call_submit", type="function", function=dotdict(name="submit", arguments='{"answer":"found cats"}')
            )
        args = json.loads(tool_call.function.arguments)
        return LMResponse(
            model="native-tool-lm",
            outputs=[
                LMOutput(
                    parts=[LMToolCallPart(id=tool_call.id, name=tool_call.function.name, args=args)],
                    finish_reason="tool_calls",
                )
            ],
            usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )


class ParallelNativeToolLM(BaseLM):
    def __init__(self):
        super().__init__("parallel-native-tool-lm", "chat", temperature=0.0, max_tokens=1000)
        self.calls = []

    @property
    @override
    def supports_function_calling(self):
        return True

    @override
    async def aforward(self, request: LMRequest) -> LMResponse:
        self.calls.append({"messages": request.messages, "kwargs": captured_lm_kwargs(request)})
        if len(self.calls) == 1:
            tool_calls = [
                dotdict(
                    id="call_provider_1", type="function", function=dotdict(name="lookup", arguments='{"query":"cats"}')
                ),
                dotdict(
                    id="call_provider_2", type="function", function=dotdict(name="lookup", arguments='{"query":"dogs"}')
                ),
            ]
        else:
            tool_calls = [
                dotdict(
                    id="call_submit",
                    type="function",
                    function=dotdict(name="submit", arguments='{"answer":"found cats and found dogs"}'),
                )
            ]
        return LMResponse(
            model="parallel-native-tool-lm",
            outputs=[
                LMOutput(
                    parts=[
                        LMToolCallPart(
                            id=tool_call.id, name=tool_call.function.name, args=json.loads(tool_call.function.arguments)
                        )
                        for tool_call in tool_calls
                    ],
                    finish_reason="tool_calls",
                )
            ],
            usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )


def test_react_v2_native_tool_loop_replays_tool_result_with_provider_id(make_run):

    def lookup(query: str) -> str:
        return f"found {query}"

    lm = NativeToolLM()
    run = make_run(lm=lm, adapter=ChatAdapter(use_native_function_calling=True))
    pred = asyncio.run(
        ReActV2(ts("question -> answer"), tools=[Tool(lookup, description="Look up a query.")])(
            question="cats", run=run
        )
    )
    assert pred.answer == "found cats"
    assert pred.turn_log.turns[0]["tool_calls"].tool_calls[0].id == "call_provider_1"
    assert "tool_call_results" not in pred.turn_log.turns[0]
    assert pred.turn_log.turns[0]["tool_calls"].tool_call_results.tool_call_results[0].call_id == "call_provider_1"
    assert any(
        message["role"] == "tool" and message["tool_call_id"] == "call_provider_1"
        for message in (message_to_openai_chat(item) for item in lm.calls[1]["messages"])
    )


def test_react_v2_native_parallel_tool_calls_are_requested_and_replayed(make_run):

    def lookup(query: str) -> str:
        return f"found {query}"

    lm = ParallelNativeToolLM()
    run = make_run(lm=lm, adapter=ChatAdapter(use_native_function_calling=True, parallel_tool_calls=True))
    pred = asyncio.run(
        ReActV2(ts("question -> answer"), tools=[Tool(lookup, description="Look up a query.")])(
            question="cats and dogs",
            run=run,
        )
    )
    assert pred.answer == "found cats and found dogs"
    assert lm.calls[0]["kwargs"]["parallel_tool_calls"] is True
    assert [call.id for call in pred.turn_log.turns[0]["tool_calls"].tool_calls] == [
        "call_provider_1",
        "call_provider_2",
    ]
    assert [
        result.call_id for result in pred.turn_log.turns[0]["tool_calls"].tool_call_results.tool_call_results
    ] == ["call_provider_1", "call_provider_2"]
    assert [
        message["tool_call_id"]
        for message in (message_to_openai_chat(item) for item in lm.calls[1]["messages"])
        if message["role"] == "tool"
    ] == ["call_provider_1", "call_provider_2"]
