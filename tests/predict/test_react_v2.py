from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types.tool import ToolCalls
from dspy.clients.base_lm import BaseLM
from dspy.dsp.utils.settings import settings
from dspy.dsp.utils.utils import dotdict
from dspy.predict.react_v2 import ReActV2
from dspy.utils.dummies import DummyLM


class ReasoningDummyLM(DummyLM):
    @property
    def supports_reasoning(self):
        return True


def test_react_v2_submit_tool_returns_original_output_fields():
    react = ReActV2("question -> answer", tools=[])  # ty:ignore[invalid-argument-type]

    assert react.tools["submit"](answer="Paris") == {"answer": "Paris"}
    assert "tool_call_results" not in react.react.signature.input_fields  # ty:ignore[unresolved-attribute]


def test_react_v2_text_mock_lm_loop_records_inputs_once():
    def lookup(query: str) -> str:
        return f"found {query}"

    lm = DummyLM(
        [
            {
                "next_thought": "I should look this up.",
                "tool_calls": ToolCalls.from_dict_list(
                    [{"name": "lookup", "args": {"query": "cats"}}]
                ),
            },
            {
                "next_thought": "I can answer now.",
                "tool_calls": ToolCalls.from_dict_list(
                    [{"name": "submit", "args": {"answer": "found cats"}}]
                ),
            },
        ]
    )

    with settings.context(lm=lm, adapter=ChatAdapter()):
        pred = ReActV2("question -> answer", tools=[lookup])(question="cats")  # ty:ignore[invalid-argument-type]

    assert pred.answer == "found cats"
    assert pred.termination_reason == "submit"
    assert sum("question" in event for event in pred.history.messages) == 1
    assert pred.history.messages[0]["tool_calls"].tool_calls[0].id == "call_0_0"
    assert "tool_call_results" not in pred.history.messages[0]
    assert pred.history.messages[0]["tool_calls"].tool_call_results.tool_call_results[0].call_id == "call_0_0"


def test_react_v2_continuation_omits_missing_original_inputs():
    def lookup(query: str) -> str:
        return f"found {query}"

    lm = DummyLM(
        [
            {
                "next_thought": "I should look this up.",
                "tool_calls": ToolCalls.from_dict_list(
                    [{"name": "lookup", "args": {"query": "cats"}}]
                ),
            },
            {
                "next_thought": "I can answer now.",
                "tool_calls": ToolCalls.from_dict_list(
                    [{"name": "submit", "args": {"answer": "found cats"}}]
                ),
            },
        ]
    )

    with settings.context(lm=lm, adapter=ChatAdapter()):
        pred = ReActV2("question -> answer", tools=[lookup])(question="cats")  # ty:ignore[invalid-argument-type]

    assert pred.answer == "found cats"
    second_call_messages = lm.history[1]["messages"]
    second_current_user_message = second_call_messages[-1]["content"]
    assert "[[ ## question ## ]]\nNone" not in second_current_user_message
    assert "[[ ## question ## ]]" not in second_current_user_message
    assert any("[[ ## question ## ]]\ncats" in message["content"] for message in second_call_messages)


def test_react_v2_text_mode_accepts_top_level_tool_arguments():
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
                "tool_calls": ToolCalls.from_dict_list(
                    [{"name": "submit", "args": {"answer": "found cats"}}]
                ),
            },
        ]
    )

    with settings.context(lm=lm, adapter=ChatAdapter(use_native_function_calling=False)):
        pred = ReActV2("question -> answer", tools=[lookup])(question="cats")  # ty:ignore[invalid-argument-type]

    assert pred.answer == "found cats"
    assert pred.termination_reason == "submit"
    assert pred.history.messages[0]["tool_calls"].tool_calls[0].args == {"query": "cats"}


def test_react_v2_text_mode_accepts_wrapped_submit_arguments():
    lm = DummyLM(
        [
            {
                "next_thought": "I can answer now.",
                "tool_calls": {"tool_calls": [{"name": "submit", "arguments": {"answer": "done"}}]},
            },
        ]
    )

    with settings.context(lm=lm, adapter=ChatAdapter(use_native_function_calling=False)):
        pred = ReActV2("question -> answer", tools=[])(question="cats")  # ty:ignore[invalid-argument-type]

    assert pred.answer == "done"
    assert pred.termination_reason == "submit"


def test_react_v2_unknown_tool_observation_can_continue():
    lm = DummyLM(
        [
            {
                "next_thought": "Try a missing tool.",
                "tool_calls": ToolCalls.from_dict_list(
                    [{"name": "missing_tool", "args": {"query": "cats"}}]
                ),
            },
            {
                "next_thought": "Recover with a final answer.",
                "tool_calls": ToolCalls.from_dict_list(
                    [{"name": "submit", "args": {"answer": "done"}}]
                ),
            },
        ]
    )

    with settings.context(lm=lm, adapter=ChatAdapter()):
        pred = ReActV2("question -> answer", tools=[])(question="cats")  # ty:ignore[invalid-argument-type]

    first_result = pred.history.messages[0]["tool_calls"].tool_call_results.tool_call_results[0]
    assert first_result.is_error is True
    assert first_result.call_id == "call_0_0"
    assert "Unknown tool" in first_result.value
    assert pred.answer == "done"


def test_react_v2_accepts_serialized_history_input():
    lm = DummyLM(
        [
            {
                "next_thought": "I can answer.",
                "tool_calls": ToolCalls.from_dict_list(
                    [{"name": "submit", "args": {"answer": "done"}}]
                ),
            }
        ]
    )

    with settings.context(lm=lm, adapter=ChatAdapter()):
        pred = ReActV2("question -> answer", tools=[])(history={"messages": [{"question": "old"}]})  # ty:ignore[invalid-argument-type]

    assert pred.answer == "done"
    assert pred.history.messages[0] == {"question": "old"}
    assert all(event for event in pred.history.messages)


def test_react_v2_forced_submit_on_empty_tool_calls():
    lm = ReasoningDummyLM(
        [
            {"next_thought": "No action.", "tool_calls": ToolCalls(tool_calls=[])},
            {
                "next_thought": "Forced final.",
                "tool_calls": ToolCalls.from_dict_list(
                    [{"name": "submit", "args": {"answer": "forced"}}]
                ),
            },
        ]
    )

    with settings.context(lm=lm, adapter=ChatAdapter()):
        pred = ReActV2("question -> answer", tools=[])(question="cats")  # ty:ignore[invalid-argument-type]

    assert pred.answer == "forced"
    assert pred.termination_reason == "forced_submit"
    assert lm.history[0]["kwargs"]["reasoning_effort"] == "low"
    assert "tool_choice" not in lm.history[1]["kwargs"]
    assert lm.history[1]["kwargs"].get("reasoning_effort") is None


class NativeToolLM(BaseLM):
    def __init__(self):
        super().__init__("native-tool-lm", "chat", 0.0, 1000, True)
        self.calls = []

    @property
    def supports_function_calling(self):
        return True

    def forward(self, prompt=None, messages=None, **kwargs: object):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if len(self.calls) == 1:
            tool_call = dotdict(
                id="call_provider_1",
                type="function",
                function=dotdict(name="lookup", arguments='{"query":"cats"}'),
            )
        else:
            tool_call = dotdict(
                id="call_submit",
                type="function",
                function=dotdict(name="submit", arguments='{"answer":"found cats"}'),
            )

        return dotdict(
            choices=[
                dotdict(
                    message=dotdict(content=None, tool_calls=[tool_call]),
                    finish_reason="tool_calls",
                )
            ],
            usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model="native-tool-lm",
        )


class ParallelNativeToolLM(BaseLM):
    def __init__(self):
        super().__init__("parallel-native-tool-lm", "chat", 0.0, 1000, True)
        self.calls = []

    @property
    def supports_function_calling(self):
        return True

    def forward(self, prompt=None, messages=None, **kwargs: object):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if len(self.calls) == 1:
            tool_calls = [
                dotdict(
                    id="call_provider_1",
                    type="function",
                    function=dotdict(name="lookup", arguments='{"query":"cats"}'),
                ),
                dotdict(
                    id="call_provider_2",
                    type="function",
                    function=dotdict(name="lookup", arguments='{"query":"dogs"}'),
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

        return dotdict(
            choices=[
                dotdict(
                    message=dotdict(content=None, tool_calls=tool_calls),
                    finish_reason="tool_calls",
                )
            ],
            usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model="parallel-native-tool-lm",
        )


def test_react_v2_native_tool_loop_replays_tool_result_with_provider_id():
    def lookup(query: str) -> str:
        return f"found {query}"

    lm = NativeToolLM()

    with settings.context(lm=lm, adapter=ChatAdapter(use_native_function_calling=True)):
        pred = ReActV2("question -> answer", tools=[lookup])(question="cats")  # ty:ignore[invalid-argument-type]

    assert pred.answer == "found cats"
    assert pred.history.messages[0]["tool_calls"].tool_calls[0].id == "call_provider_1"
    assert "tool_call_results" not in pred.history.messages[0]
    assert pred.history.messages[0]["tool_calls"].tool_call_results.tool_call_results[0].call_id == "call_provider_1"
    assert any(
        message["role"] == "tool" and message["tool_call_id"] == "call_provider_1"
        for message in lm.calls[1]["messages"]
    )


def test_react_v2_native_parallel_tool_calls_are_requested_and_replayed():
    def lookup(query: str) -> str:
        return f"found {query}"

    lm = ParallelNativeToolLM()

    with settings.context(lm=lm, adapter=ChatAdapter(use_native_function_calling=True, parallel_tool_calls=True)):
        pred = ReActV2("question -> answer", tools=[lookup])(question="cats and dogs")  # ty:ignore[invalid-argument-type]

    assert pred.answer == "found cats and found dogs"
    assert lm.calls[0]["kwargs"]["parallel_tool_calls"] is True
    assert [call.id for call in pred.history.messages[0]["tool_calls"].tool_calls] == [
        "call_provider_1",
        "call_provider_2",
    ]
    assert [
        result.call_id
        for result in pred.history.messages[0]["tool_calls"].tool_call_results.tool_call_results
    ] == [
        "call_provider_1",
        "call_provider_2",
    ]
    assert [
        message["tool_call_id"]
        for message in lm.calls[1]["messages"]
        if message["role"] == "tool"
    ] == [
        "call_provider_1",
        "call_provider_2",
    ]
