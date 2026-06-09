import asyncio
from types import SimpleNamespace
from typing import cast

import pytest
from typing_extensions import override

from dspy.adapters.base.tool_calls import attach_tool_calls_to_parsed_value
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.two_step_adapter import TwoStepAdapter
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import ToolCalls
from dspy.clients.base_lm import BaseLM
from dspy.clients.openai_format.chat_request import message_to_openai_chat, request_messages_as_openai
from dspy.core.types import LMOutput, LMRequest, LMResponse, LMTextPart, LMThinkingPart, LMUsage
from dspy.errors import AdapterParseError
from dspy.history import TurnLog
from dspy.predict.predict import Predict
from dspy.task_spec import input_field, make_task_spec, output_field
from dspy.testing import DummyLM
from tests.adapters.conftest import format_messages_and_lm_kwargs
from tests.task_spec.helpers import ts


class RecordingTextLM(BaseLM):
    def __init__(self, texts: list[str], *, model: str = "openai/gpt-4o-mini", temperature: float = 1.0):
        super().__init__(model, "chat", temperature=temperature, max_tokens=1000)
        self.texts = list(texts)
        self.requests: list[LMRequest] = []

    @override
    async def aforward(self, request: LMRequest) -> LMResponse:
        self.requests.append(request)
        text = self.texts.pop(0) if self.texts else "No more responses"
        return LMResponse.from_text(text, model=self.model)


def test_two_step_adapter_format_exact_messages_for_simple_signature_with_demo(make_run):
    QA = ts("question -> answer", instructions="Given the fields `question`, produce the fields `answer`.")
    adapter = TwoStepAdapter(DummyLM([{"answer": "x"}]), extraction_adapter=ChatAdapter())
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=adapter, task_spec=QA, demos=[{"question": "Q1", "answer": "A1"}], inputs={"question": "Q2"}
    )
    expected_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can solve tasks based on user input.\nAs input, you will be provided with:\n1. `question` (str): The question.\nYour outputs must contain:\n1. `answer` (str): The answer.\nYou should lay out your outputs in detail so that your answer can be understood by another agent\nSpecific instructions: Given the fields `question`, produce the fields `answer`.",
        },
        {"role": "user", "content": "question: Q1"},
        {"role": "assistant", "content": "answer: A1"},
        {"role": "user", "content": "question: Q2"},
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_two_step_adapter_format_exact_messages_with_typed_outputs(make_run):
    TypedSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "count": output_field("count", type_=int, desc="The count."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `question`, produce the fields `count`, `answer`.",
    )
    adapter = TwoStepAdapter(DummyLM([{"count": 1, "answer": "x"}]), extraction_adapter=ChatAdapter())
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=adapter, task_spec=TypedSignature, demos=[], inputs={"question": "Q"}
    )
    expected_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can solve tasks based on user input.\nAs input, you will be provided with:\n1. `question` (str): The question.\nYour outputs must contain:\n1. `count` (int): The count.\n2. `answer` (str): The answer.\nYou should lay out your outputs in detail so that your answer can be understood by another agent\nSpecific instructions: Given the fields `question`, produce the fields `count`, `answer`.",
        },
        {"role": "user", "content": "question: Q"},
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_two_step_adapter_format_includes_turn_log_history(make_run):
    QAWithHistory = make_task_spec(
        {
            "turn_log": input_field("turn_log", type_=TurnLog, desc="The history."),
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `turn_log`, `question`, produce the fields `answer`.",
    )
    adapter = TwoStepAdapter(DummyLM([{"answer": "x"}]), extraction_adapter=ChatAdapter())
    history = TurnLog.model_validate({"turns": [{"question": "Q1", "answer": "A1"}]})
    messages, _ = format_messages_and_lm_kwargs(
        adapter=adapter,
        task_spec=QAWithHistory,
        demos=[],
        inputs={"turn_log": history, "question": "Q2"},
    )
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "question: Q1" in messages[1]["content"]
    assert messages[2]["role"] == "assistant"
    assert "answer: A1" in messages[2]["content"]
    assert messages[3]["role"] == "user"
    assert "question: Q2" in messages[3]["content"]


def test_two_step_adapter_call(make_run):
    TestSignature = make_task_spec(
        {
            "question": input_field("question", desc="The math question to solve"),
            "solution": output_field("solution", desc="Step by step solution"),
            "answer": output_field("answer", type_=float, desc="The final numerical answer"),
        },
        instructions="Given the fields `question`, produce the fields `solution`, `answer`.",
    )
    program = Predict(TestSignature)
    main_lm = RecordingTextLM(["text from main LM"])
    extraction_lm = DummyLM([{"solution": "result", "answer": 12}])
    run = make_run(lm=main_lm, adapter=TwoStepAdapter(extraction_model=extraction_lm, extraction_adapter=ChatAdapter()))
    result = asyncio.run(program(question="What is 5 + 7?", run=run))
    assert result.answer == 12
    main_messages = [message_to_openai_chat(message) for message in main_lm.requests[0].messages]
    assert len(main_messages) == 2
    assert main_messages[0]["role"] == "system"
    content = main_messages[0]["content"]
    assert "1. `question` (str)" in content
    assert "1. `solution` (str)" in content
    assert "2. `answer` (float)" in content
    assert main_messages[1]["role"] == "user"
    content = main_messages[1]["content"]
    assert "question:" in content.lower()
    assert "What is 5 + 7?" in content
    extraction_messages = request_messages_as_openai(extraction_lm.call_log[0].request)
    assert len(extraction_messages) == 2
    assert extraction_messages[0]["role"] == "system"
    content = extraction_messages[0]["content"]
    assert "`text` (str)" in content
    assert "`solution` (str)" in content
    assert "`answer` (float)" in content
    assert extraction_messages[1]["role"] == "user"
    content = extraction_messages[1]["content"]
    assert "text from main LM" in content


@pytest.mark.asyncio
async def test_two_step_adapter_async_call(make_run):
    TestSignature = make_task_spec(
        {
            "question": input_field("question", desc="The math question to solve"),
            "solution": output_field("solution", desc="Step by step solution"),
            "answer": output_field("answer", type_=float, desc="The final numerical answer"),
        },
        instructions="Given the fields `question`, produce the fields `solution`, `answer`.",
    )
    program = Predict(TestSignature)
    main_lm = RecordingTextLM(["text from main LM"])
    extraction_lm = DummyLM([{"solution": "result", "answer": 12}])
    run = make_run(lm=main_lm, adapter=TwoStepAdapter(extraction_model=extraction_lm, extraction_adapter=ChatAdapter()))
    result = await program(question="What is 5 + 7?", run=run)
    assert result.answer == 12
    main_messages = [message_to_openai_chat(message) for message in main_lm.requests[0].messages]
    assert len(main_messages) == 2
    assert main_messages[0]["role"] == "system"
    content = main_messages[0]["content"]
    assert "1. `question` (str)" in content
    assert "1. `solution` (str)" in content
    assert "2. `answer` (float)" in content
    assert main_messages[1]["role"] == "user"
    content = main_messages[1]["content"]
    assert "question:" in content.lower()
    assert "What is 5 + 7?" in content
    extraction_messages = request_messages_as_openai(extraction_lm.call_log[0].request)
    assert len(extraction_messages) == 2
    assert extraction_messages[0]["role"] == "system"
    content = extraction_messages[0]["content"]
    assert "`text` (str)" in content
    assert "`solution` (str)" in content
    assert "`answer` (float)" in content
    assert extraction_messages[1]["role"] == "user"
    content = extraction_messages[1]["content"]
    assert "text from main LM" in content


@pytest.mark.asyncio
async def test_two_step_adapter_extraction(make_run):
    ComplexSignature = make_task_spec(
        {
            "input_text": input_field("input_text", desc="Source text to tag"),
            "tags": output_field("tags", type_=list[str], desc="List of relevant tags"),
            "confidence": output_field("confidence", type_=float, desc="Confidence score"),
        },
        instructions="Given the fields `input_text`, produce the fields `tags`, `confidence`.",
    )
    first_response = "main LM response"
    extraction_lm = DummyLM([{"tags": ["AI", "deep learning", "neural networks"], "confidence": 0.87}])
    adapter = TwoStepAdapter(extraction_lm, extraction_adapter=ChatAdapter())
    run = make_run(lm=extraction_lm, adapter=adapter)
    result = await adapter._run_extraction(original_task_spec=ComplexSignature, text=first_response, run=run)
    assert result["tags"] == ["AI", "deep learning", "neural networks"]
    assert result["confidence"] == 0.87


@pytest.mark.asyncio
async def test_two_step_adapter_native_reasoning_from_main_output(make_run):
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "reasoning": output_field("reasoning", type_=Reasoning, desc="The reasoning."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `question`, produce the fields `reasoning`, `answer`.",
    )

    class MainLM(BaseLM):
        def __init__(self):
            super().__init__("dummy", "chat", temperature=0.0, max_tokens=1000)
            self.kwargs["reasoning"] = {"effort": "low"}

        @property
        @override
        def supports_reasoning(self):
            return True

        @override
        async def aforward(self, request: LMRequest) -> LMResponse:
            _ = request
            return LMResponse(
                model=self.model,
                outputs=[
                    LMOutput(
                        parts=[
                            LMTextPart(text="answer: Paris"),
                            LMThinkingPart(text="Native provider reasoning"),
                        ]
                    )
                ],
                usage=LMUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            )

    main_lm = MainLM()
    extraction_lm = DummyLM([{"answer": "Paris"}])
    adapter = TwoStepAdapter(extraction_lm, extraction_adapter=ChatAdapter(), use_native_function_calling=True)
    run = make_run(lm=main_lm, adapter=adapter)
    results = await adapter(
        lm=main_lm,
        config={},
        task_spec=task_spec,
        demos=[],
        inputs={"question": "What is the capital of France?"},
        run=run,
    )
    assert results[0]["reasoning"] == Reasoning(content="Native provider reasoning")
    assert results[0]["answer"] == "Paris"


def test_attach_tool_calls_preserves_call_id_from_provider_dict():
    adapter = ChatAdapter()
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, produce the fields `tool_calls`.",
    )

    output = SimpleNamespace(
        tool_calls=[
            {
                "function": {"name": "search", "arguments": '{"query": "cats",}'},
                "call_id": "call_from_two_step",
                "type": "function",
            }
        ]
    )

    value = attach_tool_calls_to_parsed_value(
        value={},
        output=cast("LMOutput", output),
        tool_call_output_field_name=adapter._get_tool_call_output_field_name(task_spec),
    )
    assert value["tool_calls"] == ToolCalls(
        tool_calls=[ToolCalls.ToolCall(id="call_from_two_step", name="search", args={"query": "cats"})]
    )


@pytest.mark.asyncio
async def test_two_step_pipeline_extraction_error_not_double_wrapped(make_run):
    TestSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question to answer"),
            "answer": output_field("answer", desc="The answer to the question"),
        },
        instructions="Given the fields `question`, produce the fields `answer`.",
    )
    main_lm = RecordingTextLM(["main LM response"])
    extraction_lm = RecordingTextLM(["not parseable extraction output"])
    adapter = TwoStepAdapter(extraction_lm, extraction_adapter=ChatAdapter())
    run = make_run(lm=main_lm, adapter=adapter)
    program = Predict(TestSignature)
    with pytest.raises(AdapterParseError) as pipeline_error:
        await program(question="Q?", run=run)
    with pytest.raises(AdapterParseError) as direct_error:
        await adapter._run_extraction(original_task_spec=TestSignature, text="main LM response", run=run)
    assert pipeline_error.value.task_spec == direct_error.value.task_spec
    assert "Failed to parse response from the original completion" not in str(pipeline_error.value)


@pytest.mark.asyncio
async def test_two_step_adapter_extraction_errors(make_run):
    TestSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question to answer"),
            "answer": output_field("answer", desc="The answer to the question"),
        },
        instructions="Given the fields `question`, produce the fields `answer`.",
    )
    first_response = "main LM response"
    extraction_lm = RecordingTextLM(["not parseable extraction output"])
    adapter = TwoStepAdapter(extraction_lm, extraction_adapter=ChatAdapter())
    run = make_run(lm=extraction_lm, adapter=adapter)
    with pytest.raises(AdapterParseError):
        await adapter._run_extraction(original_task_spec=TestSignature, text=first_response, run=run)
