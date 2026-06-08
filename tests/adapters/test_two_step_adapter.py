import asyncio

import pytest
from typing_extensions import override

from dspy.adapters.two_step_adapter import TwoStepAdapter
from dspy.clients.base_lm import BaseLM
from dspy.clients.openai_format import message_to_openai_chat
from dspy.core.types import LMRequest, LMResponse
from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Predict
from dspy.task_spec import FieldSpec, make_task_spec
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import AdapterParseError
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


def test_two_step_adapter_format_exact_messages_for_simple_signature_with_demo():
    QA = ts("question -> answer", instructions="Given the fields `question`, produce the fields `answer`.")
    adapter = TwoStepAdapter(DummyLM([{"answer": "x"}]))
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=adapter,
        task_spec=QA,
        demos=[{"question": "Q1", "answer": "A1"}],
        inputs={"question": "Q2"},
    )

    expected_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can solve tasks based on user input.\n"
            "As input, you will be provided with:\n"
            "1. `question` (str):\n"
            "Your outputs must contain:\n"
            "1. `answer` (str):\n"
            "You should lay out your outputs in detail so that your answer can be understood by "
            "another agent\n"
            "Specific instructions: Given the fields `question`, produce the fields `answer`.",
        },
        {"role": "user", "content": "question: Q1"},
        {"role": "assistant", "content": "answer: A1"},
        {"role": "user", "content": "question: Q2"},
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_two_step_adapter_format_exact_messages_with_typed_outputs():
    TypedSignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "count": FieldSpec.output("count", type_=int),
            "answer": FieldSpec.output("answer"),
        },
        instructions="Given the fields `question`, produce the fields `count`, `answer`.",
    )
    adapter = TwoStepAdapter(DummyLM([{"count": 1, "answer": "x"}]))
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=adapter, task_spec=TypedSignature, demos=[], inputs={"question": "Q"}
    )

    expected_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can solve tasks based on user input.\n"
            "As input, you will be provided with:\n"
            "1. `question` (str):\n"
            "Your outputs must contain:\n"
            "1. `count` (int): \n"
            "2. `answer` (str):\n"
            "You should lay out your outputs in detail so that your answer can be understood by "
            "another agent\n"
            "Specific instructions: Given the fields `question`, produce the fields `count`, "
            "`answer`.",
        },
        {"role": "user", "content": "question: Q"},
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_two_step_adapter_call():
    TestSignature = make_task_spec(
        {
            "question": FieldSpec.input("question", desc="The math question to solve"),
            "solution": FieldSpec.output("solution", desc="Step by step solution"),
            "answer": FieldSpec.output("answer", type_=float, desc="The final numerical answer"),
        },
        instructions="Given the fields `question`, produce the fields `solution`, `answer`.",
    )
    program = Predict(TestSignature)

    main_lm = RecordingTextLM(["text from main LM"])
    extraction_lm = DummyLM([{"solution": "result", "answer": 12}])

    settings.configure(lm=main_lm, adapter=TwoStepAdapter(extraction_model=extraction_lm))

    result = asyncio.run(program.acall(question="What is 5 + 7?"))

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

    extraction_messages = extraction_lm.history[0].messages_as_openai
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
async def test_two_step_adapter_async_call():
    TestSignature = make_task_spec(
        {
            "question": FieldSpec.input("question", desc="The math question to solve"),
            "solution": FieldSpec.output("solution", desc="Step by step solution"),
            "answer": FieldSpec.output("answer", type_=float, desc="The final numerical answer"),
        },
        instructions="Given the fields `question`, produce the fields `solution`, `answer`.",
    )
    program = Predict(TestSignature)

    main_lm = RecordingTextLM(["text from main LM"])
    extraction_lm = DummyLM([{"solution": "result", "answer": 12}])

    with settings.context(lm=main_lm, adapter=TwoStepAdapter(extraction_model=extraction_lm)):
        result = await program.acall(question="What is 5 + 7?")

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

    extraction_messages = extraction_lm.history[0].messages_as_openai
    assert len(extraction_messages) == 2
    assert extraction_messages[0]["role"] == "system"
    content = extraction_messages[0]["content"]
    assert "`text` (str)" in content
    assert "`solution` (str)" in content
    assert "`answer` (float)" in content
    assert extraction_messages[1]["role"] == "user"
    content = extraction_messages[1]["content"]
    assert "text from main LM" in content


def test_two_step_adapter_parse():
    ComplexSignature = make_task_spec(
        {
            "input_text": FieldSpec.input("input_text"),
            "tags": FieldSpec.output("tags", type_=list[str], desc="List of relevant tags"),
            "confidence": FieldSpec.output("confidence", type_=float, desc="Confidence score"),
        },
        instructions="Given the fields `input_text`, produce the fields `tags`, `confidence`.",
    )
    first_response = "main LM response"

    extraction_lm = DummyLM([{"tags": ["AI", "deep learning", "neural networks"], "confidence": 0.87}])
    adapter = TwoStepAdapter(extraction_lm)
    settings.configure(adapter=adapter, lm=extraction_lm)

    result = adapter.parse(task_spec=ComplexSignature, completion=first_response)

    assert result["tags"] == ["AI", "deep learning", "neural networks"]
    assert result["confidence"] == 0.87


def test_two_step_adapter_parse_errors():
    TestSignature = ts("question -> answer", instructions="Given the fields `question`, produce the fields `answer`.")
    first_response = "main LM response"

    extraction_lm = RecordingTextLM(["not parseable extraction output"])
    adapter = TwoStepAdapter(extraction_lm)

    with pytest.raises(AdapterParseError, match="Failed to parse response"):
        adapter.parse(task_spec=TestSignature, completion=first_response)
