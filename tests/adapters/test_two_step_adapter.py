import pytest
from typing_extensions import override

from dspy.adapters.two_step_adapter import TwoStepAdapter
from dspy.clients.base_lm import BaseLM
from dspy.clients.openai_format import message_to_openai_chat
from dspy.core.types import LMRequest, LMResponse
from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Predict
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import AdapterParseError
from tests.adapters.conftest import format_messages_and_lm_kwargs


class RecordingTextLM(BaseLM):
    def __init__(self, texts: list[str], *, model: str = "openai/gpt-4o-mini", temperature: float = 1.0):
        super().__init__(model, "chat", temperature, 1000, True)
        self.texts = list(texts)
        self.requests: list[LMRequest] = []

    @override
    def forward(self, request: LMRequest) -> LMResponse:
        self.requests.append(request)
        text = self.texts.pop(0) if self.texts else "No more responses"
        return LMResponse.from_text(text, model=self.model)

    @override
    async def aforward(self, request: LMRequest) -> LMResponse:
        return self.forward(request)


def test_two_step_adapter_format_exact_messages_for_simple_signature_with_demo():
    class QA(Signature):
        question: str = InputField()
        answer: str = OutputField()

    adapter = TwoStepAdapter(DummyLM([{"answer": "x"}]))
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter, QA, [{"question": "Q1", "answer": "A1"}], {"question": "Q2"}
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
    class TypedSignature(Signature):
        question: str = InputField()
        count: int = OutputField()
        answer: str = OutputField()

    adapter = TwoStepAdapter(DummyLM([{"count": 1, "answer": "x"}]))
    messages, lm_kwargs = format_messages_and_lm_kwargs(adapter, TypedSignature, [], {"question": "Q"})

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
    class TestSignature(Signature):
        question: str = InputField(desc="The math question to solve")
        solution: str = OutputField(desc="Step by step solution")
        answer: float = OutputField(desc="The final numerical answer")

    program = Predict(TestSignature)

    main_lm = RecordingTextLM(["text from main LM"])
    extraction_lm = DummyLM([{"solution": "result", "answer": 12}])

    settings.configure(lm=main_lm, adapter=TwoStepAdapter(extraction_model=extraction_lm))

    result = program(question="What is 5 + 7?")

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
    class TestSignature(Signature):
        question: str = InputField(desc="The math question to solve")
        solution: str = OutputField(desc="Step by step solution")
        answer: float = OutputField(desc="The final numerical answer")

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
    class ComplexSignature(Signature):
        input_text: str = InputField()
        tags: list[str] = OutputField(desc="List of relevant tags")
        confidence: float = OutputField(desc="Confidence score")

    first_response = "main LM response"

    extraction_lm = DummyLM([{"tags": ["AI", "deep learning", "neural networks"], "confidence": 0.87}])
    adapter = TwoStepAdapter(extraction_lm)
    settings.configure(adapter=adapter, lm=extraction_lm)

    result = adapter.parse(ComplexSignature, first_response)

    assert result["tags"] == ["AI", "deep learning", "neural networks"]
    assert result["confidence"] == 0.87


def test_two_step_adapter_parse_errors():
    class TestSignature(Signature):
        question: str = InputField()
        answer: str = OutputField()

    first_response = "main LM response"

    extraction_lm = RecordingTextLM(["not parseable extraction output"])
    adapter = TwoStepAdapter(extraction_lm)

    with pytest.raises(AdapterParseError, match="Failed to parse response"):
        adapter.parse(TestSignature, first_response)
