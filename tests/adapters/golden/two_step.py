from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.two_step_adapter import TwoStepAdapter
from tests.adapters.golden.types import GoldenPromptCase
from tests.adapters.scenarios.two_step_cases import simple_qa_with_demo_two_step, typed_outputs_two_step
from tests.test_utils import DummyLM


def simple_qa_with_demo_two_step_adapter_builder() -> TwoStepAdapter:
    return TwoStepAdapter(DummyLM([{"answer": "x"}]), extraction_adapter=ChatAdapter())


def typed_outputs_two_step_adapter_builder() -> TwoStepAdapter:
    return TwoStepAdapter(DummyLM([{"count": 1, "answer": "x"}]), extraction_adapter=ChatAdapter())


TWO_STEP_GOLDEN_CASES: tuple[GoldenPromptCase, ...] = (
    GoldenPromptCase(
        id="two_step/simple_qa_with_demo",
        adapter_builder=simple_qa_with_demo_two_step_adapter_builder,
        scenario=simple_qa_with_demo_two_step(),
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can solve tasks based on user input.\nAs input, you will be provided with:\n1. `question` (str): The question.\nYour outputs must contain:\n1. `answer` (str): The answer.\nYou should lay out your outputs in detail so that your answer can be understood by another agent\nSpecific instructions: Given the fields `question`, produce the fields `answer`.",
            },
            {"role": "user", "content": "question: Q1"},
            {"role": "assistant", "content": "answer: A1"},
            {"role": "user", "content": "question: Q2"},
        ],
    ),
    GoldenPromptCase(
        id="two_step/typed_outputs",
        adapter_builder=typed_outputs_two_step_adapter_builder,
        scenario=typed_outputs_two_step(),
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can solve tasks based on user input.\nAs input, you will be provided with:\n1. `question` (str): The question.\nYour outputs must contain:\n1. `count` (int): The count.\n2. `answer` (str): The answer.\nYou should lay out your outputs in detail so that your answer can be understood by another agent\nSpecific instructions: Given the fields `question`, produce the fields `count`, `answer`.",
            },
            {"role": "user", "content": "question: Q"},
        ],
    ),
)
