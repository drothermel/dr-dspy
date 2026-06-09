from dspy.task_spec import input_field, make_task_spec, output_field
from tests.adapters.scenarios.case import FormatScenarioCase
from tests.task_spec.helpers import ts

SIMPLE_QA_CONTRACT_SIGNATURE = ts("question -> answer", instructions="Given the fields, produce the outputs.")
SIMPLE_QA_CONTRACT_INPUTS = {"question": "What is the capital of France?"}


def simple_qa_chat() -> FormatScenarioCase:
    return FormatScenarioCase(
        task_spec=ts("question -> answer", instructions="Answer the question."),
        demos=(),
        inputs={"question": "What is the capital of France?"},
    )


def simple_qa_json() -> FormatScenarioCase:
    return FormatScenarioCase(
        task_spec=ts("question -> answer", instructions="Given the fields `question`, produce the fields `answer`."),
        demos=(),
        inputs={"question": "What is the capital of France?"},
    )


def simple_qa_xml() -> FormatScenarioCase:
    return FormatScenarioCase(
        task_spec=ts("question -> answer", instructions="Given the fields `question`, produce the fields `answer`."),
        demos=(),
        inputs={"question": "why did a chicken cross the kitchen?"},
    )


def incomplete_demo() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "context": input_field("context", desc="The context."),
            "answer": output_field("answer", desc="The answer."),
            "confidence": output_field("confidence", type_=float, desc="The confidence."),
        },
        instructions="Given the fields `question`, `context`, produce the fields `answer`, `confidence`.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=({"question": "Q1", "answer": "A1"},),
        inputs={"question": "Q2", "context": "C2"},
    )
