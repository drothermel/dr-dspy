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


def two_input_judgement_xml() -> FormatScenarioCase:
    return FormatScenarioCase(
        task_spec=ts(
            "question, answer -> judgement",
            instructions="Given the fields `question`, `answer`, produce the fields `judgement`.",
        ),
        demos=(),
        inputs={"question": "why did a chicken cross the kitchen?", "answer": "To get to the other side!"},
    )


def demo_typed_outputs_chat() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "answers": output_field("answers", type_=list[str], desc="The answers."),
            "scores": output_field("scores", type_=list[float], desc="The scores."),
        },
        instructions="Answer the question with multiple answers and scores",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=({"question": "Q1", "answers": ["A1", "A2"], "scores": [0.1, 0.9]},),
        inputs={"question": "Q2"},
    )


def demo_typed_outputs_json() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", desc="The answer."),
            "confidence": output_field("confidence", type_=float, desc="The confidence."),
        },
        instructions="Given the fields `question`, produce the fields `answer`, `confidence`.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=({"question": "Q1", "answer": "A1", "confidence": 0.9},),
        inputs={"question": "Q2"},
    )


def demo_typed_outputs_xml() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", desc="The answer."),
            "score": output_field("score", type_=float, desc="The score."),
        },
        instructions="Given the fields `question`, produce the fields `answer`, `score`.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=({"question": "Q1", "answer": "A1", "score": 0.9},),
        inputs={"question": "Q2"},
    )


def incomplete_demo_chat() -> FormatScenarioCase:
    return _incomplete_demo_chat()


def incomplete_demo_json() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "context": input_field("context", desc="The context."),
            "answer": output_field("answer", desc="The answer."),
            "score": output_field("score", type_=float, desc="The score."),
        },
        instructions="Given the fields `question`, `context`, produce the fields `answer`, `score`.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=({"question": "Q1", "answer": "A1"},),
        inputs={"question": "Q2", "context": "C2"},
    )


def incomplete_demo_xml() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "context": input_field("context", desc="The context."),
            "answer": output_field("answer", desc="The answer."),
            "score": output_field("score", type_=float, desc="The score."),
        },
        instructions="Given the fields `question`, `context`, produce the fields `answer`, `score`.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=({"question": "Q1", "answer": "A1"},),
        inputs={"question": "Q2", "context": "C2"},
    )


def _incomplete_demo_chat() -> FormatScenarioCase:
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
