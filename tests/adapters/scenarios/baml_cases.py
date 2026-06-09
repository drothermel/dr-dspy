import pydantic

from dspy.task_spec import input_field, make_task_spec, output_field
from tests.adapters.scenarios.case import FormatScenarioCase
from tests.task_spec.helpers import ts


def simple_qa_with_demo_baml() -> FormatScenarioCase:
    return FormatScenarioCase(
        task_spec=ts("question -> answer", instructions="Given the fields `question`, produce the fields `answer`."),
        demos=({"question": "Q1", "answer": "A1"},),
        inputs={"question": "Q2"},
    )


def nested_output_baml() -> FormatScenarioCase:
    class BamlNested(pydantic.BaseModel):
        value: int
        tags: list[str]

    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", type_=BamlNested, desc="The answer."),
        },
        instructions="Given the fields `question`, produce the fields `answer`.",
    )
    return FormatScenarioCase(task_spec=task_spec, demos=(), inputs={"question": "Q"})
