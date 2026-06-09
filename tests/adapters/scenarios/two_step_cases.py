from dspy.task_spec import input_field, make_task_spec, output_field
from tests.adapters.scenarios.case import FormatScenarioCase
from tests.task_spec.helpers import ts


def simple_qa_with_demo_two_step() -> FormatScenarioCase:
    return FormatScenarioCase(
        task_spec=ts("question -> answer", instructions="Given the fields `question`, produce the fields `answer`."),
        demos=({"question": "Q1", "answer": "A1"},),
        inputs={"question": "Q2"},
    )


def typed_outputs_two_step() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "count": output_field("count", type_=int, desc="The count."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `question`, produce the fields `count`, `answer`.",
    )
    return FormatScenarioCase(task_spec=task_spec, demos=(), inputs={"question": "Q"})
