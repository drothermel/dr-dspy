from dspy.task_spec import input_field, make_task_spec, output_field
from tests.adapters.scenarios.case import FormatScenarioCase
from tests.adapters.scenarios.pydantic_models import (
    Address,
    JsonNestedSummary,
    Person,
    Summary,
    XmlSummary,
)


def nested_pydantic_chat() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "person": input_field("person", type_=Person, desc="The person."),
            "summary": output_field("summary", type_=Summary, desc="The summary."),
        },
        instructions="Given the fields `person`, produce the fields `summary`.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=(),
        inputs={"person": Person(name="Ada", address=Address(city="London", country="UK"), tags=["math", "code"])},
    )


def nested_pydantic_json() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "summary": output_field("summary", type_=JsonNestedSummary, desc="The summary."),
        },
        instructions="Given the fields `question`, produce the fields `summary`.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=(),
        inputs={"question": "Summarize"},
    )


def nested_pydantic_xml() -> FormatScenarioCase:
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "summary": output_field("summary", type_=XmlSummary, desc="The summary."),
        },
        instructions="Given the fields `question`, produce the fields `summary`.",
    )
    return FormatScenarioCase(
        task_spec=task_spec,
        demos=(),
        inputs={"question": "Summarize"},
    )
