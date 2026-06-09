from typing import Any, cast

from dspy.task_spec import make_task_spec
from dspy.task_spec.pydantic_bridge import task_spec_to_pydantic_model
from tests.task_spec.helpers import ts


def test_task_spec_to_pydantic_model_fields():
    spec = ts("question: int, context: list[str] -> answer", instructions="Answer.")
    model = task_spec_to_pydantic_model(spec)
    assert "question" in model.model_fields
    assert "answer" in model.model_fields
    assert model.model_fields["question"].annotation is int
    assert model.model_fields["question"].json_schema_extra["__dspy_field_type"] == "input"
    assert model.model_fields["answer"].json_schema_extra["__dspy_field_type"] == "output"


def test_task_spec_to_pydantic_model_instance():
    spec = make_task_spec("q -> a", instructions="Test")
    model = task_spec_to_pydantic_model(spec)
    instance = cast("Any", model(q="hello", a="world"))
    assert instance.q == "hello"
    assert instance.a == "world"
