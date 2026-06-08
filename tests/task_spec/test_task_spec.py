import pytest

from dspy.task_spec import FieldSpec, TaskSpec, default_task_instructions, infer_prefix, make_task_spec
from tests.task_spec.helpers import ts


def test_make_task_spec_from_string():
    spec = make_task_spec("question, context -> answer", instructions="Answer the question.")
    assert spec.instructions == "Answer the question."
    assert list(spec.input_fields) == ["question", "context"]
    assert list(spec.output_fields) == ["answer"]
    assert spec.input_fields["question"].type_ is str


def test_make_task_spec_requires_instructions():
    with pytest.raises(ValueError, match="instructions is required"):
        make_task_spec("q -> a", instructions="")


def test_make_task_spec_from_field_dict():
    spec = make_task_spec(
        {
            "question": FieldSpec.input("question", desc="The question"),
            "answer": FieldSpec.output("answer", desc="The answer"),
        },
        instructions="Answer briefly.",
        name="QA",
    )
    assert spec.name == "QA"
    assert spec.input_fields["question"].desc == "The question"
    assert spec.output_fields["answer"].role == "output"


def test_duplicate_input_output_names_raise():
    with pytest.raises(ValueError, match="distinct names"):
        make_task_spec("value -> value", instructions="Do it.")


def test_default_task_instructions():
    text = default_task_instructions(inputs=("a", "b"), outputs=("c",))
    assert text == "Given the fields `a`, `b`, produce the fields `c`."


def test_with_instructions_is_immutable():
    original = ts("q -> a", instructions="First")
    updated = original.with_instructions("Second")
    assert original.instructions == "First"
    assert updated.instructions == "Second"
    assert original is not updated


def test_with_updated_field():
    original = ts("input1, input2 -> output", instructions="Test")
    updated = original.with_updated_field("input1", prefix="Modified:")
    assert updated.input_fields["input1"].prefix == "Modified:"
    assert original.input_fields["input1"].prefix == "Input 1:"


def test_append_prepend_delete():
    base = ts("q -> a", instructions="Test")
    with_reasoning = base.append(FieldSpec.output("reasoning", desc="Chain of thought"))
    assert list(with_reasoning.output_fields) == ["a", "reasoning"]

    with_context = base.prepend(FieldSpec.input("context", desc="Background"))
    assert list(with_context.input_fields) == ["context", "q"]

    trimmed = with_context.delete("context")
    assert list(trimmed.input_fields) == ["q"]


def test_equals_and_fingerprint():
    spec1 = ts("q -> a", instructions="Same")
    spec2 = ts("q -> a", instructions="Same")
    spec3 = ts("q -> a", instructions="Different")
    assert spec1.equals(spec2)
    assert not spec1.equals(spec3)
    assert spec1.fingerprint() == spec2.fingerprint()
    assert spec1.fingerprint() != spec3.fingerprint()


def test_serialize_round_trip():
    original = make_task_spec(
        "question: int, context: list[str] -> answer",
        instructions="Answer using context.",
        name="RAG",
    )
    restored = TaskSpec.from_dict(original.to_dict())
    assert original.equals(restored)
    assert restored.name == "RAG"
    assert restored.input_fields["question"].type_ is int


def test_to_declaration():
    spec = ts("q -> a", instructions="Answer.")
    declaration = spec.to_declaration()
    assert "TaskSpec" in declaration
    assert "Answer." in declaration
    assert "q -> a" in declaration


def test_infer_prefix():
    assert infer_prefix("camelCaseText") == "Camel Case Text"
    assert infer_prefix("snake_case_text") == "Snake Case Text"


def test_field_spec_default_desc():
    field = FieldSpec.input("my_field")
    assert field.desc == "${my_field}"
    assert field.prefix == "My Field:"


def test_custom_types_in_string_spec():
    class CustomType:
        pass

    spec = make_task_spec(
        "input: CustomType -> output",
        instructions="Use custom type.",
        custom_types={"CustomType": CustomType},
    )
    assert spec.input_fields["input"].type_ is CustomType
