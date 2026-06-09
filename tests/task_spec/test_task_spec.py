import pytest

from dspy.task_spec import (
    FieldRole,
    TaskSpec,
    default_task_instructions,
    infer_prefix,
    input_field,
    make_task_spec,
    output_field,
)
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
            "question": input_field("question", desc="The question"),
            "answer": output_field("answer", desc="The answer"),
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


def test_duplicate_input_names_via_field_lists_raise():
    with pytest.raises(ValueError, match="Duplicate input field name"):
        make_task_spec(
            inputs=[
                input_field("q", desc="First question."),
                input_field("q", desc="Duplicate question."),
            ],
            outputs=[output_field("a", desc="The answer.")],
            instructions="Answer.",
        )


def test_duplicate_input_names_via_direct_task_spec_raise():
    field = input_field("q", desc="The question.")
    with pytest.raises(ValueError, match="Duplicate input field name"):
        TaskSpec(name="QA", instructions="Answer.", inputs=(field, field), outputs=())


def test_duplicate_output_names_via_field_lists_raise():
    with pytest.raises(ValueError, match="Duplicate output field name"):
        make_task_spec(
            inputs=[input_field("q", desc="The question.")],
            outputs=[
                output_field("a", desc="First answer."),
                output_field("a", desc="Duplicate answer."),
            ],
            instructions="Answer.",
        )


def test_duplicate_output_names_via_direct_task_spec_raise():
    field = output_field("a", desc="The answer.")
    with pytest.raises(ValueError, match="Duplicate output field name"):
        TaskSpec(name="QA", instructions="Answer.", inputs=(), outputs=(field, field))


def test_task_spec_rejects_empty_fields_on_direct_construct():
    with pytest.raises(ValueError, match="at least one input or output field"):
        TaskSpec(name="Empty", instructions="Do nothing.", inputs=(), outputs=())


def test_make_task_spec_rejects_empty_fields():
    with pytest.raises(ValueError, match="at least one"):
        make_task_spec(inputs=[], outputs=[], instructions="Test.")


def test_from_dict_rejects_empty_fields():
    payload = make_task_spec("q -> a", instructions="Answer.").to_dict()
    payload["inputs"] = []
    payload["outputs"] = []
    with pytest.raises(ValueError, match="at least one input or output field"):
        TaskSpec.from_dict(payload)


def test_from_dict_rejects_duplicate_input_names():
    payload = make_task_spec("q -> a", instructions="Answer.").to_dict()
    payload["inputs"].append(payload["inputs"][0])
    with pytest.raises(ValueError, match="Duplicate input field name"):
        TaskSpec.from_dict(payload)


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
    with_reasoning = base.append(output_field("reasoning", desc="Chain of thought"))
    assert list(with_reasoning.output_fields) == ["a", "reasoning"]
    with_context = base.prepend(input_field("context", desc="Background"))
    assert list(with_context.input_fields) == ["context", "q"]
    trimmed = with_context.delete("context")
    assert list(trimmed.input_fields) == ["q"]


def test_equality_and_fingerprint():
    spec1 = ts("q -> a", instructions="Same")
    spec2 = ts("q -> a", instructions="Same")
    spec3 = ts("q -> a", instructions="Different")
    assert spec1 == spec2
    assert spec1 != spec3
    assert spec1.fingerprint() == spec2.fingerprint()
    assert spec1.fingerprint() != spec3.fingerprint()


def test_name_is_part_of_identity():
    spec_a = make_task_spec(
        inputs=[input_field("q", desc="The question")],
        outputs=[output_field("a", desc="The answer")],
        instructions="Same",
        name="SpecA",
    )
    spec_b = make_task_spec(
        inputs=[input_field("q", desc="The question")],
        outputs=[output_field("a", desc="The answer")],
        instructions="Same",
        name="SpecB",
    )
    assert spec_a != spec_b
    assert spec_a.fingerprint() != spec_b.fingerprint()


def test_serialize_round_trip():
    original = make_task_spec(
        "question: int, context: list[str] -> answer", instructions="Answer using context.", name="RAG"
    )
    restored = TaskSpec.from_dict(original.to_dict())
    assert original == restored
    assert restored.name == "RAG"
    assert restored.input_fields["question"].type_ is int


def test_to_debug_string():
    spec = ts("q -> a", instructions="Answer.")
    declaration = spec.to_debug_string()
    assert "TaskSpec" in declaration
    assert "Answer." in declaration
    assert "q -> a" in declaration


def test_infer_prefix():
    assert infer_prefix("camelCaseText") == "Camel Case Text"
    assert infer_prefix("snake_case_text") == "Snake Case Text"


def test_field_spec_requires_explicit_desc():
    field = input_field("my_field", desc="Description of my field.")
    assert field.desc == "Description of my field."
    assert field.prefix == "My Field:"


def test_input_field_and_output_field():
    inp = input_field("question", desc="The question")
    out = output_field("answer", type_=int, desc="The answer")
    assert inp.role == FieldRole.INPUT
    assert out.role == FieldRole.OUTPUT
    assert inp.type_ is str
    assert out.type_ is int


def test_make_task_spec_from_field_lists():
    spec = make_task_spec(
        inputs=[input_field("question", desc="The question")],
        outputs=[output_field("answer", desc="The answer")],
        instructions="Answer briefly.",
        name="QA",
    )
    assert spec.name == "QA"
    assert list(spec.input_fields) == ["question"]
    assert list(spec.output_fields) == ["answer"]


def test_make_task_spec_rejects_mismatched_field_role():
    with pytest.raises(ValueError, match=r"expected.*output"):
        make_task_spec(outputs=[input_field("answer", desc="The answer.")], instructions="Test.")


def test_make_task_spec_rejects_spec_and_lists():
    with pytest.raises(TypeError, match="not both"):
        make_task_spec("q -> a", inputs=[input_field("q", desc="The q.")], instructions="Test.")


def test_custom_types_in_string_spec():

    class CustomType:
        pass

    spec = make_task_spec(
        "input: CustomType -> output", instructions="Use custom type.", custom_types={"CustomType": CustomType}
    )
    assert spec.input_fields["input"].type_ is CustomType


def test_optional_syntax_parses_equivalent_to_pep604_union():
    optional_spec = make_task_spec("q: Optional[str] -> a", instructions="Answer.")
    pep604_spec = make_task_spec("q: str | None -> a", instructions="Answer.")
    assert optional_spec.input_fields["q"].type_ == pep604_spec.input_fields["q"].type_
