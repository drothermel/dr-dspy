from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature, make_signature
from dspy.task_spec import make_task_spec
from dspy.task_spec.bridge import task_spec_from_signature


def test_task_spec_from_signature_string_signature():
    signature = make_signature("question, context -> answer", instructions="Answer the question.")
    expected = make_task_spec(
        "question, context -> answer", instructions="Answer the question.", name="StringSignature"
    )
    actual = task_spec_from_signature(signature)
    assert actual.equals(expected.model_copy(update={"name": signature.__name__}))


def test_task_spec_from_signature_subclass():
    class QA(Signature):
        """Answer briefly."""

        question: str = InputField(desc="The question")
        answer: str = OutputField(desc="The answer")

    actual = task_spec_from_signature(QA)
    assert actual.name == "QA"
    assert actual.instructions == "Answer briefly."
    assert actual.input_fields["question"].desc == "The question"
    assert actual.output_fields["answer"].desc == "The answer"
