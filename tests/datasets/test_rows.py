import pytest

from dspy.datasets.rows import rows_to_examples


def test_rows_to_examples_empty() -> None:
    assert rows_to_examples([], fields=None, input_keys=("question",)) == []


def test_rows_to_examples_builds_examples() -> None:
    rows = [{"question": "What is DSPy?", "answer": "A framework."}]
    examples = rows_to_examples(rows, fields=None, input_keys=("question",))

    assert len(examples) == 1
    assert examples[0].question == "What is DSPy?"
    assert examples[0].answer == "A framework."


def test_rows_to_examples_preserves_explicit_field_order() -> None:
    rows = [{"answer": "A framework.", "question": "What is DSPy?"}]
    examples = rows_to_examples(rows, fields=("question", "answer"), input_keys=("question",))

    assert list(examples[0].to_dict().keys())[:2] == ["question", "answer"]


def test_rows_to_examples_empty_fields() -> None:
    rows = [{"question": "What is DSPy?", "answer": "A framework."}]
    examples = rows_to_examples(rows, fields=[], input_keys=())

    assert examples[0].to_dict() == {}


def test_rows_to_examples_missing_field_raises() -> None:
    rows = [{"question": "What is DSPy?"}]
    with pytest.raises(KeyError):
        rows_to_examples(rows, fields=("question", "answer"), input_keys=("question",))


def test_rows_to_examples_inconsistent_schema_raises() -> None:
    rows = [{"question": "q1", "answer": "a1"}, {"question": "q2"}]
    with pytest.raises(KeyError):
        rows_to_examples(rows, fields=None, input_keys=("question",))
