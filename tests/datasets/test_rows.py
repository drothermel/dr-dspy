from dspy.datasets.rows import rows_to_examples


def test_rows_to_examples_empty() -> None:
    assert rows_to_examples([], fields=None, input_keys=("question",)) == []


def test_rows_to_examples_builds_examples() -> None:
    rows = [{"question": "What is DSPy?", "answer": "A framework."}]
    examples = rows_to_examples(rows, fields=None, input_keys=("question",))

    assert len(examples) == 1
    assert examples[0].question == "What is DSPy?"
    assert examples[0].answer == "A framework."
