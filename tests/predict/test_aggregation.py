from typing import Any

import pytest

from dspy.evaluate.metrics import normalize_text
from dspy.predict.aggregation import majority
from dspy.primitives import Completions, Prediction


def test_majority_with_prediction():
    prediction = Prediction.from_completions([{"answer": "2"}, {"answer": "2"}, {"answer": "3"}])
    result = majority(prediction)
    assert result["answer"] == "2"


def test_completions_len_empty():
    assert len(Completions({})) == 0


def test_majority_with_completions():
    completions = Completions([{"answer": "2"}, {"answer": "2"}, {"answer": "3"}])
    result = majority(completions)
    assert result["answer"] == "2"


def test_majority_with_list():
    completions = [{"answer": "2"}, {"answer": "2"}, {"answer": "3"}]
    result = majority(completions)
    assert result["answer"] == "2"


def test_majority_with_normalize():
    completions = [{"answer": "2"}, {"answer": " 2"}, {"answer": "3"}]
    result = majority(completions, normalize=normalize_text)
    assert result["answer"] == "2"


def test_majority_with_field():
    completions = [{"answer": "2", "other": "1"}, {"answer": "2", "other": "1"}, {"answer": "3", "other": "2"}]
    result = majority(completions, field="other")
    assert result["other"] == "1"


def test_majority_with_no_majority():
    completions = [{"answer": "2"}, {"answer": "3"}, {"answer": "4"}]
    result = majority(completions)
    assert result["answer"] == "2"


def test_majority_invalid_type_raises():
    invalid_input: Any = 42
    with pytest.raises(TypeError, match="majority expected one of"):
        majority(invalid_input)
