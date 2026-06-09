import pytest

from dspy.primitives import Completions, Example, Prediction


def test_completions_accepts_list_of_dicts():
    completions = Completions([{"answer": "a"}, {"answer": "b"}])
    assert completions["answer"] == ["a", "b"]


def test_completions_accepts_dict_of_lists():
    completions = Completions({"answer": ["a", "b"]})
    assert len(completions) == 2


def test_completions_rejects_non_list_values():
    with pytest.raises(ValueError, match="All Completions values must be lists"):
        Completions({"answer": "not-a-list"})


def test_completions_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="All Completions lists must have the same length"):
        Completions({"answer": ["a", "b"], "reasoning": ["only-one"]})


def test_prediction_from_record():
    prediction = Prediction.from_record({"answer": "Paris"})
    assert prediction.answer == "Paris"


def test_prediction_from_record_rejects_input_keys():
    with pytest.raises(TypeError):
        Prediction.from_record({"answer": "Paris"}, input_keys=("answer",))


def test_prediction_is_not_example():
    prediction = Prediction.from_record({"answer": "Paris"})
    example = Example.from_record({"answer": "Paris"})
    assert not isinstance(prediction, Example)
    assert prediction != example
