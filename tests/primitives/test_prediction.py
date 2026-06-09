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
        Completions({"answer": "not-a-list"})  # ty: ignore[invalid-argument-type]


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


def test_prediction_is_unhashable():
    prediction = Prediction.from_record({"answer": "Paris"})
    with pytest.raises(TypeError):
        hash(prediction)


def test_prediction_eq():
    prediction1 = Prediction.from_record({"answer": "Paris"})
    prediction2 = Prediction.from_record({"answer": "Paris"})
    prediction3 = Prediction.from_record({"answer": "London"})
    assert prediction1 == prediction2
    assert prediction1 != prediction3


def test_prediction_values():
    prediction = Prediction.from_record({"answer": "Paris", "dspy_hidden": 1})
    assert prediction.values() == ["Paris"]
    assert prediction.values(include_dspy=True) == ["Paris", 1]


def test_prediction_score_numeric_ops():
    low = Prediction(score=0.3)
    high = Prediction(score=0.9)
    assert float(low) == 0.3
    assert low + 0.1 == pytest.approx(0.4)
    assert 1.0 + high == pytest.approx(1.9)
    assert high / 2 == pytest.approx(0.45)
    assert 1.0 / high == pytest.approx(1.0 / 0.9)
    assert low < high
    assert high > low
    assert low <= 0.3
    assert high >= 0.9


def test_prediction_score_comparison_requires_score_field():
    prediction = Prediction.from_record({"answer": "Paris"})
    with pytest.raises(ValueError, match="does not have a 'score' field"):
        float(prediction)
    with pytest.raises(ValueError, match="does not have a 'score' field"):
        _ = prediction < 0.5


def test_prediction_score_comparison_rejects_unsupported_types():
    prediction = Prediction(score=0.5)
    with pytest.raises(TypeError, match="Unsupported type for comparison"):
        _ = prediction.__lt__("0.5")  # ty: ignore[invalid-argument-type]


def test_prediction_lm_usage_round_trip():
    usage = {"openai/gpt-4o-mini": {"total_tokens": 10}}
    prediction = Prediction(score=1.0)
    prediction.set_lm_usage(usage)
    assert prediction.get_lm_usage() == usage
