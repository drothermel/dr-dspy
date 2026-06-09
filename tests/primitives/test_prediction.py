import pytest

from dspy.primitives.prediction import Completions


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
