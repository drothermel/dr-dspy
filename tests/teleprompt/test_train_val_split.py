import pytest

from dspy.predict.predict import Predict  # noqa: F401 — initialize predict before primitives lazy import
from dspy.primitives import Example
from dspy.teleprompt.core.split import split_trainset_holdout


def _examples(count: int) -> list[Example]:
    return [
        Example.from_record({"input": f"example-{index}", "output": str(index)}, input_keys=("input",))
        for index in range(count)
    ]


def test_split_trainset_holdout_same_seed_is_reproducible():
    trainset = _examples(10)
    train_a, val_a = split_trainset_holdout(trainset, holdout_ratio=0.2, seed=7)
    train_b, val_b = split_trainset_holdout(trainset, holdout_ratio=0.2, seed=7)
    assert [example.input for example in train_a] == [example.input for example in train_b]
    assert [example.input for example in val_a] == [example.input for example in val_b]


def test_split_trainset_holdout_different_seed_changes_split():
    trainset = _examples(10)
    _train_a, val_a = split_trainset_holdout(trainset, holdout_ratio=0.2, seed=1)
    _train_b, val_b = split_trainset_holdout(trainset, holdout_ratio=0.2, seed=2)
    assert [example.input for example in val_a] != [example.input for example in val_b]


def test_split_trainset_holdout_sizes():
    trainset = _examples(10)
    train, val = split_trainset_holdout(trainset, holdout_ratio=0.2, seed=0)
    assert len(val) == 2
    assert len(train) == 8
    assert len(train) + len(val) == len(trainset)


def test_split_trainset_holdout_rejects_invalid_ratio():
    trainset = _examples(5)
    with pytest.raises(ValueError, match="holdout_ratio"):
        split_trainset_holdout(trainset, holdout_ratio=0.0, seed=0)
