from unittest.mock import patch

import pytest

from dspy.datasets.dataset import Dataset
from dspy.integrations.datasets.alfworld.alfworld import AlfWorld
from dspy.integrations.datasets.gsm8k import GSM8K
from dspy.integrations.datasets.hotpotqa import HotPotQA
from dspy.integrations.datasets.math import MATH
from dspy.integrations.datasets.metrics import gsm8k_metric, hotpotqa_metric, math_metric
from dspy.primitives import Example, Prediction


def test_gsm8k_metric_signature():
    example = Example.from_record({"question": "q", "answer": "42"}, input_keys=("question",))
    pred = Prediction(answer="The answer is 42")
    assert gsm8k_metric(example, pred, None) is True


def test_alfworld_dataset_contract():
    with patch("dspy.integrations.datasets.alfworld.alfworld.EnvPool"):
        dataset = AlfWorld(max_threads=1, train_size=4, dev_size=2)
    assert isinstance(dataset, Dataset)
    assert dataset.default_input_keys == ("idx",)
    assert len(dataset.train) == 4
    assert dataset.train[0].idx is not None
    assert not hasattr(dataset, "default_metric")


@pytest.mark.integration
def test_gsm8k_dataset_contract():
    dataset = GSM8K(train_size=4, dev_size=2, test_size=2)
    assert isinstance(dataset, Dataset)
    assert GSM8K.default_metric is gsm8k_metric
    assert dataset.default_input_keys == ("question",)
    train = dataset.train
    assert len(train) == 4
    assert all(isinstance(example, Example) for example in train)
    assert all(example.dspy_split == "train" for example in train)
    assert "dspy_uuid" in train[0]


@pytest.mark.integration
def test_math_dataset_contract():
    dataset = MATH("algebra", train_size=2, dev_size=2, test_size=2)
    assert isinstance(dataset, Dataset)
    assert MATH.default_metric is math_metric
    assert len(dataset.train) == 2
    assert dataset.train[0].dspy_split == "train"


@pytest.mark.integration
def test_hotpotqa_dataset_contract():
    dataset = HotPotQA(train_size=2, dev_size=2, test_size=2)
    assert isinstance(dataset, Dataset)
    assert HotPotQA.default_metric is hotpotqa_metric
    assert len(dataset.test) == 2
