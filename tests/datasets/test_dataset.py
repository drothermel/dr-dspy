import tempfile
import uuid
from typing import Any

import pytest

from dspy.datasets.dataset import Dataset
from dspy.primitives import Example

dummy_data = 'content,question,answer\n"This is content 1","What is this?","This is answer 1"\n"This is content 2","What is that?","This is answer 2"\n'


class CSVDataset(Dataset):
    def __init__(self, file_path, input_keys: list[str] | None = None, **kwargs: Any) -> None:
        import pandas as pd

        super().__init__(input_keys=input_keys, **kwargs)
        active_input_keys = input_keys or []
        df = pd.read_csv(file_path)
        data = df.to_dict(orient="records")
        self._train = [
            Example.from_record(
                {**record, "dspy_uuid": str(uuid.uuid4()), "dspy_split": "train"},
                input_keys=tuple(active_input_keys),
            )
            for record in data[:1]
        ]
        self._dev = [
            Example.from_record(
                {**record, "dspy_uuid": str(uuid.uuid4()), "dspy_split": "dev"},
                input_keys=tuple(active_input_keys),
            )
            for record in data[1:2]
        ]


class SeedDataset(Dataset):
    def __init__(self, records: list[dict[str, str]], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._train = [dict(record) for record in records]
        self._dev = [dict(record) for record in records]
        self._test = [dict(record) for record in records]


@pytest.fixture
def csv_file():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv") as tmp_file:
        tmp_file.write(dummy_data)
        tmp_file.flush()
        yield tmp_file.name


def test_input_keys(csv_file):
    dataset = CSVDataset(csv_file, input_keys=["content", "question"])
    assert dataset.train is not None
    for example in dataset.train:
        inputs = example.as_inputs()
        assert inputs is not None
        assert "content" in inputs
        assert "question" in inputs
        assert example._input_keys is not None
        assert set(example._input_keys) == {"content", "question"}


def test_reset_seeds_accepts_zero():
    records = [{"id": str(i)} for i in range(5)]
    dataset = SeedDataset(records, train_seed=1, dev_seed=2, test_seed=3)
    _ = dataset.train
    dataset.reset_seeds(train_seed=0, dev_seed=0, test_seed=0)
    assert dataset.train_seed == 0
    assert dataset.dev_seed == 0
    assert dataset.test_seed == 0


def test_train_size_zero_yields_empty_train():
    records = [{"id": str(i)} for i in range(5)]
    dataset = SeedDataset(records, train_size=0)
    assert dataset.train == []


def test_dev_and_test_seeds_shuffle_independently():
    records = [{"id": str(i)} for i in range(10)]
    dataset_a = SeedDataset(records, dev_seed=1, test_seed=1)
    dataset_b = SeedDataset(records, dev_seed=1, test_seed=2)
    dev_ids_a = [example["id"] for example in dataset_a.dev]
    test_ids_b = [example["id"] for example in dataset_b.test]
    dev_ids_b = [example["id"] for example in dataset_b.dev]
    assert dev_ids_a == dev_ids_b
    assert test_ids_b != dev_ids_a
