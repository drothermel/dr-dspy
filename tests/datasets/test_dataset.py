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


@pytest.fixture
def csv_file():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv") as tmp_file:
        tmp_file.write(dummy_data)
        tmp_file.flush()
        yield tmp_file.name


@pytest.mark.extra
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
