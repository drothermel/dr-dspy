import random
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from dspy.datasets.dataset import Dataset
from dspy.primitives.example import Example
from dspy.runtime.run_context import RunContext

if TYPE_CHECKING:
    import pandas as pd


def _rows_to_examples(
    rows: Iterable[Mapping[str, object]], fields: Sequence[str] | None, input_keys: tuple[str, ...]
) -> list[Example]:
    rows_list = list(rows)
    if not rows_list:
        return []
    resolved_fields = list(fields) if fields is not None else list(rows_list[0])
    return [
        Example.from_record({field: row[field] for field in resolved_fields}, input_keys=input_keys)
        for row in rows_list
    ]


class DataLoader(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def from_huggingface(
        self,
        dataset_name: str,
        *args: Any,
        input_keys: tuple[str, ...] = (),
        fields: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> Mapping[str, list[Example]] | list[Example]:
        if fields and (not isinstance(fields, tuple)):
            raise ValueError("Invalid fields provided. Please provide a tuple of fields.")
        if not isinstance(input_keys, tuple):
            raise TypeError("Invalid input keys provided. Please provide a tuple of input keys.")
        from datasets import DatasetDict, load_dataset

        dataset = load_dataset(dataset_name, *args, **kwargs)
        if isinstance(dataset, list) and isinstance(kwargs.get("split"), list):
            split_names = cast("list[str]", kwargs["split"])
            return {
                split_name: _rows_to_examples(
                    rows=cast("Iterable[Mapping[str, object]]", split_rows), fields=fields, input_keys=input_keys
                )
                for split_name, split_rows in zip(split_names, dataset, strict=False)
            }
        if isinstance(dataset, DatasetDict):
            return {
                split_name: _rows_to_examples(rows=rows, fields=fields, input_keys=input_keys)
                for split_name, rows in dataset.items()
            }
        return _rows_to_examples(
            rows=cast("Iterable[Mapping[str, object]]", dataset), fields=fields, input_keys=input_keys
        )

    def from_csv(
        self, file_path: str, fields: list[str] | None = None, input_keys: tuple[str, ...] = ()
    ) -> list[Example]:
        from datasets import load_dataset

        loaded_dataset: Any = load_dataset("csv", data_files=file_path)
        dataset = loaded_dataset["train"]
        return _rows_to_examples(
            rows=cast("Iterable[Mapping[str, object]]", dataset), fields=fields, input_keys=input_keys
        )

    def from_pandas(
        self, df: "pd.DataFrame", fields: list[str] | None = None, input_keys: tuple[str, ...] = ()
    ) -> list[Example]:
        if fields is None:
            fields = list(df.columns)
        return [
            Example.from_record({field: row[field] for field in fields}, input_keys=input_keys)
            for _, row in df.iterrows()
        ]

    def from_json(
        self, file_path: str, fields: list[str] | None = None, input_keys: tuple[str, ...] = ()
    ) -> list[Example]:
        from datasets import load_dataset

        loaded_dataset: Any = load_dataset("json", data_files=file_path)
        dataset = loaded_dataset["train"]
        return _rows_to_examples(
            rows=cast("Iterable[Mapping[str, object]]", dataset), fields=fields, input_keys=input_keys
        )

    def from_parquet(
        self, file_path: str, fields: list[str] | None = None, input_keys: tuple[str, ...] = ()
    ) -> list[Example]:
        from datasets import load_dataset

        loaded_dataset: Any = load_dataset("parquet", data_files=file_path)
        dataset = loaded_dataset["train"]
        return _rows_to_examples(
            rows=cast("Iterable[Mapping[str, object]]", dataset), fields=fields, input_keys=input_keys
        )

    def from_rm(self, run: RunContext, num_samples: int, fields: list[str], input_keys: list[str]) -> list[Example]:
        rm = run.retrieval
        if rm is None:
            raise ValueError("Retrieval module not found. Pass retrieval=... when creating RunContext.")
        try:
            return _rows_to_examples(
                rows=cast("Iterable[Mapping[str, object]]", rm.get_objects(num_samples=num_samples, fields=fields)),
                fields=fields,
                input_keys=tuple(input_keys),
            )
        except AttributeError:
            raise ValueError(
                "Retrieval module does not support `get_objects`. Please use a different retrieval module."
            )

    def sample(self, dataset: list[Example], n: int) -> list[Example]:
        if not isinstance(dataset, list):
            raise TypeError(
                f"Invalid dataset provided of type {type(dataset)}. Please provide a list of `dspy.primitives.example.Example`s."
            )
        return random.sample(dataset, n)

    def train_test_split(
        self,
        dataset: list[Example],
        train_size: int | float = 0.75,
        test_size: int | float | None = None,
        random_state: int | None = None,
    ) -> Mapping[str, list[Example]]:
        if random_state is not None:
            random.seed(random_state)
        dataset_shuffled = list(dataset)
        random.shuffle(dataset_shuffled)
        if train_size is not None and isinstance(train_size, float) and (0 < train_size < 1):
            train_end = int(len(dataset_shuffled) * train_size)
        elif train_size is not None and isinstance(train_size, int):
            train_end = train_size
        else:
            raise ValueError(
                f"Invalid `train_size`. Please provide a float between 0 and 1 to represent the proportion of the dataset to include in the train split or an int to represent the absolute number of samples to include in the train split. Received `train_size`: {train_size}."
            )
        if test_size is not None:
            if isinstance(test_size, float) and 0 < test_size < 1:
                test_end = int(len(dataset_shuffled) * test_size)
            elif isinstance(test_size, int):
                test_end = test_size
            else:
                raise ValueError(
                    f"Invalid `test_size`. Please provide a float between 0 and 1 to represent the proportion of the dataset to include in the test split or an int to represent the absolute number of samples to include in the test split. Received `test_size`: {test_size}."
                )
            if train_end + test_end > len(dataset_shuffled):
                raise ValueError(
                    f"`train_size` + `test_size` cannot exceed the total number of samples. Received `train_size`: {train_end}, `test_size`: {test_end}, and `dataset_size`: {len(dataset_shuffled)}."
                )
        else:
            test_end = len(dataset_shuffled) - train_end
        train_dataset = dataset_shuffled[:train_end]
        test_dataset = dataset_shuffled[train_end : train_end + test_end]
        return {"train": train_dataset, "test": test_dataset}
