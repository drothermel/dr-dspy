from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from dspy._internal.lazy_import import _detect_dspy_dist
from dspy.datasets.dataset import Dataset
from dspy.datasets.rows import rows_to_examples
from dspy.primitives.example import Example

try:
    from datasets import DatasetDict, load_dataset
except ImportError as err:
    raise ImportError(
        f"The 'datasets' extra is required for Hugging Face dataset loading. "
        f"Install it with `pip install {_detect_dspy_dist()}[datasets]`."
    ) from err


class HuggingFaceDataLoader(Dataset):
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

        dataset = load_dataset(dataset_name, *args, **kwargs)
        if isinstance(dataset, list) and isinstance(kwargs.get("split"), list):
            split_names = cast("list[str]", kwargs["split"])
            return {
                split_name: rows_to_examples(
                    rows=cast("Iterable[Mapping[str, object]]", split_rows), fields=fields, input_keys=input_keys
                )
                for split_name, split_rows in zip(split_names, dataset, strict=False)
            }
        if isinstance(dataset, DatasetDict):
            return {
                split_name: rows_to_examples(rows=rows, fields=fields, input_keys=input_keys)
                for split_name, rows in dataset.items()
            }
        return rows_to_examples(
            rows=cast("Iterable[Mapping[str, object]]", dataset), fields=fields, input_keys=input_keys
        )

    def from_csv(
        self, file_path: str, fields: Sequence[str] | None = None, input_keys: tuple[str, ...] = ()
    ) -> list[Example]:
        loaded_dataset: Any = load_dataset("csv", data_files=file_path)
        dataset = loaded_dataset["train"]
        return rows_to_examples(
            rows=cast("Iterable[Mapping[str, object]]", dataset), fields=fields, input_keys=input_keys
        )

    def from_json(
        self, file_path: str, fields: Sequence[str] | None = None, input_keys: tuple[str, ...] = ()
    ) -> list[Example]:
        loaded_dataset: Any = load_dataset("json", data_files=file_path)
        dataset = loaded_dataset["train"]
        return rows_to_examples(
            rows=cast("Iterable[Mapping[str, object]]", dataset), fields=fields, input_keys=input_keys
        )

    def from_parquet(
        self, file_path: str, fields: Sequence[str] | None = None, input_keys: tuple[str, ...] = ()
    ) -> list[Example]:
        loaded_dataset: Any = load_dataset("parquet", data_files=file_path)
        dataset = loaded_dataset["train"]
        return rows_to_examples(
            rows=cast("Iterable[Mapping[str, object]]", dataset), fields=fields, input_keys=input_keys
        )
