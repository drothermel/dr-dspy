from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from dspy.datasets.rows import rows_to_examples
from dspy.integrations.datasets.import_ import import_datasets
from dspy.primitives import Example


def _hf_datasets() -> Any:
    return import_datasets(feature="examples_from_huggingface")


def _examples_from_hf_file(
    format_name: str, file_path: str, fields: Sequence[str] | None, input_keys: tuple[str, ...]
) -> list[Example]:
    load_dataset = _hf_datasets().load_dataset
    loaded_dataset: Any = load_dataset(format_name, data_files=file_path)
    dataset = loaded_dataset["train"]
    return rows_to_examples(rows=cast("Iterable[Mapping[str, object]]", dataset), fields=fields, input_keys=input_keys)


def examples_from_huggingface(
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

    hf_datasets = _hf_datasets()
    load_dataset = hf_datasets.load_dataset
    dataset_dict = hf_datasets.DatasetDict

    dataset = load_dataset(dataset_name, *args, **kwargs)
    if isinstance(dataset, list) and isinstance(kwargs.get("split"), list):
        split_names = cast("list[str]", kwargs["split"])
        return {
            split_name: rows_to_examples(
                rows=cast("Iterable[Mapping[str, object]]", split_rows), fields=fields, input_keys=input_keys
            )
            for split_name, split_rows in zip(split_names, dataset, strict=False)
        }
    if isinstance(dataset, dataset_dict):
        return {
            split_name: rows_to_examples(rows=rows, fields=fields, input_keys=input_keys)
            for split_name, rows in dataset.items()
        }
    return rows_to_examples(rows=cast("Iterable[Mapping[str, object]]", dataset), fields=fields, input_keys=input_keys)


def examples_from_csv(
    file_path: str, fields: Sequence[str] | None = None, input_keys: tuple[str, ...] = ()
) -> list[Example]:
    return _examples_from_hf_file("csv", file_path, fields, input_keys)


def examples_from_json(
    file_path: str, fields: Sequence[str] | None = None, input_keys: tuple[str, ...] = ()
) -> list[Example]:
    return _examples_from_hf_file("json", file_path, fields, input_keys)


def examples_from_parquet(
    file_path: str, fields: Sequence[str] | None = None, input_keys: tuple[str, ...] = ()
) -> list[Example]:
    return _examples_from_hf_file("parquet", file_path, fields, input_keys)
