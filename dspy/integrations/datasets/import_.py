from typing import Any

from dspy._internal.lazy_import import _detect_dspy_dist


def import_datasets(*, feature: str = "Dataset integrations") -> Any:
    try:
        import datasets
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"{feature} requires optional dependency 'datasets'. "
            f"Install it with `pip install {_detect_dspy_dist()}[datasets]`."
        ) from exc
    return datasets
