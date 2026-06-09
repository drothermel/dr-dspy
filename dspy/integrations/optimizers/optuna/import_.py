from typing import Any

from dspy._internal.lazy_import import _detect_dspy_dist


def import_optuna(*, feature: str = "Optuna") -> Any:
    try:
        import optuna
    except ModuleNotFoundError as exc:
        if exc.name == "optuna":
            raise ImportError(
                f"{feature} requires optional dependency 'optuna'. "
                f"Install it with `pip install {_detect_dspy_dist()}[optuna]`."
            ) from exc
        raise
    return optuna
