from typing import Any


def import_optuna(*, feature: str = "Optuna") -> Any:
    try:
        import optuna
    except ModuleNotFoundError as exc:
        if exc.name == "optuna":
            raise ImportError(
                f"{feature} requires optional dependency 'optuna'. Install it with `pip install dspy[optuna]`."
            ) from exc
        raise
    return optuna
