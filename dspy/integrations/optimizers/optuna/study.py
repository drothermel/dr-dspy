from collections.abc import Awaitable, Callable
from typing import Any

from dspy.integrations.optimizers.optuna.import_ import import_optuna


def create_maximize_study(*, seed: int | None = None, feature: str = "Optuna") -> Any:
    optuna = import_optuna(feature=feature)
    if seed is not None:
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        return optuna.create_study(direction="maximize", sampler=sampler)
    return optuna.create_study(direction="maximize")


def add_observed_trial(
    study: Any,
    *,
    params: dict[str, Any],
    distributions: dict[str, Any],
    value: float,
    feature: str = "Optuna",
) -> None:
    optuna = import_optuna(feature=feature)
    trial = optuna.trial.create_trial(params=params, distributions=distributions, value=value)
    study.add_trial(trial)


async def run_ask_tell_loop(
    study: Any,
    num_trials: int,
    trial_fn: Callable[[Any], Awaitable[float]],
) -> None:
    for _ in range(num_trials):
        trial = study.ask()
        score = await trial_fn(trial)
        study.tell(trial, score)
