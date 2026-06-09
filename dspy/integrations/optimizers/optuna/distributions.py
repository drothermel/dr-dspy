from typing import Any

from dspy.integrations.optimizers.optuna.import_ import import_optuna


def get_param_distributions(
    program: Any,
    instruction_candidates: dict[int, list[str]],
    demo_candidates: list | None,
) -> dict[str, Any]:
    optuna = import_optuna(feature="MIPROv2")
    categorical_distribution = optuna.distributions.CategoricalDistribution
    param_distributions = {}
    for i in range(len(instruction_candidates)):
        param_distributions[f"{i}_predictor_instruction"] = categorical_distribution(
            range(len(instruction_candidates[i]))
        )
        if demo_candidates:
            param_distributions[f"{i}_predictor_demos"] = categorical_distribution(range(len(demo_candidates[i])))
    return param_distributions
