def import_optuna():
    try:
        import optuna
    except ModuleNotFoundError as exc:
        if exc.name == "optuna":
            raise ImportError(
                "MIPROv2 requires optional dependency 'optuna'. Install it with `pip install dspy[optuna]`."
            ) from exc
        raise
    return optuna


def get_param_distributions(program, instruction_candidates, demo_candidates):
    optuna = import_optuna()
    CategoricalDistribution = optuna.distributions.CategoricalDistribution
    param_distributions = {}
    for i in range(len(instruction_candidates)):
        param_distributions[f"{i}_predictor_instruction"] = CategoricalDistribution(
            range(len(instruction_candidates[i]))
        )
        if demo_candidates:
            param_distributions[f"{i}_predictor_demos"] = CategoricalDistribution(range(len(demo_candidates[i])))
    return param_distributions
