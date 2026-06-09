import random

from dspy.runtime.run_context import RunContext


def create_minibatch(*, trainset, batch_size=50, rng=None):
    batch_size = min(batch_size, len(trainset))
    rng = rng or random
    sampled_indices = rng.sample(range(len(trainset)), batch_size)
    return [trainset[i] for i in sampled_indices]


async def eval_candidate_program(*, batch_size, trainset, candidate_program, evaluate, run: RunContext, rng=None):
    if batch_size >= len(trainset):
        return await evaluate(
            candidate_program, run=run, devset=trainset, callback_metadata={"metric_key": "eval_full"}
        )
    return await evaluate(
        candidate_program,
        run=run,
        devset=create_minibatch(trainset=trainset, batch_size=batch_size, rng=rng),
        callback_metadata={"metric_key": "eval_minibatch"},
    )


def get_program_with_highest_avg_score(*, param_score_dict, fully_evaled_param_combos):
    results = []
    for key, values in param_score_dict.items():
        scores = [v[0] for v in values]
        mean = sum(scores) / len(scores)
        program = values[0][1]
        params = values[0][2]
        results.append((key, mean, program, params))
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    for combination in sorted_results:
        key, mean, program, params = combination
        if key in fully_evaled_param_combos:
            continue
        return (program, mean, key, params)
    raise ValueError("No unevaluated parameter combination found with a recorded score.")


async def get_task_model_history_for_full_example(candidate_program, task_model, devset, evaluate, run: RunContext):
    _ = await evaluate(candidate_program, run=run, devset=devset[:1])
    n = len(candidate_program.predictors())
    return list(task_model.call_log[-n:])
