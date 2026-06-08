import logging
import math
import random

from dspy.primitives.prediction import Prediction
from dspy.runtime.run_context import RunContext

logger = logging.getLogger(__name__)


def create_minibatch(*, trainset, batch_size=50, rng=None):
    batch_size = min(batch_size, len(trainset))
    rng = rng or random
    sampled_indices = rng.sample(range(len(trainset)), batch_size)
    return [trainset[i] for i in sampled_indices]


async def eval_candidate_program(*, batch_size, trainset, candidate_program, evaluate, run: RunContext, rng=None):
    try:
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
    except Exception:
        logger.error("An exception occurred during evaluation", exc_info=True)
        return Prediction(score=0.0, results=[])


async def eval_candidate_program_with_pruning(
    trial, trial_logs, trainset, candidate_program, evaluate, run: RunContext, trial_num, batch_size=100
):
    total_score = 0
    num_batches = math.ceil(len(trainset) / batch_size)
    total_eval_size = 0
    curr_weighted_avg_score = 0.0
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(trainset))
        split_trainset = trainset[start_index:end_index]
        split_score = (await evaluate(candidate_program, run=run, devset=split_trainset, display_table=0)).score
        total_eval_size += len(split_trainset)
        total_score += split_score * len(split_trainset)
        curr_weighted_avg_score = total_score / min((i + 1) * batch_size, len(trainset))
        trial.report(curr_weighted_avg_score, i)
        if trial.should_prune():
            trial_logs[trial_num]["score"] = curr_weighted_avg_score
            trial_logs[trial_num]["num_eval_calls"] = total_eval_size
            trial_logs[trial_num]["pruned"] = True
            return (curr_weighted_avg_score, trial_logs, total_eval_size, True)
    score = curr_weighted_avg_score
    trial_logs[trial_num]["full_eval"] = False
    trial_logs[trial_num]["score"] = score
    trial_logs[trial_num]["pruned"] = False
    return (score, trial_logs, total_eval_size, False)


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


async def calculate_last_n_proposed_quality(base_program, trial_logs, evaluate, run: RunContext, trainset, devset, n):
    last_n_trial_nums = list(trial_logs.keys())[-n:]
    total_train_score = 0
    best_train_score = 0
    total_dev_score = 0
    best_dev_score = 0
    for trial_num in last_n_trial_nums:
        full_eval = trial_logs[trial_num]["full_eval"]
        if not full_eval:
            raise NotImplementedError(
                "Still need to implement non full eval handling in calculate_last_n_proposed_quality"
            )
        train_score = trial_logs[trial_num]["score"]
        program = base_program.deepcopy()
        program.load(trial_logs[trial_num]["program_path"])
        dev_score = (await evaluate(program, run=run, devset=devset)).score
        total_train_score += train_score
        total_dev_score += dev_score
        if train_score > best_train_score:
            best_train_score = train_score
            best_dev_score = dev_score
    return (best_train_score, total_train_score / n, best_dev_score, total_dev_score / n)


async def get_task_model_history_for_full_example(candidate_program, task_model, devset, evaluate, run: RunContext):
    _ = await evaluate(candidate_program, run=run, devset=devset[:1])
    _ = task_model.inspect_history(n=len(candidate_program.predictors()))
    return task_model.inspect_history(n=len(candidate_program.predictors()))
