import logging
from typing import TYPE_CHECKING, Any

from dspy.evaluate.evaluate import Evaluate
from dspy.runtime.run_context import RunContext
from dspy.teleprompt.eval_batch import eval_candidate_program, get_program_with_highest_avg_score
from dspy.teleprompt.log_utils import save_candidate_program
from dspy.teleprompt.mipro.optuna_helpers import get_param_distributions, import_optuna
from dspy.teleprompt.mipro.settings import ENDC, GREEN
from dspy.teleprompt.task_spec_context import get_task_spec, set_task_spec

if TYPE_CHECKING:
    import optuna

    from dspy.teleprompt.mipro.optimizer import MIPROv2
logger = logging.getLogger(__name__)


def log_minibatch_eval(
    optimizer: "MIPROv2",
    score,
    best_score,
    batch_size,
    chosen_params,
    score_data,
    trial,
    adjusted_num_trials,
    trial_logs,
    trial_num,
    candidate_program,
    total_eval_calls,
) -> None:
    trial_logs[trial_num]["mb_program_path"] = save_candidate_program(
        program=candidate_program, log_dir=optimizer.log_dir, trial_num=trial_num
    )
    trial_logs[trial_num]["mb_score"] = score
    trial_logs[trial_num]["total_eval_calls_so_far"] = total_eval_calls
    trial_logs[trial_num]["mb_program"] = candidate_program.deepcopy()
    logger.info(f"Score: {score} on minibatch of size {batch_size} with parameters {chosen_params}.")
    minibatch_scores = ", ".join([f"{s['score']}" for s in score_data if not s["full_eval"]])
    logger.info(f"Minibatch scores so far: {'[' + minibatch_scores + ']'}")
    full_eval_scores = ", ".join([f"{s['score']}" for s in score_data if s["full_eval"]])
    trajectory = "[" + full_eval_scores + "]"
    logger.info(f"Full eval scores so far: {trajectory}")
    logger.info(f"Best full score so far: {best_score}")
    logger.info(f"{'=' * len(f'== Trial {trial.number + 1} / {adjusted_num_trials} - Minibatch Evaluation ==')}\n\n")


def log_normal_eval(
    optimizer: "MIPROv2",
    score,
    best_score,
    chosen_params,
    score_data,
    trial,
    num_trials,
    trial_logs,
    trial_num,
    valset,
    batch_size,
    candidate_program,
    total_eval_calls,
) -> None:
    trial_logs[trial_num]["full_eval_program_path"] = save_candidate_program(
        program=candidate_program, log_dir=optimizer.log_dir, trial_num=trial_num
    )
    trial_logs[trial_num]["full_eval_score"] = score
    trial_logs[trial_num]["total_eval_calls_so_far"] = total_eval_calls
    trial_logs[trial_num]["full_eval_program"] = candidate_program.deepcopy()
    logger.info(f"Score: {score} with parameters {chosen_params}.")
    full_eval_scores = ", ".join([f"{s['score']}" for s in score_data if s["full_eval"]])
    logger.info(f"Scores so far: {'[' + full_eval_scores + ']'}")
    logger.info(f"Best score so far: {best_score}")
    logger.info(f"{'=' * len(f'===== Trial {trial.number + 1} / {num_trials} =====')}\n\n")


def select_and_insert_instructions_and_demos(
    candidate_program: Any,
    instruction_candidates: dict[int, list[str]],
    demo_candidates: list | None,
    trial: "optuna.trial.Trial",
    trial_logs: dict,
    trial_num: int,
) -> tuple[list[str], dict[str, int]]:
    chosen_params = []
    raw_chosen_params = {}
    for i, predictor in enumerate(candidate_program.predictors()):
        instruction_idx = trial.suggest_categorical(f"{i}_predictor_instruction", range(len(instruction_candidates[i])))
        selected_instruction = instruction_candidates[i][instruction_idx]
        updated_task_spec = get_task_spec(predictor).with_instructions(selected_instruction)
        set_task_spec(predictor=predictor, task_spec=updated_task_spec)
        trial_logs[trial_num][f"{i}_predictor_instruction"] = instruction_idx
        chosen_params.append(f"Predictor {i}: Instruction {instruction_idx}")
        raw_chosen_params[f"{i}_predictor_instruction"] = instruction_idx
        if demo_candidates:
            demos_idx = trial.suggest_categorical(f"{i}_predictor_demos", range(len(demo_candidates[i])))
            predictor.demos = demo_candidates[i][demos_idx]
            trial_logs[trial_num][f"{i}_predictor_demos"] = demos_idx
            chosen_params.append(f"Predictor {i}: Few-Shot Set {demos_idx}")
            raw_chosen_params[f"{i}_predictor_demos"] = demos_idx
    return (chosen_params, raw_chosen_params)


async def perform_full_evaluation(
    optimizer: "MIPROv2",
    trial_num: int,
    adjusted_num_trials: int,
    param_score_dict: dict,
    fully_evaled_param_combos: dict,
    evaluate: Evaluate,
    valset: list,
    trial_logs: dict,
    total_eval_calls: int,
    score_data,
    best_score: float,
    best_program: Any,
    study: "optuna.Study",
    instruction_candidates: dict[int, list[str]],
    demo_candidates: list | None,
    run: RunContext,
):
    optuna = import_optuna()
    logger.info(f"===== Trial {trial_num + 1} / {adjusted_num_trials} - Full Evaluation =====")
    highest_mean_program, mean_score, combo_key, params = get_program_with_highest_avg_score(
        param_score_dict=param_score_dict, fully_evaled_param_combos=fully_evaled_param_combos
    )
    logger.info(f"Doing full eval on next top averaging program (Avg Score: {mean_score}) from minibatch trials...")
    full_eval_score = (
        await eval_candidate_program(
            batch_size=len(valset),
            trainset=valset,
            candidate_program=highest_mean_program,
            evaluate=evaluate,
            run=run,
            rng=optimizer.rng,
        )
    ).score
    score_data.append({"score": full_eval_score, "program": highest_mean_program, "full_eval": True})
    trial = optuna.trial.create_trial(
        params=params,
        distributions=get_param_distributions(
            program=best_program, instruction_candidates=instruction_candidates, demo_candidates=demo_candidates
        ),
        value=full_eval_score,
    )
    study.add_trial(trial)
    fully_evaled_param_combos[combo_key] = {"program": highest_mean_program, "score": full_eval_score}
    total_eval_calls += len(valset)
    trial_logs[trial_num + 1] = {}
    trial_logs[trial_num + 1]["total_eval_calls_so_far"] = total_eval_calls
    trial_logs[trial_num + 1]["full_eval_program_path"] = save_candidate_program(
        program=highest_mean_program, log_dir=optimizer.log_dir, trial_num=trial_num + 1, note="full_eval"
    )
    trial_logs[trial_num + 1]["full_eval_program"] = highest_mean_program
    trial_logs[trial_num + 1]["full_eval_score"] = full_eval_score
    if full_eval_score > best_score:
        logger.info(f"{GREEN}New best full eval score!{ENDC} Score: {full_eval_score}")
        best_score = full_eval_score
        best_program = highest_mean_program.deepcopy()
    full_eval_scores = ", ".join([f"{s['score']}" for s in score_data if s["full_eval"]])
    trajectory = "[" + full_eval_scores + "]"
    logger.info(f"Full eval scores so far: {trajectory}")
    logger.info(f"Best full score so far: {best_score}")
    logger.info(len(f"===== Full Eval {len(fully_evaled_param_combos) + 1} =====") * "=")
    logger.info("\n")
    return (best_score, best_program, total_eval_calls)
