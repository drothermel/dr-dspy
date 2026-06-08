import asyncio
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from dspy.evaluate.evaluate import Evaluate
from dspy.teleprompt.mipro.evaluate import (
    log_minibatch_eval,
    log_normal_eval,
    perform_full_evaluation,
    select_and_insert_instructions_and_demos,
)
from dspy.teleprompt.mipro.settings import ENDC, GREEN
from dspy.teleprompt.utils import eval_candidate_program, print_full_program, save_candidate_program

if TYPE_CHECKING:
    import optuna

    from dspy.teleprompt.mipro.optimizer import MIPROv2

logger = logging.getLogger(__name__)

_mipro_optuna_executor = ThreadPoolExecutor(max_workers=1)


def run_async_from_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return _mipro_optuna_executor.submit(asyncio.run, coro).result()


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


def objective(
    optimizer: "MIPROv2",
    trial: "optuna.trial.Trial",
    *,
    program: Any,
    instruction_candidates: dict[int, list[str]],
    demo_candidates: list | None,
    evaluate: Evaluate,
    valset: list,
    num_trials: int,
    minibatch: bool,
    minibatch_size: int,
    minibatch_full_eval_steps: int,
    adjusted_num_trials: int,
    study: "optuna.Study",
    state: dict[str, Any],
) -> float:
    best_program = state["best_program"]
    best_score = state["best_score"]
    trial_logs = state["trial_logs"]
    total_eval_calls = state["total_eval_calls"]
    score_data = state["score_data"]
    param_score_dict = state["param_score_dict"]
    fully_evaled_param_combos = state["fully_evaled_param_combos"]

    trial_num = trial.number + 1
    if minibatch:
        logger.info(f"== Trial {trial_num} / {adjusted_num_trials} - Minibatch ==")
    else:
        logger.info(f"===== Trial {trial_num} / {num_trials} =====")

    trial_logs[trial_num] = {}

    candidate_program = program.deepcopy()

    chosen_params, raw_chosen_params = select_and_insert_instructions_and_demos(
        candidate_program,
        instruction_candidates,
        demo_candidates,
        trial,
        trial_logs,
        trial_num,
    )

    if optimizer.verbose:
        logger.info("Evaluating the following candidate program...\n")
        print_full_program(candidate_program)

    batch_size = minibatch_size if minibatch else len(valset)
    score = run_async_from_sync(
        eval_candidate_program(
            batch_size=batch_size,
            trainset=valset,
            candidate_program=candidate_program,
            evaluate=evaluate,
            rng=optimizer.rng,
        )
    ).score
    total_eval_calls += batch_size

    if not minibatch and score > best_score:
        best_score = score
        best_program = candidate_program.deepcopy()
        logger.info(f"{GREEN}Best full score so far!{ENDC} Score: {score}")

    score_data.append(
        {"score": score, "program": candidate_program, "full_eval": batch_size >= len(valset)}
    )  # score, prog, full_eval
    if minibatch:
        log_minibatch_eval(
            optimizer,
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
        )
    else:
        log_normal_eval(
            optimizer,
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
        )
    categorical_key = ",".join(map(str, chosen_params))
    param_score_dict[categorical_key].append(
        (score, candidate_program, raw_chosen_params),
    )

    # If minibatch, perform full evaluation at intervals (and at the very end)
    if minibatch and ((trial_num % (minibatch_full_eval_steps + 1) == 0) or (trial_num == (adjusted_num_trials - 1))):
        best_score, best_program, total_eval_calls = run_async_from_sync(
            perform_full_evaluation(
                optimizer,
                trial_num,
                adjusted_num_trials,
                param_score_dict,
                fully_evaled_param_combos,
                evaluate,
                valset,
                trial_logs,
                total_eval_calls,
                score_data,
                best_score,
                best_program,
                study,
                instruction_candidates,
                demo_candidates,
            )
        )

    state["best_program"] = best_program
    state["best_score"] = best_score
    state["total_eval_calls"] = total_eval_calls

    return score


async def optimize_prompt_parameters(
    optimizer: "MIPROv2",
    program: Any,
    instruction_candidates: dict[int, list[str]],
    demo_candidates: list | None,
    evaluate: Evaluate,
    valset: list,
    num_trials: int,
    minibatch: bool,
    minibatch_size: int,
    minibatch_full_eval_steps: int,
    seed: int,
) -> Any | None:
    optuna = import_optuna()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info("==> STEP 3: FINDING OPTIMAL PROMPT PARAMETERS <==")
    logger.info(
        "We will evaluate the program over a series of trials with different combinations of instructions and few-shot examples to find the optimal combination using Bayesian Optimization.\n"
    )

    run_additional_full_eval_at_end = 1 if num_trials % minibatch_full_eval_steps != 0 else 0
    adjusted_num_trials = int(
        (num_trials + num_trials // minibatch_full_eval_steps + 1 + run_additional_full_eval_at_end)
        if minibatch
        else num_trials
    )
    logger.info(f"== Trial {1} / {adjusted_num_trials} - Full Evaluation of Default Program ==")

    default_score = (
        await eval_candidate_program(
            batch_size=len(valset),
            trainset=valset,
            candidate_program=program,
            evaluate=evaluate,
            rng=optimizer.rng,
        )
    ).score
    logger.info(f"Default program score: {default_score}\n")

    trial_logs = {}
    trial_logs[1] = {}
    trial_logs[1]["full_eval_program_path"] = save_candidate_program(
        program=program, log_dir=optimizer.log_dir, trial_num=-1
    )
    trial_logs[1]["full_eval_score"] = default_score
    trial_logs[1]["total_eval_calls_so_far"] = len(valset)
    trial_logs[1]["full_eval_program"] = program.deepcopy()

    best_score = default_score
    best_program = program.deepcopy()
    total_eval_calls = len(valset)
    score_data = [{"score": best_score, "program": program.deepcopy(), "full_eval": True}]
    param_score_dict = defaultdict(list)
    fully_evaled_param_combos = {}

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    default_params = {f"{i}_predictor_instruction": 0 for i in range(len(program.predictors()))}
    if demo_candidates:
        default_params.update({f"{i}_predictor_demos": 0 for i in range(len(program.predictors()))})

    # TODO: Account for the default program being evaluated on a different sample count before adding it as an Optuna baseline trial.
    trial = optuna.trial.create_trial(
        params=default_params,
        distributions=get_param_distributions(
            program=program,
            instruction_candidates=instruction_candidates,
            demo_candidates=demo_candidates,
        ),
        value=default_score,
    )
    study.add_trial(trial)

    state = {
        "best_program": best_program,
        "best_score": best_score,
        "trial_logs": trial_logs,
        "total_eval_calls": total_eval_calls,
        "score_data": score_data,
        "param_score_dict": param_score_dict,
        "fully_evaled_param_combos": fully_evaled_param_combos,
    }

    study.optimize(
        lambda trial: objective(
            optimizer,
            trial,
            program=program,
            instruction_candidates=instruction_candidates,
            demo_candidates=demo_candidates,
            evaluate=evaluate,
            valset=valset,
            num_trials=num_trials,
            minibatch=minibatch,
            minibatch_size=minibatch_size,
            minibatch_full_eval_steps=minibatch_full_eval_steps,
            adjusted_num_trials=adjusted_num_trials,
            study=study,
            state=state,
        ),
        n_trials=num_trials,
    )

    best_program = state["best_program"]
    best_score = state["best_score"]
    trial_logs = state["trial_logs"]
    score_data = state["score_data"]

    # Attach logs to best program
    if best_program is not None and optimizer.track_stats:
        best_program.trial_logs = trial_logs
        best_program.score = best_score
        best_program.prompt_model_total_calls = optimizer.prompt_model_total_calls
        best_program.total_calls = optimizer.total_calls
        sorted_candidate_programs = sorted(score_data, key=lambda x: x["score"], reverse=True)
        # Attach all minibatch programs
        best_program.mb_candidate_programs = [
            score_data for score_data in sorted_candidate_programs if not score_data["full_eval"]
        ]
        # Attach all programs that were evaluated on the full trainset, in descending order of score
        best_program.candidate_programs = [
            score_data for score_data in sorted_candidate_programs if score_data["full_eval"]
        ]

    logger.info(f"Returning best identified program with score {best_score}!")

    return best_program
