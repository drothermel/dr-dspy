import logging
from typing import TYPE_CHECKING, Any

from dspy.evaluate.evaluator import Evaluate
from dspy.integrations.optimizers.optuna.distributions import get_param_distributions
from dspy.integrations.optimizers.optuna.import_ import import_optuna
from dspy.integrations.optimizers.optuna.study import add_observed_trial, create_maximize_study, run_ask_tell_loop
from dspy.runtime.run_context import RunContext
from dspy.teleprompt.compilation import CompileResult, CompileStats, ProgramCandidate
from dspy.teleprompt.console_styles import ENDC, GREEN
from dspy.teleprompt.eval_batch import eval_candidate_program
from dspy.teleprompt.log_utils import save_candidate_program
from dspy.teleprompt.mipro.evaluate import (
    log_minibatch_eval,
    log_normal_eval,
    perform_full_evaluation,
    select_and_insert_instructions_and_demos,
)
from dspy.teleprompt.mipro.session import MIPROSearchSession

if TYPE_CHECKING:
    import optuna

    from dspy.teleprompt.mipro.optimizer import MIPROv2
logger = logging.getLogger(__name__)


async def run_trial(
    optimizer: "MIPROv2",
    trial: "optuna.trial.Trial",
    *,
    program: Any,
    instruction_candidates: dict[int, list[str]],
    demo_candidates: dict[int, list] | None,
    evaluate: Evaluate,
    valset: list,
    num_trials: int,
    minibatch: bool,
    minibatch_size: int,
    minibatch_full_eval_steps: int,
    adjusted_num_trials: int,
    study: "optuna.Study",
    session: MIPROSearchSession,
    run: RunContext,
) -> float:
    best_program = session.best_program
    best_score = session.best_score
    trial_logs = session.trial_logs
    total_eval_calls = session.total_eval_calls
    score_data = session.score_data
    param_score_dict = session.param_score_dict
    fully_evaled_param_combos = session.fully_evaled_param_combos
    trial_num = trial.number + 1
    if minibatch:
        logger.info(f"== Trial {trial_num} / {adjusted_num_trials} - Minibatch ==")
    else:
        logger.info(f"===== Trial {trial_num} / {num_trials} =====")
    trial_logs[trial_num] = {}
    candidate_program = program.deepcopy()
    chosen_params, raw_chosen_params = select_and_insert_instructions_and_demos(
        candidate_program, instruction_candidates, demo_candidates, trial, trial_logs, trial_num
    )
    if optimizer.verbose:
        logger.info("Evaluating the following candidate program...")
        for name, _ in candidate_program.named_predictors():
            logger.info("Evaluating candidate predictor: %s", name)
    batch_size = minibatch_size if minibatch else len(valset)
    score = (
        await eval_candidate_program(
            batch_size=batch_size,
            trainset=valset,
            candidate_program=candidate_program,
            evaluate=evaluate,
            run=run,
            rng=optimizer.rng,
        )
    ).score
    total_eval_calls += batch_size
    if not minibatch and score > best_score:
        best_score = score
        best_program = candidate_program.deepcopy()
        logger.info(f"{GREEN}Best full score so far!{ENDC} Score: {score}")
    score_data.append(ProgramCandidate(score=score, program=candidate_program, full_eval=batch_size >= len(valset)))
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
    param_score_dict[categorical_key].append((score, candidate_program, raw_chosen_params))
    if minibatch and (trial_num % (minibatch_full_eval_steps + 1) == 0 or trial_num == adjusted_num_trials - 1):
        best_score, best_program, total_eval_calls = await perform_full_evaluation(
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
            run=run,
        )
    session.best_program = best_program
    session.best_score = best_score
    session.total_eval_calls = total_eval_calls
    return score


async def optimize_prompt_parameters(
    optimizer: "MIPROv2",
    program: Any,
    instruction_candidates: dict[int, list[str]],
    demo_candidates: dict[int, list] | None,
    evaluate: Evaluate,
    valset: list,
    num_trials: int,
    minibatch: bool,
    minibatch_size: int,
    minibatch_full_eval_steps: int,
    seed: int,
    run: RunContext,
) -> CompileResult:
    optuna = import_optuna()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info("==> STEP 3: FINDING OPTIMAL PROMPT PARAMETERS <==")
    logger.info(
        "We will evaluate the program over a series of trials with different combinations of instructions and few-shot examples to find the optimal combination using Bayesian Optimization.\n"
    )
    run_additional_full_eval_at_end = 1 if num_trials % minibatch_full_eval_steps != 0 else 0
    adjusted_num_trials = int(
        num_trials + num_trials // minibatch_full_eval_steps + 1 + run_additional_full_eval_at_end
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
            run=run,
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
    score_data = [ProgramCandidate(score=best_score, program=program.deepcopy(), full_eval=True)]
    study = create_maximize_study(seed=seed, feature="MIPROv2")
    default_params = {f"{i}_predictor_instruction": 0 for i in range(len(program.predictors()))}
    if demo_candidates:
        default_params.update({f"{i}_predictor_demos": 0 for i in range(len(program.predictors()))})
    add_observed_trial(
        study,
        params=default_params,
        distributions=get_param_distributions(
            program=program, instruction_candidates=instruction_candidates, demo_candidates=demo_candidates
        ),
        value=default_score,
        feature="MIPROv2",
    )
    session = MIPROSearchSession(
        best_program=best_program,
        best_score=best_score,
        trial_logs=trial_logs,
        total_eval_calls=total_eval_calls,
        score_data=score_data,
    )

    async def _trial_fn(trial):
        return await run_trial(
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
            session=session,
            run=run,
        )

    await run_ask_tell_loop(study, num_trials, _trial_fn)
    best_program = session.best_program
    best_score = session.best_score
    trial_logs = session.trial_logs
    score_data = session.score_data
    if best_program is None:
        return CompileResult(program=program)
    candidates: list[ProgramCandidate] = []
    if optimizer.track_stats:
        candidates = sorted(
            score_data,
            key=lambda entry: entry.score if entry.score is not None else float("-inf"),
            reverse=True,
        )
    logger.info(f"Returning best identified program with score {best_score}!")
    return CompileResult(
        program=best_program,
        candidates=candidates,
        stats=CompileStats(
            metric_calls=session.total_eval_calls,
            best_score=best_score,
            trial_logs=trial_logs if optimizer.track_stats else {},
        ),
    )
