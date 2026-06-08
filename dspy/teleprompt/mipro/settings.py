import logging
import math
import random
from typing import TYPE_CHECKING, Any

from dspy.teleprompt.eval_batch import create_minibatch

if TYPE_CHECKING:
    from dspy.teleprompt.mipro.optimizer import MIPROv2
logger = logging.getLogger(__name__)
BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT = 3
LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT = 0
MIN_MINIBATCH_SIZE = 50
AUTO_RUN_SETTINGS = {
    "light": {"n": 6, "val_size": 100},
    "medium": {"n": 12, "val_size": 300},
    "heavy": {"n": 18, "val_size": 1000},
}
YELLOW = "\x1b[93m"
GREEN = "\x1b[92m"
BLUE = "\x1b[94m"
BOLD = "\x1b[1m"
ENDC = "\x1b[0m"


def set_random_seeds(optimizer: "MIPROv2", seed: int) -> None:
    optimizer.rng = random.Random(seed)


def set_num_trials_from_num_candidates(
    optimizer: "MIPROv2", program: Any, zeroshot_opt: bool, num_candidates: int
) -> int:
    num_vars = len(program.predictors())
    if not zeroshot_opt:
        num_vars *= 2
    return int(max(2 * num_vars * math.log2(num_candidates), 1.5 * num_candidates))


def set_hyperparams_from_run_mode(
    optimizer: "MIPROv2",
    program: Any,
    num_trials: int | None,
    minibatch: bool,
    zeroshot_opt: bool,
    valset: list,
    num_instruct_candidates: int | None,
    num_fewshot_candidates: int | None,
) -> tuple[int, list, bool, int, int]:
    if optimizer.auto is None:
        if num_trials is None:
            raise ValueError("num_trials must be provided when auto is None.")
        if num_instruct_candidates is None or num_fewshot_candidates is None:
            raise ValueError("num_candidates must be provided when auto is None.")
        return (num_trials, valset, minibatch, num_instruct_candidates, num_fewshot_candidates)
    auto_settings = AUTO_RUN_SETTINGS[optimizer.auto]
    valset = create_minibatch(trainset=valset, batch_size=auto_settings["val_size"], rng=optimizer.rng)
    minibatch = len(valset) > MIN_MINIBATCH_SIZE
    num_instruct_candidates = auto_settings["n"] if zeroshot_opt else int(auto_settings["n"] * 0.5)
    num_fewshot_candidates = auto_settings["n"]
    num_trials = set_num_trials_from_num_candidates(
        optimizer=optimizer, program=program, zeroshot_opt=zeroshot_opt, num_candidates=auto_settings["n"]
    )
    return (num_trials, valset, minibatch, num_instruct_candidates, num_fewshot_candidates)


def set_and_validate_datasets(trainset: list, valset: list | None) -> tuple[list, list]:
    if not trainset:
        raise ValueError("Trainset cannot be empty.")
    if valset is None:
        if len(trainset) < 2:
            raise ValueError("Trainset must have at least 2 examples if no valset specified.")
        valset_size = min(1000, max(1, int(len(trainset) * 0.8)))
        cutoff = len(trainset) - valset_size
        valset = trainset[cutoff:]
        trainset = trainset[:cutoff]
    elif len(valset) < 1:
        raise ValueError("Validation set must have at least 1 example.")
    return (trainset, valset)


def print_auto_run_settings(
    optimizer: "MIPROv2",
    num_trials: int,
    minibatch: bool,
    valset: list,
    num_fewshot_candidates: int,
    num_instruct_candidates: int,
) -> None:
    assert optimizer.auto is not None
    logger.info(
        f"\nRUNNING WITH THE FOLLOWING {optimizer.auto.upper()} AUTO RUN SETTINGS:\nnum_trials: {num_trials}\nminibatch: {minibatch}\nnum_fewshot_candidates: {num_fewshot_candidates}\nnum_instruct_candidates: {num_instruct_candidates}\nvalset size: {len(valset)}\n"
    )


def estimate_lm_calls(
    optimizer: "MIPROv2",
    program: Any,
    num_trials: int,
    minibatch: bool,
    minibatch_size: int,
    minibatch_full_eval_steps: int,
    valset: list,
    program_aware_proposer: bool,
    num_instruct_candidates: int,
) -> tuple[str, str]:
    num_predictors = len(program.predictors())
    estimated_prompt_model_calls = (
        10 + num_instruct_candidates * num_predictors + (num_predictors + 1 if program_aware_proposer else 0)
    )
    prompt_model_line = f"{YELLOW}- Prompt Generation: {BLUE}{BOLD}10{ENDC}{YELLOW} data summarizer calls + {BLUE}{BOLD}{num_instruct_candidates}{ENDC}{YELLOW} * {BLUE}{BOLD}{num_predictors}{ENDC}{YELLOW} lm calls in program + ({BLUE}{BOLD}{num_predictors + 1}{ENDC}{YELLOW}) lm calls in program-aware proposer = {BLUE}{BOLD}{estimated_prompt_model_calls}{ENDC}{YELLOW} prompt model calls{ENDC}"
    if not minibatch:
        estimated_task_model_calls = len(valset) * num_trials
        task_model_line = f"{YELLOW}- Program Evaluation: {BLUE}{BOLD}{len(valset)}{ENDC}{YELLOW} examples in val set * {BLUE}{BOLD}{num_trials}{ENDC}{YELLOW} batches = {BLUE}{BOLD}{estimated_task_model_calls}{ENDC}{YELLOW} LM program calls{ENDC}"
    else:
        full_eval_steps = num_trials // minibatch_full_eval_steps + 1
        estimated_task_model_calls = minibatch_size * num_trials + len(valset) * full_eval_steps
        task_model_line = f"{YELLOW}- Program Evaluation: {BLUE}{BOLD}{minibatch_size}{ENDC}{YELLOW} examples in minibatch * {BLUE}{BOLD}{num_trials}{ENDC}{YELLOW} batches + {BLUE}{BOLD}{len(valset)}{ENDC}{YELLOW} examples in val set * {BLUE}{BOLD}{full_eval_steps}{ENDC}{YELLOW} full evals = {BLUE}{BOLD}{estimated_task_model_calls}{ENDC}{YELLOW} LM Program calls{ENDC}"
    return (prompt_model_line, task_model_line)
