"""Dataset summary generation for grounded instruction proposal.

Import ``create_dataset_summary`` from ``dspy.propose.dataset_summary_generator``.
"""

import logging
import re

from dspy.core.types.config import LMConfig
from dspy.predict.predict import Predict
from dspy.propose.task_specs import (
    DatasetDescriptorTaskSpec,
    DatasetDescriptorWithPriorObservationsTaskSpec,
    ObservationSummarizerTaskSpec,
)
from dspy.propose.utils import strip_prefix
from dspy.runtime.run_context import RunContext
from dspy.task_spec.predictor_context import resolve_optimizer_lm
from dspy.teleprompt.utils import optimizer_lm_context

logger = logging.getLogger(__name__)

__all__ = ["create_dataset_summary"]


def order_input_keys_in_string(unordered_repr):
    pattern = "input_keys=\\{([^\\}]+)\\}"

    def reorder_keys(match) -> str:
        keys_str = match.group(1)
        keys = sorted(key.strip() for key in keys_str.split(","))
        return f"input_keys={{{', '.join(keys)}}}"

    return re.sub(pattern, reorder_keys, unordered_repr)


async def create_dataset_summary(
    *, trainset, view_data_batch_size, prompt_model, run: RunContext, log_file=None, verbose=False
):
    if not trainset:
        raise ValueError("trainset must be non-empty for dataset summary")
    if verbose:
        logger.info("Creating dataset summary for %s examples", len(trainset))
    upper_lim = min(len(trainset), view_data_batch_size)
    prompt_model = resolve_optimizer_lm(prompt_model, run=run)
    with optimizer_lm_context(run, lm=prompt_model, phase="propose.dataset_summary", lm_role="prompt_model") as opt_run:
        observation = await Predict(DatasetDescriptorTaskSpec(), config=LMConfig(n=1, temperature=1.0))(
            examples=order_input_keys_in_string(trainset[0:upper_lim].__repr__()), run=opt_run
        )
    observations = observation["observations"]
    if log_file:
        log_file.write("PRODUCING DATASET SUMMARY\n")
    skips = 0
    max_calls = 10
    calls = 0
    for b in range(view_data_batch_size, len(trainset), view_data_batch_size):
        calls += 1
        if calls >= max_calls:
            break
        upper_lim = min(len(trainset), b + view_data_batch_size)
        if verbose:
            logger.info(
                "Dataset summary incremental batch %s/%s (examples %s:%s)",
                calls,
                max_calls,
                b,
                upper_lim,
            )
        with optimizer_lm_context(
            run, lm=prompt_model, phase="propose.dataset_summary", lm_role="prompt_model"
        ) as opt_run:
            output = await Predict(
                DatasetDescriptorWithPriorObservationsTaskSpec(), config=LMConfig(n=1, temperature=1.0)
            )(
                prior_observations=observations,
                examples=order_input_keys_in_string(trainset[b:upper_lim].__repr__()),
                run=opt_run,
            )
        if len(output["observations"]) >= 8 and output["observations"][:8].upper() == "COMPLETE":
            skips += 1
            if skips >= 5:
                break
            continue
        observations += "\n" + output["observations"]
        if log_file:
            log_file.write(f"observations {observations}\n")
    with optimizer_lm_context(run, lm=prompt_model, phase="propose.dataset_summary", lm_role="prompt_model") as opt_run:
        summary = await Predict(ObservationSummarizerTaskSpec(), config=LMConfig(n=1, temperature=1.0))(
            observations=observations, run=opt_run
        )
    if log_file:
        log_file.write(f"summary: {summary}\n")
    return strip_prefix(summary.summary)
