"""Dataset summary generation for grounded instruction proposal.

Import ``create_dataset_summary`` from ``dspy.propose.dataset_summary_generator``.
"""

import logging
import re

from dspy.core.types.config import LMConfig
from dspy.predict.predict import Predict
from dspy.propose.utils import strip_prefix
from dspy.runtime.run_context import RunContext
from dspy.task_spec import FieldSpec, TaskSpec, input_field, output_field
from dspy.teleprompt.task_spec_context import get_prompt_model
from dspy.teleprompt.utils import optimizer_lm_context

logger = logging.getLogger(__name__)

__all__ = ["create_dataset_summary"]


class ObservationSummarizerTaskSpec(TaskSpec):
    name: str = "framework.propose.observation_summarizer"
    instructions: str = "Given a series of observations I have made about my dataset, please summarize them into a brief 2-3 sentence summary which highlights only the most important details."
    inputs: tuple[FieldSpec, ...] = (
        input_field("observations", str, desc="Observations I have made about my dataset"),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field(
            "summary",
            str,
            desc="Two-to-three sentence summary of only the most significant highlights of my observations",
        ),
    )


class DatasetDescriptorTaskSpec(TaskSpec):
    name: str = "framework.propose.dataset_descriptor"
    instructions: str = "Given several examples from a dataset please write observations about trends that hold for most or all of the samples. Some areas you may consider in your observations: topics, content, syntax, conciseness, etc. It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative"
    inputs: tuple[FieldSpec, ...] = (input_field("examples", str, desc="Sample data points from the dataset"),)
    outputs: tuple[FieldSpec, ...] = (
        output_field(
            "observations",
            str,
            desc="Something that holds true for most or all of the data you observed",
        ),
    )


class DatasetDescriptorWithPriorObservationsTaskSpec(TaskSpec):
    name: str = "framework.propose.dataset_descriptor_with_prior"
    instructions: str = "Given several examples from a dataset please write observations about trends that hold for most or all of the samples. I will also provide you with a few observations I have already made. Please add your own observations or if you feel the observations are comprehensive say 'COMPLETE'. Some areas you may consider in your observations: topics, content, syntax, conciseness, etc. It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative"
    inputs: tuple[FieldSpec, ...] = (
        input_field("examples", str, desc="Sample data points from the dataset"),
        input_field("prior_observations", str, desc="Some prior observations I made about the data"),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field(
            "observations",
            str,
            desc="Something that holds true for most or all of the data you observed or COMPLETE if you have nothing to add",
        ),
    )


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
    if verbose:
        logger.debug("Creating dataset summary for %s examples", len(trainset))
    upper_lim = min(len(trainset), view_data_batch_size)
    prompt_model = get_prompt_model(prompt_model, run)
    with optimizer_lm_context(run, lm=prompt_model, phase="propose.dataset_summary", lm_role="prompt_model") as opt_run:
        observation = await Predict(DatasetDescriptorTaskSpec(), config=LMConfig(n=1, temperature=1.0))(
            examples=order_input_keys_in_string(trainset[0:upper_lim].__repr__()), run=opt_run
        )
    observations = observation["observations"]
    if log_file:
        log_file.write("PRODUCING DATASET SUMMARY\n")
    skips = 0
    try:
        max_calls = 10
        calls = 0
        for b in range(view_data_batch_size, len(trainset), view_data_batch_size):
            calls += 1
            if calls >= max_calls:
                break
            upper_lim = min(len(trainset), b + view_data_batch_size)
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
            observations += output["observations"]
            if log_file:
                log_file.write(f"observations {observations}\n")
    except Exception:
        logger.debug(
            "Incremental dataset summary observation failed; continuing with partial observations.", exc_info=True
        )
    with optimizer_lm_context(run, lm=prompt_model, phase="propose.dataset_summary", lm_role="prompt_model") as opt_run:
        summary = await Predict(ObservationSummarizerTaskSpec(), config=LMConfig(n=1, temperature=1.0))(
            observations=observations, run=opt_run
        )
    if log_file:
        log_file.write(f"summary: {summary}\n")
    return strip_prefix(summary.summary)
