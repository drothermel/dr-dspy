import re

from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Predict
from dspy.propose.utils import strip_prefix
from dspy.task_spec import FieldSpec, TaskSpec, input_field, output_field


class ObservationSummarizerTaskSpec(TaskSpec):
    name: str = "ObservationSummarizer"
    instructions: str = (
        "Given a series of observations I have made about my dataset, please summarize them into a brief 2-3 sentence "
        "summary which highlights only the most important details."
    )
    inputs: tuple[FieldSpec, ...] = (
        input_field("observations", str, desc="Observations I have made about my dataset"),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field(
            "summary",
            str,
            desc="Two to Three sentence summary of only the most significant highlights of my observations",
        ),
    )


class DatasetDescriptorTaskSpec(TaskSpec):
    name: str = "DatasetDescriptor"
    instructions: str = (
        "Given several examples from a dataset please write observations about trends that hold for most or all of "
        "the samples. Some areas you may consider in your observations: topics, content, syntax, conciseness, etc. "
        "It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be "
        "afraid to be creative"
    )
    inputs: tuple[FieldSpec, ...] = (input_field("examples", str, desc="Sample data points from the dataset"),)
    outputs: tuple[FieldSpec, ...] = (
        output_field(
            "observations",
            str,
            desc="Somethings that holds true for most or all of the data you observed",
        ),
    )


class DatasetDescriptorWithPriorObservationsTaskSpec(TaskSpec):
    name: str = "DatasetDescriptorWithPriorObservations"
    instructions: str = (
        "Given several examples from a dataset please write observations about trends that hold for most or all of the "
        "samples. I will also provide you with a few observations I have already made. Please add your own observations "
        "or if you feel the observations are comprehensive say 'COMPLETE'. Some areas you may consider in your "
        "observations: topics, content, syntax, conciceness, etc. It will be useful to make an educated guess as to the "
        "nature of the task this dataset will enable. Don't be afraid to be creative"
    )
    inputs: tuple[FieldSpec, ...] = (
        input_field("examples", str, desc="Sample data points from the dataset"),
        input_field("prior_observations", str, desc="Some prior observations I made about the data"),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field(
            "observations",
            str,
            desc="Somethings that holds true for most or all of the data you observed or COMPLETE if you have nothing to add",
        ),
    )


def order_input_keys_in_string(unordered_repr):
    """Sort input_keys={...} repr fragments for deterministic dataset summaries."""
    pattern = r"input_keys=\{([^\}]+)\}"

    def reorder_keys(match) -> str:
        keys_str = match.group(1)
        keys = sorted(key.strip() for key in keys_str.split(","))
        return f"input_keys={{{', '.join(keys)}}}"

    return re.sub(pattern, reorder_keys, unordered_repr)


async def create_dataset_summary(*, trainset, view_data_batch_size, prompt_model, log_file=None, verbose=False):
    if verbose:
        pass
    upper_lim = min(len(trainset), view_data_batch_size)
    prompt_model = prompt_model if prompt_model else settings.lm
    with settings.context(lm=prompt_model):
        observation = await Predict(DatasetDescriptorTaskSpec(), n=1, temperature=1.0)(
            examples=order_input_keys_in_string(trainset[0:upper_lim].__repr__())
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
            if verbose:
                pass
            upper_lim = min(len(trainset), b + view_data_batch_size)
            with settings.context(lm=prompt_model):
                output = await Predict(DatasetDescriptorWithPriorObservationsTaskSpec(), n=1, temperature=1.0)(
                    prior_observations=observations,
                    examples=order_input_keys_in_string(trainset[b:upper_lim].__repr__()),
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
        if verbose:
            pass

    if prompt_model:
        with settings.context(lm=prompt_model):
            summary = await Predict(ObservationSummarizerTaskSpec(), n=1, temperature=1.0)(observations=observations)
    else:
        summary = await Predict(ObservationSummarizerTaskSpec(), n=1, temperature=1.0)(observations=observations)
    if verbose:
        pass
    if log_file:
        log_file.write(f"summary: {summary}\n")

    if verbose:
        pass

    return strip_prefix(summary.summary)
