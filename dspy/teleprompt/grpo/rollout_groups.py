from __future__ import annotations

import logging
import random  # noqa: TC003 — fill strategy selector at runtime
from typing import Any, Literal, cast

from dspy.adapters.base import Adapter  # noqa: TC001 — adapter resolution at runtime
from dspy.clients.finetune import (
    FinetuneAssistantMessage,
    FinetuneChatMessage,
    GRPOChatData,
    GRPORolloutGroup,
)
from dspy.clients.lm import LM  # noqa: TC001 — adapter dict keys at runtime
from dspy.clients.openai_format.chat_request import message_to_openai_chat
from dspy.primitives import Example, Module  # noqa: TC001 — trace grid and student program
from dspy.runtime.optimization_trace import FailedPrediction
from dspy.runtime.run_context import RunContext  # noqa: TC001 — adapter fallback at runtime
from dspy.runtime.transparency.resolve import require_adapter
from dspy.task_spec.predictor_context import get_task_spec
from dspy.teleprompt.grpo.trace_grid import TraceGrid  # noqa: TC001 — trace grid shape
from dspy.teleprompt.grpo.types import PredictorRolloutBatches  # noqa: TC001 — rollout batches

logger = logging.getLogger(__name__)


def validate_trace_data_and_log_issues(
    trace_data: TraceGrid,
    *,
    subsample_training_dataset: list[Example],
    num_teachers: int,
    num_samples_per_input: int,
    pred_signature_hash_to_ind: dict[str, int],
) -> None:
    if len(trace_data) != len(subsample_training_dataset):
        raise ValueError(
            f"Trace data length {len(trace_data)} does not match the number of examples "
            f"{len(subsample_training_dataset)}"
        )
    if len(trace_data[0]) != num_teachers:
        raise ValueError(f"Trace data length {len(trace_data[0])} does not match the number of teachers {num_teachers}")
    if len(trace_data[0][0]) == 0:
        logger.warning(
            "Trace data for example 0 and teacher 0 is empty. This is likely due to all examples in the "
            "training set input, resulting in the model generating output not following the dspy response format."
        )
    elif len(trace_data[0][0]) != num_samples_per_input:
        logger.warning(
            f"Trace data length {len(trace_data[0][0])} does not match the expected number of samples per "
            f"input {num_samples_per_input}"
        )
        if "trace" not in trace_data[0][0][0]:
            raise ValueError("Trace data does not contain the 'trace' key")
        if len(trace_data[0][0][0]["trace"]) == 0:
            raise ValueError("Trace data is empty")
        if len(trace_data[0][0][0]["trace"][0]) != 3:
            raise ValueError(
                f"Trace tuple length {len(trace_data[0][0][0]['trace'][0])} does not match the expected length 3"
            )
    for example_data in trace_data:
        for teacher_data in example_data:
            for sample in teacher_data:
                for t in sample["trace"]:
                    if get_task_spec(t[0]).fingerprint() not in pred_signature_hash_to_ind:
                        raise ValueError(f"Unknown predictor fingerprint in trace: {get_task_spec(t[0]).fingerprint()}")


def build_rollout_batches(
    trace_data: TraceGrid,
    *,
    student: Module,
    pred_signature_hash_to_ind: dict[str, int],
    num_rollouts_per_grpo_step: int,
    adapter: dict[LM, Adapter] | Adapter,
    run: RunContext,
    format_failure_score: float,
    variably_invoked_predictor_grouping_mode: Literal["truncate", "fill", "ragged"],
    variably_invoked_predictor_fill_strategy: Literal["randint", "max"] | None,
    rng: random.Random,
) -> PredictorRolloutBatches:
    num_student_predictors = len(student.predictors())
    train_batch_per_predictor: PredictorRolloutBatches = [[] for _ in range(num_student_predictors)]

    for pred_id in range(num_student_predictors):
        for example_ind, example_data in enumerate(trace_data):
            predictor_example_invocations: list[list[tuple[Any, ...]]] = []
            for teacher_data in example_data:
                for sample in teacher_data:
                    if sample["example_ind"] != example_ind:
                        raise ValueError(
                            f"Example index {sample['example_ind']} does not match the expected index {example_ind}"
                        )
                    trace_instances_for_current_pred = [
                        (*t, sample["score"])
                        for t in sample["trace"]
                        if get_task_spec(t[0]).fingerprint()
                        == get_task_spec(student.predictors()[pred_id]).fingerprint()
                    ]
                    predictor_example_invocations.append(trace_instances_for_current_pred)
            if len(predictor_example_invocations) == 0:
                logger.warning(
                    f"Skipping example {example_ind} for predictor {pred_id} as it has no invocations. "
                    "This is likely due to all examples in the training set input, resulting in the model "
                    "generating output not following the dspy response format."
                )
                continue
            if len(predictor_example_invocations) != num_rollouts_per_grpo_step:
                logger.warning(
                    f"Number of predictor example invocations {len(predictor_example_invocations)} does not match "
                    f"the expected batch size {num_rollouts_per_grpo_step}. This is likely due to all examples in "
                    "the training set input, resulting in the model generating output not following the dspy "
                    "response format."
                )
            min_len = min(len(invocation) for invocation in predictor_example_invocations)
            max_len = max(len(invocation) for invocation in predictor_example_invocations)
            if min_len == 0:
                logger.warning(f"Skipping example {example_ind} for predictor {pred_id} as it has no invocations.")
                continue
            if variably_invoked_predictor_grouping_mode == "truncate":
                predictor_example_invocations = [invocation[:min_len] for invocation in predictor_example_invocations]
            elif variably_invoked_predictor_grouping_mode == "fill":
                if variably_invoked_predictor_fill_strategy == "randint":

                    def selector(options: list[tuple[Any, ...]]) -> tuple[Any, ...]:
                        return rng.choice(options)
                else:

                    def selector(options: list[tuple[Any, ...]]) -> tuple[Any, ...]:
                        return options[-1]

                predictor_example_invocations = [
                    invocation + [selector(invocation) for _ in range(max_len - len(invocation))]
                    for invocation in predictor_example_invocations
                ]
            elif variably_invoked_predictor_grouping_mode != "ragged":
                raise ValueError(
                    f"Unknown variably invoked predictor grouping mode {variably_invoked_predictor_grouping_mode}"
                )
            max_len = max(len(invocation) for invocation in predictor_example_invocations)
            example_training_data: list[GRPORolloutGroup] = [[] for _ in range(max_len)]
            for group_idx in range(max_len):
                for rollout_idx in range(len(predictor_example_invocations)):
                    trace_instance = predictor_example_invocations[rollout_idx][group_idx]
                    score = trace_instance[3]
                    trace_pred_id = pred_signature_hash_to_ind.get(get_task_spec(trace_instance[0]).fingerprint())
                    if trace_pred_id != pred_id:
                        raise ValueError(
                            f"Trace predictor id {trace_pred_id} does not match expected predictor {pred_id}"
                        )
                    predictor = trace_instance[0]
                    pred_lm = predictor.lm
                    if isinstance(adapter, dict):
                        adapter_candidate = cast("Adapter | None", adapter[pred_lm])
                    else:
                        adapter_candidate = adapter
                    resolved_adapter = require_adapter(
                        adapter_candidate if adapter_candidate is not None else run.adapter
                    )
                    if not resolved_adapter.capabilities.supports_finetune:
                        raise TypeError(
                            f"Adapter {resolved_adapter} does not support finetune data formatting. "
                            "GRPO training requires an adapter with capabilities.supports_finetune=True."
                        )
                    inp_messages = [
                        message_to_openai_chat(message)
                        for message in resolved_adapter.format(
                            task_spec=get_task_spec(trace_instance[0]), inputs=trace_instance[1], demos=[]
                        )
                    ]
                    if isinstance(trace_instance[2], FailedPrediction):
                        score = trace_instance[2].format_reward or format_failure_score
                        example_training_data[group_idx].append(
                            GRPOChatData(
                                messages=[FinetuneChatMessage.model_validate(m) for m in inp_messages],
                                completion=FinetuneAssistantMessage(content=trace_instance[2].completion_text),
                                reward=float(score),
                            )
                        )
                        logger.warning(
                            f"Adding a format failure example to the training data for predictor {pred_id} "
                            f"and example {example_ind}."
                        )
                    else:
                        all_messages = resolved_adapter.format_finetune_data(
                            task_spec=get_task_spec(trace_instance[0]),
                            inputs=trace_instance[1],
                            outputs=trace_instance[2],
                            demos=[],
                        )["messages"]
                        if all_messages[:-1] != inp_messages:
                            raise ValueError(
                                f"Input messages {inp_messages} do not match the expected messages {all_messages[:-1]}"
                            )
                        example_training_data[group_idx].append(
                            GRPOChatData(
                                messages=[FinetuneChatMessage.model_validate(m) for m in inp_messages],
                                completion=FinetuneAssistantMessage(content=all_messages[-1]["content"]),
                                reward=float(score),
                            )
                        )
            train_batch_per_predictor[pred_id].extend(example_training_data)
    return train_batch_per_predictor


def log_rollout_batch_warnings(
    train_batch_per_predictor: PredictorRolloutBatches,
    *,
    num_rollouts_per_grpo_step: int,
) -> None:
    for predictor_train_batch in train_batch_per_predictor:
        for grpo_train_group in predictor_train_batch:
            if len(grpo_train_group) != num_rollouts_per_grpo_step:
                logger.warning(
                    f"Number of completions {len(grpo_train_group)} does not match the expected number "
                    f"num_rollouts_per_grpo_step={num_rollouts_per_grpo_step}"
                )
                if len(grpo_train_group) > num_rollouts_per_grpo_step:
                    raise ValueError(
                        f"Number of completions {len(grpo_train_group)} is greater than the expected number "
                        f"num_rollouts_per_grpo_step={num_rollouts_per_grpo_step}"
                    )
            if len(set(map(repr, grpo_train_group))) < 2:
                logger.warning(
                    f"GRPOGroup has no diversity. This could be due to low temperature or a low number of "
                    f"rollouts. The GRPOGroup is {grpo_train_group}."
                )
