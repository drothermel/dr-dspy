"""Expand opt-dec-format evaluations into concrete bottleneck workflow jobs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dspy.experiments.opt_dec_format._bottleneck_spec import workflow_job_spec

REQUIRED_METADATA_KEYS = frozenset(
    {
        "experiment_id",
        "optimizer_run_id",
        "candidate_id",
        "task_id",
        "split",
        "round_index",
        "seed_index",
        "metric_id",
        "candidate_surface",
        "evaluator_model_id",
    }
)


class DecoderOnlyWorkflowJobInput(BaseModel):
    """Inputs needed to evaluate one decoder prompt on one sample."""

    model_config = ConfigDict(extra="forbid")

    workflow_id: str
    decode_input_queue: str
    eval_input_queue: str
    result_queue: str | None = None
    evaluator_model_id: str
    rendered_decoder_prompt: str
    task_id: str
    decoder_input: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _metadata_is_complete(self) -> DecoderOnlyWorkflowJobInput:
        validate_required_metadata(self.metadata)
        if self.metadata["task_id"] != self.task_id:
            msg = "metadata['task_id'] must match task_id."
            raise ValueError(msg)
        if self.metadata["evaluator_model_id"] != self.evaluator_model_id:
            msg = "metadata['evaluator_model_id'] must match evaluator_model_id."
            raise ValueError(msg)
        return self


def validate_required_metadata(metadata: dict[str, Any]) -> None:
    """Validate required correlation metadata for expanded workflow jobs."""
    missing = sorted(REQUIRED_METADATA_KEYS - set(metadata))
    if missing:
        msg = f"Missing required workflow metadata keys: {missing}"
        raise ValueError(msg)


def expand_decoder_only_workflow_job(
    request: DecoderOnlyWorkflowJobInput,
) -> Any:
    """Build a decode -> eval workflow payload without entry-point config."""
    spec = workflow_job_spec()
    return spec.WorkflowJobPayload(
        workflow_id=request.workflow_id,
        steps=(
            spec.WorkflowStepSpec(
                name="decode",
                job_kind=spec.JobKind.LLM_QUERY_STATIC,
                input_queue=request.decode_input_queue,
                output_queue=request.eval_input_queue,
            ),
            spec.WorkflowStepSpec(
                name="eval",
                job_kind=spec.JobKind.EVAL_FROM_PREVIOUS,
                input_queue=request.eval_input_queue,
                output_queue=request.result_queue,
            ),
        ),
        step_configs={
            "decode": spec.LLMQueryStaticConfig(
                metadata=request.metadata,
                model_id=request.evaluator_model_id,
                prompt=request.rendered_decoder_prompt,
            ),
            "eval": spec.EvalFromPreviousConfig(
                metadata=request.metadata,
                suite="humaneval_plus",
                task_id=request.task_id,
                decoder_input=request.decoder_input,
            ),
        },
        metadata=request.metadata,
    )
