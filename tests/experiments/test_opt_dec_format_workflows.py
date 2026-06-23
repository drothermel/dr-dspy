from __future__ import annotations

import pytest

from dspy.experiments.opt_dec_format.workflow_jobs import (
    DecoderOnlyWorkflowJobInput,
    expand_decoder_only_workflow_job,
)


def _metadata() -> dict[str, object]:
    return {
        "experiment_id": "exp",
        "optimizer_run_id": "run",
        "candidate_id": "cand",
        "task_id": "HumanEval/0",
        "split": "train",
        "round_index": 0,
        "seed_index": 0,
        "metric_id": "test_pass_rate",
        "candidate_surface": "bounded_slots",
        "evaluator_model_id": ("openrouter.openai__gpt-oss-20b.reasoning-low.temp-0p7.top-p-0p95.v0"),
    }


def test_decoder_only_workflow_expansion_uses_public_bottleneck_schema() -> None:
    model_id = str(_metadata()["evaluator_model_id"])
    payload = expand_decoder_only_workflow_job(
        DecoderOnlyWorkflowJobInput(
            workflow_id="wf",
            decode_input_queue="decode-in",
            eval_input_queue="eval-in",
            result_queue="results",
            evaluator_model_id=model_id,
            rendered_decoder_prompt="Write code",
            task_id="HumanEval/0",
            decoder_input="description",
            metadata=_metadata(),
        )
    )

    assert payload.steps[0].name == "decode"
    assert payload.steps[0].job_kind.value == "llm_query_static"
    assert payload.steps[1].job_kind.value == "eval_from_previous"
    assert payload.step_configs["eval"].task_id == "HumanEval/0"
    assert "entry_point" not in payload.step_configs["eval"].model_dump()


def test_decoder_only_workflow_requires_correlation_metadata() -> None:
    metadata = _metadata()
    del metadata["metric_id"]

    with pytest.raises(ValueError, match="Missing required workflow metadata"):
        DecoderOnlyWorkflowJobInput(
            workflow_id="wf",
            decode_input_queue="decode-in",
            eval_input_queue="eval-in",
            evaluator_model_id=str(metadata["evaluator_model_id"]),
            rendered_decoder_prompt="Write code",
            task_id="HumanEval/0",
            decoder_input="description",
            metadata=metadata,
        )
