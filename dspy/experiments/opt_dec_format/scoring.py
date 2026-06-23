"""Optimizer-facing scoring for opt-dec-format workflow outputs."""

from __future__ import annotations

from enum import StrEnum
from importlib import import_module
from statistics import mean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dspy.experiments.opt_dec_format._bottleneck_spec import workflow_job_spec


class MetricId(StrEnum):
    """First-pass optimizer metric ids."""

    PARSE_BINARY = "parse_binary"
    TEST_PASS_BINARY = "test_pass_binary"  # noqa: S105
    TEST_PASS_RATE = "test_pass_rate"  # noqa: S105
    COMPRESSION_RATIO_VS_GROUND_TRUTH = "compression_ratio_vs_ground_truth"
    PASS_RATE_WITH_BOUNDED_COMPRESSION_PENALTY = "pass_rate_with_bounded_compression_penalty"  # noqa: S105


class BoundedCompressionMetricConfig(BaseModel):
    """Config for bounded correctness/compression reward."""

    model_config = ConfigDict(extra="forbid")

    metric_id: MetricId = MetricId.PASS_RATE_WITH_BOUNDED_COMPRESSION_PENALTY
    weight: float = Field(default=0.10, ge=0.0, le=1.0)
    min_compression_ratio: float = 0.01
    max_compression_ratio: float = 4.0


class WorkflowScoreSummary(BaseModel):
    """Aggregate score and diagnostics for a workflow output batch."""

    model_config = ConfigDict(extra="forbid")

    metric_id: MetricId
    score: float
    output_count: int
    parse_rate: float
    all_tests_passed_rate: float
    mean_test_pass_rate: float
    failure_buckets: dict[str, int]


def score_workflow_outputs(
    outputs: list[Any | dict[str, Any]],
    *,
    metric_id: MetricId | str,
    compression_config: BoundedCompressionMetricConfig | None = None,
) -> WorkflowScoreSummary:
    """Aggregate eval outputs into an optimizer-facing score."""
    parsed = [_as_eval_output(output) for output in outputs]
    resolved_metric = MetricId(metric_id)
    if not parsed:
        return WorkflowScoreSummary(
            metric_id=resolved_metric,
            score=0.0,
            output_count=0,
            parse_rate=0.0,
            all_tests_passed_rate=0.0,
            mean_test_pass_rate=0.0,
            failure_buckets={},
        )

    parse_rate = mean(1.0 if output.parse_success else 0.0 for output in parsed)
    pass_rate = mean(1.0 if output.all_tests_passed else 0.0 for output in parsed)
    mean_test_pass_rate = mean(output.test_pass_rate for output in parsed)
    failure_buckets = _failure_buckets(parsed)
    score = mean(
        _score_one(
            output,
            metric_id=resolved_metric,
            compression_config=compression_config,
        )
        for output in parsed
    )
    return WorkflowScoreSummary(
        metric_id=resolved_metric,
        score=score,
        output_count=len(parsed),
        parse_rate=parse_rate,
        all_tests_passed_rate=pass_rate,
        mean_test_pass_rate=mean_test_pass_rate,
        failure_buckets=failure_buckets,
    )


def _score_one(
    output: Any,
    *,
    metric_id: MetricId,
    compression_config: BoundedCompressionMetricConfig | None,
) -> float:
    if metric_id == MetricId.PARSE_BINARY:
        return 1.0 if output.parse_success else 0.0
    if metric_id == MetricId.TEST_PASS_BINARY:
        return 1.0 if output.all_tests_passed else 0.0
    if metric_id == MetricId.TEST_PASS_RATE:
        return output.test_pass_rate
    if metric_id == MetricId.COMPRESSION_RATIO_VS_GROUND_TRUTH:
        return _compression_ratio(output)
    if metric_id == MetricId.PASS_RATE_WITH_BOUNDED_COMPRESSION_PENALTY:
        config = compression_config or BoundedCompressionMetricConfig()
        return _bounded_compression_score(output, config=config)
    msg = f"Unsupported metric id: {metric_id}"
    raise ValueError(msg)


def _bounded_compression_score(
    output: Any,
    *,
    config: BoundedCompressionMetricConfig,
) -> float:
    ratio = _compression_ratio(output)
    clamped = min(
        max(ratio, config.min_compression_ratio),
        config.max_compression_ratio,
    )
    compression_score = (config.max_compression_ratio - clamped) / (
        config.max_compression_ratio - config.min_compression_ratio
    )
    return output.test_pass_rate * ((1.0 - config.weight) + config.weight * compression_score)


def _compression_ratio(output: Any) -> float:
    encoder_output = output.metadata.get("encoder_output")
    ground_truth_code = output.metadata.get("ground_truth_code")
    if not isinstance(encoder_output, str) or not isinstance(ground_truth_code, str):
        return 1.0
    decoder_input_compression = _decoder_input_compression()
    _raw, compressed = decoder_input_compression(encoder_output)
    _gt_raw, gt_compressed = decoder_input_compression(ground_truth_code)
    if gt_compressed == 0:
        return 1.0
    return compressed / gt_compressed


def _failure_buckets(outputs: list[Any]) -> dict[str, int]:
    buckets: dict[str, int] = {}
    for output in outputs:
        key = output.failure_bucket or "passed"
        buckets[key] = buckets.get(key, 0) + 1
    return buckets


def _as_eval_output(
    output: Any | dict[str, Any],
) -> Any:
    eval_output_type = workflow_job_spec().EvalFromPreviousOutput
    if isinstance(output, eval_output_type):
        return output
    return eval_output_type.model_validate(output)


def _decoder_input_compression() -> Any:
    try:
        return import_module("dr_code.analysis.compress").decoder_input_compression
    except ModuleNotFoundError:
        import sys
        from pathlib import Path

        sibling_src = (Path(__file__).resolve().parents[3] / "../dr-code/src").resolve()
        if sibling_src.exists():
            sys.path.insert(0, str(sibling_src))
            return import_module("dr_code.analysis.compress").decoder_input_compression
        raise
