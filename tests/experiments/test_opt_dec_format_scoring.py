from __future__ import annotations

from dspy.experiments.opt_dec_format.scoring import (
    BoundedCompressionMetricConfig,
    MetricId,
    score_workflow_outputs,
)


def _output(
    *,
    parse_success: bool = True,
    test_pass_rate: float = 1.0,
    all_tests_passed: bool = True,
    failure_bucket: str | None = None,
) -> dict[str, object]:
    return {
        "metadata": {},
        "parse_success": parse_success,
        "test_pass_rate": test_pass_rate,
        "all_tests_passed": all_tests_passed,
        "selected_function_name": "candidate",
        "candidate_functions": [{"name": "candidate", "positional_arity": 2}],
        "expected_entry_point_present": False,
        "failure_bucket": failure_bucket,
    }


def test_score_workflow_outputs_uses_functional_recovery_rate() -> None:
    summary = score_workflow_outputs(
        [
            _output(test_pass_rate=1.0, all_tests_passed=True),
            _output(test_pass_rate=0.25, all_tests_passed=False, failure_bucket="failed_assertions"),
        ],
        metric_id=MetricId.TEST_PASS_RATE,
    )

    assert summary.score == 0.625
    assert summary.parse_rate == 1.0
    assert summary.all_tests_passed_rate == 0.5
    assert summary.failure_buckets == {"passed": 1, "failed_assertions": 1}


def test_bounded_compression_metric_stays_in_unit_interval() -> None:
    output = _output()
    output["metadata"] = {
        "encoder_output": "short description",
        "ground_truth_code": "def f(x):\n    return x\n" * 20,
    }

    summary = score_workflow_outputs(
        [output],
        metric_id=MetricId.PASS_RATE_WITH_BOUNDED_COMPRESSION_PENALTY,
        compression_config=BoundedCompressionMetricConfig(weight=0.1),
    )

    assert 0.9 <= summary.score <= 1.0
