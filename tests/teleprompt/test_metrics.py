import asyncio

import pytest

from dspy.evaluate.metric_invoke import invoke_metric, normalize_metric_score
from dspy.primitives import Example, Prediction
from dspy.testing import DummyLM


def test_invoke_metric_async_function(make_run):
    run = make_run(lm=DummyLM([]))
    example = Example.from_record({"question": "q", "answer": "a"}, input_keys=("question",))
    pred = Prediction.from_record({"answer": "a"})

    async def async_metric(example, prediction, trace=None):
        del trace
        await asyncio.sleep(0)
        return example.answer == prediction.answer

    score = asyncio.run(
        invoke_metric(
            async_metric,
            example=example,
            prediction=pred,
            trace=None,
            run=run,
        )
    )
    assert score == 1.0


@pytest.mark.parametrize("trace", [None, [{"step": 1}]])
def test_invoke_metric_sync_with_optional_trace(make_run, trace):
    run = make_run(lm=DummyLM([]))
    example = Example.from_record({"question": "q"}, input_keys=("question",))
    pred = Prediction.from_record({"answer": "a"})
    seen_traces = []

    def record_trace(example, prediction, trace=None):
        del example, prediction
        seen_traces.append(trace)
        return 0.5

    score = asyncio.run(
        invoke_metric(
            record_trace,
            example=example,
            prediction=pred,
            trace=trace,
            run=run,
        )
    )
    assert score == 0.5
    assert seen_traces == [trace]


def test_normalize_metric_score_rejects_out_of_range_for_teleprompt():
    with pytest.raises(ValueError, match="Metric score must be in \\[0, 1\\]"):
        normalize_metric_score(1.5)
