import asyncio

import pytest

from dspy.evaluate.auto_evaluation import SemanticF1
from dspy.evaluate.metric_invoke import call_metric, invoke_metric, normalize_metric_score
from dspy.primitives import Example, Prediction
from dspy.testing import DummyLM


def test_normalize_metric_score_bool():
    assert normalize_metric_score(True) == 1.0
    assert normalize_metric_score(False) == 0.0


def test_normalize_metric_score_number():
    assert normalize_metric_score(0.75) == 0.75


def test_normalize_metric_score_prediction():
    assert normalize_metric_score(Prediction(score=0.25)) == 0.25


def test_normalize_metric_score_missing_prediction_score_raises():
    with pytest.raises(ValueError, match="must contain a `score` field"):
        normalize_metric_score(Prediction(answer="x"))


def test_normalize_metric_score_unsupported_type_raises():
    with pytest.raises(TypeError, match="unsupported type"):
        normalize_metric_score("bad")


def test_call_metric_module_passes_use_threshold(make_run):
    run = make_run(
        lm=DummyLM(
            [
                {"reasoning": "Comparing", "precision": 1.0, "recall": 1.0},
                {"reasoning": "Comparing", "precision": 1.0, "recall": 1.0},
            ]
        )
    )
    example = Example.from_record({"question": "q", "response": "a"})
    pred = Prediction.from_record({"response": "a"})
    metric = SemanticF1(threshold=0.5)

    with_trace = asyncio.run(
        call_metric(
            metric,
            example=example,
            prediction=pred,
            trace=[{"step": 1}],
            run=run,
        )
    )
    without_trace = asyncio.run(
        call_metric(
            metric,
            example=example,
            prediction=pred,
            trace=None,
            run=run,
        )
    )

    assert isinstance(with_trace.score, bool)
    assert isinstance(without_trace.score, float)


def test_invoke_metric_normalizes_module_threshold(make_run):
    run = make_run(
        lm=DummyLM(
            [
                {"reasoning": "Comparing", "precision": 1.0, "recall": 1.0},
                {"reasoning": "Comparing", "precision": 1.0, "recall": 1.0},
            ]
        )
    )
    example = Example.from_record({"question": "q", "response": "a"})
    pred = Prediction.from_record({"response": "a"})
    metric = SemanticF1(threshold=0.5)

    score = asyncio.run(
        invoke_metric(
            metric,
            example=example,
            prediction=pred,
            trace=[{"step": 1}],
            run=run,
        )
    )
    assert score == 1.0


def test_invoke_metric_sync_function(make_run):
    run = make_run(lm=DummyLM([]))
    example = Example.from_record({"question": "q", "answer": "a"}, input_keys=("question",))
    pred = Prediction.from_record({"answer": "a"})

    def exact(example, prediction, trace=None):
        del trace
        return example.answer == prediction.answer

    score = asyncio.run(
        invoke_metric(
            exact,
            example=example,
            prediction=pred,
            trace=None,
            run=run,
        )
    )
    assert score == 1.0
