import asyncio

from dspy.evaluate.auto_evaluation import CompleteAndGrounded, SemanticF1
from dspy.primitives import Example, Prediction
from dspy.testing import DummyLM


def test_semantic_f1_returns_prediction_without_trace(make_run):
    run = make_run(lm=DummyLM([{"reasoning": "Comparing the responses", "precision": 1.0, "recall": 1.0}]))
    example = Example.from_record({"question": "What is 1+1?", "response": "2"})
    pred = Prediction.from_record({"response": "2"})
    metric = SemanticF1()
    result = asyncio.run(metric.aforward(example, pred, run=run))
    assert isinstance(result, Prediction)
    assert hasattr(result, "score")
    assert isinstance(result.score, (int, float, bool))


def test_semantic_f1_returns_prediction_with_trace(make_run):
    run = make_run(lm=DummyLM([{"reasoning": "Comparing the responses", "precision": 1.0, "recall": 1.0}]))
    example = Example.from_record({"question": "What is 1+1?", "response": "2"})
    pred = Prediction.from_record({"response": "2"})
    metric = SemanticF1(threshold=0.5)
    result = asyncio.run(metric.aforward(example, pred, trace=True, run=run))
    assert isinstance(result, Prediction)
    assert hasattr(result, "score")
    assert isinstance(result.score, bool)


def test_semantic_f1_score_value(make_run):
    run = make_run(lm=DummyLM([{"reasoning": "Comparing the responses", "precision": 0.8, "recall": 0.6}]))
    example = Example.from_record({"question": "test", "response": "answer"})
    pred = Prediction.from_record({"response": "response"})
    metric = SemanticF1()
    result = asyncio.run(metric.aforward(example, pred, run=run))
    expected_f1 = 2 * (0.8 * 0.6) / (0.8 + 0.6)
    assert isinstance(result, Prediction)
    assert abs(result.score - expected_f1) < 0.001


def test_complete_and_grounded_returns_prediction_without_trace(make_run):
    run = make_run(
        lm=DummyLM(
            [
                {
                    "reasoning": "Analyzing completeness",
                    "ground_truth_key_ideas": "the answer is 2",
                    "system_response_key_ideas": "the answer is 2",
                    "discussion": "both match",
                    "completeness": 1.0,
                },
                {
                    "reasoning": "Analyzing groundedness",
                    "system_response_claims": "1+1=2",
                    "discussion": "supported by context",
                    "groundedness": 1.0,
                },
            ]
        )
    )
    example = Example.from_record({"question": "What is 1+1?", "response": "2"})
    pred = Prediction.from_record({"response": "2", "context": "context"})
    metric = CompleteAndGrounded()
    result = asyncio.run(metric.aforward(example, pred, run=run))
    assert isinstance(result, Prediction)
    assert hasattr(result, "score")
    assert isinstance(result.score, (int, float, bool))


def test_complete_and_grounded_returns_prediction_with_trace(make_run):
    run = make_run(
        lm=DummyLM(
            [
                {
                    "reasoning": "Analyzing completeness",
                    "ground_truth_key_ideas": "the answer is 2",
                    "system_response_key_ideas": "the answer is 2",
                    "discussion": "both match",
                    "completeness": 0.9,
                },
                {
                    "reasoning": "Analyzing groundedness",
                    "system_response_claims": "1+1=2",
                    "discussion": "supported by context",
                    "groundedness": 0.8,
                },
            ]
        )
    )
    example = Example.from_record({"question": "What is 1+1?", "response": "2"})
    pred = Prediction.from_record({"response": "2", "context": "context"})
    metric = CompleteAndGrounded(threshold=0.7)
    result = asyncio.run(metric.aforward(example, pred, trace=True, run=run))
    assert isinstance(result, Prediction)
    assert hasattr(result, "score")
    assert isinstance(result.score, bool)


def test_semantic_f1_prediction_can_be_compared(make_run):
    run = make_run(
        lm=DummyLM(
            [
                {"reasoning": "Comparing first response", "precision": 0.8, "recall": 0.6},
                {"reasoning": "Comparing second response", "precision": 0.9, "recall": 0.7},
            ]
        )
    )
    metric = SemanticF1()
    example1 = Example.from_record({"question": "test1", "response": "answer1"})
    pred1 = Prediction.from_record({"response": "response1"})
    result1 = asyncio.run(metric.aforward(example1, pred1, run=run))
    example2 = Example.from_record({"question": "test2", "response": "answer2"})
    pred2 = Prediction.from_record({"response": "response2"})
    result2 = asyncio.run(metric.aforward(example2, pred2, run=run))
    assert isinstance(result1, Prediction)
    assert isinstance(result2, Prediction)
    assert result2.score > result1.score
