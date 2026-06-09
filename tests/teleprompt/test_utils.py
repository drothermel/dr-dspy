import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.teleprompt.demo_sets import create_n_fewshot_demo_sets
from dspy.teleprompt.eval_batch import eval_candidate_program
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts


class DummyModule(Module):
    def __init__(self):
        super().__init__()

    async def aforward(self, *, run, options=None, **inputs):
        pass


def test_eval_candidate_program_full_trainset(make_run):
    trainset = [1, 2, 3, 4, 5]
    candidate_program = DummyModule()
    evaluate = AsyncMock(return_value=0)
    batch_size = 10
    run = make_run(lm=DummyLM([{}]))
    result = asyncio.run(
        eval_candidate_program(
            batch_size=batch_size,
            trainset=trainset,
            candidate_program=candidate_program,
            evaluate=evaluate,
            run=run,
        )
    )
    evaluate.assert_awaited_once()
    assert evaluate.await_args is not None
    _, called_kwargs = evaluate.await_args
    assert len(called_kwargs["devset"]) == len(trainset)
    assert called_kwargs["callback_metadata"] == {"metric_key": "eval_full"}
    assert result == 0


def test_eval_candidate_program_minibatch(make_run):
    trainset = [1, 2, 3, 4, 5]
    candidate_program = DummyModule()
    evaluate = AsyncMock(return_value=0)
    batch_size = 3
    run = make_run(lm=DummyLM([{}]))
    result = asyncio.run(
        eval_candidate_program(
            batch_size=batch_size,
            trainset=trainset,
            candidate_program=candidate_program,
            evaluate=evaluate,
            run=run,
        )
    )
    evaluate.assert_awaited_once()
    assert evaluate.await_args is not None
    _, called_kwargs = evaluate.await_args
    assert len(called_kwargs["devset"]) == batch_size
    assert called_kwargs["callback_metadata"] == {"metric_key": "eval_minibatch"}
    assert result == 0


def test_eval_candidate_program_failure(make_run):
    trainset = [1, 2, 3, 4, 5]
    candidate_program = DummyModule()
    evaluate = AsyncMock(side_effect=ValueError("Error"))
    batch_size = 3
    run = make_run(lm=DummyLM([{}]))
    result = asyncio.run(
        eval_candidate_program(
            batch_size=batch_size,
            trainset=trainset,
            candidate_program=candidate_program,
            evaluate=evaluate,
            run=run,
        )
    )
    assert result.score == 0.0


def test_create_n_fewshot_demo_sets_passes_metric_threshold_for_unshuffled(make_run):
    student = DummyModule()
    cast("Any", student).predictor = Predict(ts("input -> output"))
    trainset = [Example.from_record({"input": "test", "output": "test"}, input_keys=("input",))]
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    with patch("dspy.teleprompt.demo_sets.BootstrapFewShot") as MockBootstrap:
        mock_instance = Mock()
        mock_instance.compile = AsyncMock(return_value=student)
        MockBootstrap.return_value = mock_instance
        asyncio.run(
            create_n_fewshot_demo_sets(
                student=student,
                num_candidate_sets=4,
                trainset=trainset,
                max_labeled_demos=1,
                max_bootstrapped_demos=1,
                metric=lambda _ex, _pred, _trace=None: 1.0,
                run=run,
                metric_threshold=0.9,
            )
        )
        calls = MockBootstrap.call_args_list
        assert len(calls) >= 1, "BootstrapFewShot was never called"
        for call in calls:
            _, kwargs = call
            assert "metric_threshold" in kwargs, f"metric_threshold missing from BootstrapFewShot call: {kwargs}"
            assert kwargs["metric_threshold"] == 0.9, f"metric_threshold={kwargs['metric_threshold']}, expected 0.9"
