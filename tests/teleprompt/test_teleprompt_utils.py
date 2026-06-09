import asyncio
from unittest.mock import AsyncMock

import pytest

from dspy.predict.predict import Predict  # noqa: F401 — initialize predict before primitives lazy import
from dspy.primitives import Module
from dspy.teleprompt.eval_batch import eval_candidate_program
from dspy.testing import DummyLM


class DummyModule(Module):
    def __init__(self):
        super().__init__()

    async def _aforward_impl(self, *, run, options=None, **inputs):
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
    with pytest.raises(ValueError, match="Error"):
        asyncio.run(
            eval_candidate_program(
                batch_size=batch_size,
                trainset=trainset,
                candidate_program=candidate_program,
                evaluate=evaluate,
                run=run,
            )
        )
