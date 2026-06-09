import asyncio

import pytest

from dspy.predict.predict import Predict
from dspy.predict.refine import Refine
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts


class DummyModule(Module):
    def __init__(self, task_spec, forward_fn):
        super().__init__()
        self.predictor = Predict(task_spec)
        self.forward_fn = forward_fn

    async def aforward(self, *, run, options=None, **inputs) -> Prediction:
        return await self.forward_fn(self, run=run, options=options, **inputs)


def test_refine_forward_success_first_attempt(make_run):
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    run = make_run(lm=lm)
    module_call_count = [0]

    async def count_calls(self, *, run, options=None, **inputs):
        module_call_count[0] += 1
        return await self.predictor(run=run, options=options, **inputs)

    reward_call_count = [0]

    def reward_fn(kwargs, pred: Prediction) -> float:
        reward_call_count[0] += 1
        return 1.0 if len(pred.answer) == 1 else 0.0

    predict = DummyModule(ts("question -> answer"), count_calls)
    refine = Refine(module=predict, N=3, reward_fn=reward_fn, threshold=1.0)
    result = asyncio.run(refine(question="What is the capital of Belgium?", run=run))
    assert result.answer == "Brussels", "Result should be `Brussels`"
    assert reward_call_count[0] > 0, "Reward function should have been called"
    assert module_call_count[0] == 3, (
        "Module should have been called exactly 3 times, but was called %d times" % module_call_count[0]
    )


def test_refine_module_default_fail_count(make_run):
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    run = make_run(lm=lm)

    async def always_raise(self, *, run, options=None, **inputs):
        raise ValueError("Deliberately failing")

    predict = DummyModule(ts("question -> answer"), always_raise)
    refine = Refine(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0)
    with pytest.raises(ValueError, match=r"Deliberately failing"):
        asyncio.run(refine(question="What is the capital of Belgium?", run=run))


def test_refine_module_custom_fail_count(make_run):
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    run = make_run(lm=lm)
    module_call_count = [0]

    async def raise_on_second_call(self, *, run, options=None, **inputs):
        if module_call_count[0] < 2:
            module_call_count[0] += 1
            raise ValueError("Deliberately failing")
        return await self.predictor(run=run, options=options, **inputs)

    predict = DummyModule(ts("question -> answer"), raise_on_second_call)
    refine = Refine(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=1)
    with pytest.raises(ValueError, match=r"Deliberately failing"):
        asyncio.run(refine(question="What is the capital of Belgium?", run=run))
    assert module_call_count[0] == 2, (
        "Module should have been called exactly 2 times, but was called %d times" % module_call_count[0]
    )
