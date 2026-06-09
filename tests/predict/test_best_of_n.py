import asyncio

import pytest

from dspy.errors import AdapterParseError, SamplingExhaustedError
from dspy.predict.best_of_n import BestOfN
from dspy.predict.predict import Predict
from dspy.predict.sampling import get_sampling_metadata
from dspy.primitives import Module, Prediction
from tests.task_spec.helpers import ts
from tests.test_utils import DummyLM

_PARSE_TASK_SPEC = ts("question -> answer")


def _parse_error() -> AdapterParseError:
    return AdapterParseError(
        adapter_name="TestAdapter",
        task_spec=_PARSE_TASK_SPEC,
        lm_response="bad",
        message="parse failed",
    )


class DummyModule(Module):
    def __init__(self, task_spec, forward_fn):
        super().__init__()
        self.predictor = Predict(task_spec)
        self.forward_fn = forward_fn

    async def _aforward_impl(self, *, run, options=None, **inputs) -> Prediction:
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
    best_of_n = BestOfN(module=predict, num_samples=3, reward_fn=reward_fn, threshold=1.0)
    result = asyncio.run(best_of_n(question="What is the capital of Belgium?", run=run))
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
    best_of_n = BestOfN(module=predict, num_samples=3, reward_fn=lambda _, __: 1.0, threshold=0.0)
    with pytest.raises(ValueError, match=r"Deliberately failing"):
        asyncio.run(best_of_n(question="What is the capital of Belgium?", run=run))


def test_refine_module_custom_fail_count(make_run):
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    run = make_run(lm=lm)
    module_call_count = [0]

    async def raise_on_second_call(self, *, run, options=None, **inputs):
        if module_call_count[0] < 2:
            module_call_count[0] += 1
            raise _parse_error()
        return await self.predictor(run=run, options=options, **inputs)

    predict = DummyModule(ts("question -> answer"), raise_on_second_call)
    best_of_n = BestOfN(module=predict, num_samples=3, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=1)
    with pytest.raises(SamplingExhaustedError):
        asyncio.run(best_of_n(question="What is the capital of Belgium?", run=run))
    assert module_call_count[0] == 2, (
        "Module should have been called exactly 2 times, but was called %d times" % module_call_count[0]
    )


def test_best_of_n_all_attempts_fail_raises_sampling_exhausted(make_run):
    run = make_run(lm=DummyLM([{"answer": "Brussels"}]))

    async def always_raise(self, *, run, options=None, **inputs):
        raise _parse_error()

    predict = DummyModule(ts("question -> answer"), always_raise)
    best_of_n = BestOfN(module=predict, num_samples=3, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=10)
    with pytest.raises(SamplingExhaustedError) as exc_info:
        asyncio.run(best_of_n(question="What is the capital of Belgium?", run=run))
    assert exc_info.value.n_attempts == 3
    assert isinstance(exc_info.value.__cause__, AdapterParseError)


def test_best_of_n_instance_reuse_preserves_fail_count(make_run):
    run = make_run(lm=DummyLM([{"answer": "Brussels"}]))
    call_counts = []

    async def always_raise(self, *, run, options=None, **inputs):
        call_counts.append(1)
        raise _parse_error()

    predict = DummyModule(ts("question -> answer"), always_raise)
    best_of_n = BestOfN(module=predict, num_samples=3, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=10)
    for _ in range(2):
        call_counts.clear()
        with pytest.raises(SamplingExhaustedError):
            asyncio.run(best_of_n(question="What is the capital of Belgium?", run=run))
        assert len(call_counts) == 3


def test_best_of_n_exposes_threshold_miss_metadata(make_run):
    run = make_run(lm=DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}]))

    async def forward(self, *, run, options=None, **inputs):
        return await self.predictor(run=run, options=options, **inputs)

    predict = DummyModule(ts("question -> answer"), forward)
    best_of_n = BestOfN(
        module=predict,
        num_samples=2,
        reward_fn=lambda _inputs, pred: 0.5 if pred.answer == "City of Brussels" else 0.25,
        threshold=0.9,
    )
    result = asyncio.run(best_of_n(question="What is the capital of Belgium?", run=run))
    metadata = get_sampling_metadata(result)
    assert result.answer == "City of Brussels"
    assert metadata is not None
    assert metadata.threshold_met is False
    assert metadata.best_reward == 0.5
