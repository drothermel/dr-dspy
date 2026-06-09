from __future__ import annotations

import asyncio

import pytest

from dspy.errors import AdapterParseError, SamplingExhaustedError
from dspy.predict.sampling import SamplingAttempt, sample_with_reward
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


class _DummyModule(Module):
    async def _aforward_impl(self, *, run, options=None, **inputs):
        return Prediction.from_record({"answer": "ok"})


@pytest.mark.asyncio
async def test_sample_with_reward_failure_budget_counts_failures_not_attempt_index(make_run):
    call_count = 0

    async def execute_attempt(_attempt: SamplingAttempt) -> tuple[Prediction, list]:
        nonlocal call_count
        call_count += 1
        if call_count in {1, 3}:
            raise _parse_error()
        return Prediction.from_record({"answer": "ok"}), []

    run = make_run(lm=DummyLM([{}]))
    result = await sample_with_reward(
        module=_DummyModule(),
        num_samples=4,
        fail_count=2,
        reward_fn=lambda _inputs, _outputs: 1.0,
        run=run,
        options=None,
        inputs={"question": "q"},
        execute_attempt=execute_attempt,
    )
    assert result["answer"] == "ok"
    assert call_count == 4


@pytest.mark.asyncio
async def test_sample_with_reward_exhausts_on_too_many_failures(make_run):
    async def execute_attempt(_attempt: SamplingAttempt) -> tuple[Prediction, list]:
        raise _parse_error()

    run = make_run(lm=DummyLM([{}]))
    with pytest.raises(SamplingExhaustedError) as exc_info:
        await sample_with_reward(
            module=_DummyModule(),
            num_samples=5,
            fail_count=2,
            reward_fn=lambda _inputs, _outputs: 1.0,
            run=run,
            options=None,
            inputs={"question": "q"},
            execute_attempt=execute_attempt,
        )
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, AdapterParseError)


@pytest.mark.asyncio
async def test_sample_with_reward_propagates_cancellation(make_run):
    async def execute_attempt(_attempt: SamplingAttempt) -> tuple[Prediction, list]:
        raise asyncio.CancelledError

    run = make_run(lm=DummyLM([{}]))
    with pytest.raises(asyncio.CancelledError):
        await sample_with_reward(
            module=_DummyModule(),
            num_samples=3,
            fail_count=2,
            reward_fn=lambda _inputs, _outputs: 1.0,
            run=run,
            options=None,
            inputs={"question": "q"},
            execute_attempt=execute_attempt,
        )


@pytest.mark.asyncio
async def test_sample_with_reward_does_not_retry_value_error(make_run):
    async def execute_attempt(_attempt: SamplingAttempt) -> tuple[Prediction, list]:
        raise ValueError("not transient")

    run = make_run(lm=DummyLM([{}]))
    with pytest.raises(ValueError, match="not transient"):
        await sample_with_reward(
            module=_DummyModule(),
            num_samples=3,
            fail_count=2,
            reward_fn=lambda _inputs, _outputs: 1.0,
            run=run,
            options=None,
            inputs={"question": "q"},
            execute_attempt=execute_attempt,
        )
