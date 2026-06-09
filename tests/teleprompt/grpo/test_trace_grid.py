from unittest.mock import AsyncMock

import pytest

from dspy.primitives import Example, Prediction
from dspy.runtime.optimization_trace import TraceData
from dspy.teleprompt.grpo import trace_grid as trace_grid_module
from dspy.teleprompt.grpo.trace_grid import collect_teacher_trace_grid
from dspy.testing import DummyLM


def _trace_entry(example_ind: int, *, score: float = 1.0) -> TraceData:
    return {
        "example_ind": example_ind,
        "example": Example.from_record({"q": "x"}, input_keys=("q",)),
        "prediction": Prediction(answer="y"),
        "trace": [],
        "score": score,
    }


@pytest.mark.asyncio
async def test_collect_teacher_trace_grid_remaps_example_indices(monkeypatch, make_run):
    teachers = [object(), object()]
    subsample = [
        Example.from_record({"q": "a"}, input_keys=("q",)),
        Example.from_record({"q": "b"}, input_keys=("q",)),
    ]

    async def fake_collect(*, program, dataset, **_kwargs):
        return [_trace_entry(example_ind) for example_ind, _example in enumerate(dataset)]

    monkeypatch.setattr(trace_grid_module, "collect_trace_data", fake_collect)

    grid = await collect_teacher_trace_grid(
        teachers=teachers,
        subsample=subsample,
        num_samples_per_input=2,
        run=make_run(lm=DummyLM([{}])),
        metric=None,
        max_concurrency=1,
        failure_score=0.0,
        format_failure_score=-1.0,
    )

    assert len(grid) == 2
    assert len(grid[0]) == 2
    assert len(grid[1]) == 2
    assert len(grid[0][0]) == 2
    assert len(grid[0][1]) == 2
    for example_ind in (0, 1):
        for teacher_ind in (0, 1):
            for sample in grid[example_ind][teacher_ind]:
                assert sample["example_ind"] == example_ind


@pytest.mark.asyncio
async def test_collect_teacher_trace_grid_calls_collect_per_teacher(monkeypatch, make_run):
    teachers = [object(), object()]
    subsample = [Example.from_record({"q": "a"}, input_keys=("q",))]
    mock_collect = AsyncMock(return_value=[_trace_entry(0)])
    monkeypatch.setattr(trace_grid_module, "collect_trace_data", mock_collect)

    await collect_teacher_trace_grid(
        teachers=teachers,
        subsample=subsample,
        num_samples_per_input=1,
        run=make_run(lm=DummyLM([{}])),
        metric=None,
        max_concurrency=2,
        failure_score=0.0,
        format_failure_score=-1.0,
    )

    assert mock_collect.await_count == 2
