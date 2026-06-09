from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from dspy.predict.predict import Predict
from dspy.primitives import Example, Module
from dspy.task_spec.predictor_context import get_task_spec
from dspy.teleprompt.avatar_optimizer import AvatarOptimizer
from dspy.teleprompt.compile_params import AvatarOptimizerCompileParams
from tests.task_spec.helpers import ts
from tests.test_utils import DummyLM


class _ActorModule(Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = Predict(ts("question -> answer"))
        self.tools: list[object] = []


@pytest.mark.asyncio
async def test_avatar_optimizer_rejects_worse_candidate_instruction(make_run):
    student = _ActorModule()
    student.set_lm(DummyLM([{"answer": "ok"}]))
    trainset = [Example.from_record({"question": "q", "answer": "a"}, input_keys=("question",))]
    optimizer = AvatarOptimizer(metric=lambda *_args, **_kwargs: 1.0, max_iters=1)
    optimizer.comparator = AsyncMock(return_value=type("Feedback", (), {"feedback": "be concise"})())
    optimizer.feedback_instruction = AsyncMock(
        return_value=type("Instruction", (), {"new_instruction": "bad instruction"})()
    )

    scores = iter([0.9, 0.1])

    async def fake_get_pos_neg_results(_actor, _trainset, *, run):
        return next(scores), [], []

    with patch.object(AvatarOptimizer, "_get_pos_neg_results", side_effect=fake_get_pos_neg_results):
        result = await optimizer.compile(
            student,
            params=AvatarOptimizerCompileParams(trainset=trainset),
            run=make_run(lm=DummyLM([{"answer": "ok"}])),
        )

    assert get_task_spec(result.program.actor).instructions != "bad instruction"


@pytest.mark.asyncio
async def test_avatar_optimizer_accepts_better_candidate_instruction(make_run):
    student = _ActorModule()
    student.set_lm(DummyLM([{"answer": "ok"}]))
    trainset = [Example.from_record({"question": "q", "answer": "a"}, input_keys=("question",))]
    optimizer = AvatarOptimizer(metric=lambda *_args, **_kwargs: 1.0, max_iters=1)
    optimizer.comparator = AsyncMock(return_value=type("Feedback", (), {"feedback": "be concise"})())
    optimizer.feedback_instruction = AsyncMock(
        return_value=type("Instruction", (), {"new_instruction": "good instruction"})()
    )

    scores = iter([0.1, 0.9])

    async def fake_get_pos_neg_results(_actor, _trainset, *, run):
        return next(scores), [], []

    with patch.object(AvatarOptimizer, "_get_pos_neg_results", side_effect=fake_get_pos_neg_results):
        result = await optimizer.compile(
            student,
            params=AvatarOptimizerCompileParams(trainset=trainset),
            run=make_run(lm=DummyLM([{"answer": "ok"}])),
        )

    assert get_task_spec(result.program.actor).instructions == "good instruction"
