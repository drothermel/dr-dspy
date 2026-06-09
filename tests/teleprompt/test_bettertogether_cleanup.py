from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from dspy.predict.predict import Predict
from dspy.primitives import Example, Module
from dspy.teleprompt.bettertogether import BetterTogether
from dspy.teleprompt.compile_params import BetterTogetherCompileParams
from tests.task_spec.helpers import ts
from tests.test_utils import DummyLM


class _SimpleModule(Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = Predict(ts("input -> output"))

    async def _aforward_impl(self, *, run, options=None, **inputs):
        return await self.predictor(run=run, options=options, **inputs)


@pytest.mark.asyncio
async def test_bettertogether_kills_lms_when_baseline_evaluation_fails(make_run):
    student = _SimpleModule()
    student.set_lm(DummyLM([{"output": "ok"}]))
    valset = [Example.from_record({"input": "q", "output": "a"}, input_keys=("input",))]
    optimizer = BetterTogether(metric=lambda *_args, **_kwargs: 1.0)

    with (
        patch("dspy.teleprompt.bettertogether.launch_lms") as mock_launch,
        patch("dspy.teleprompt.bettertogether.kill_lms") as mock_kill,
        patch(
            "dspy.teleprompt.bettertogether.BetterTogether._evaluate_on_valset",
            new_callable=AsyncMock,
            side_effect=RuntimeError("baseline failed"),
        ),
    ):
        params = BetterTogetherCompileParams(trainset=valset, valset=valset, strategy=["p"])
        with pytest.raises(RuntimeError, match="baseline failed"):
            await optimizer.compile(
                student,
                params=params,
                run=make_run(lm=DummyLM([{"output": "ok"}])),
            )

    mock_launch.assert_called_once_with(student)
    mock_kill.assert_called_once_with(student)
