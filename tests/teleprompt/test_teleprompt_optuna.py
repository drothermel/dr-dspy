import asyncio

import pytest

from dspy.predict.predict import Predict
from dspy.primitives import Example, Module
from dspy.teleprompt.compile_params import BootstrapOptunaCompileParams
from dspy.teleprompt.teleprompt_optuna import BootstrapFewShotWithOptuna
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts


def simple_metric(example, prediction, trace=None):
    return example.output == prediction.output


class SimpleModule(Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    async def _aforward_impl(self, *, run, options=None, **inputs):
        return await self.predictor(run=run, options=options, **inputs)


trainset = [
    Example.from_record({"input": "a", "output": "1"}, input_keys=("input",)),
]


def test_optuna_rejects_list_teacher(make_run):
    student = SimpleModule(ts("input -> output"))
    teacher_a = SimpleModule(ts("input -> output"))
    teacher_b = SimpleModule(ts("input -> output"))
    run = make_run(lm=DummyLM([{"output": "1"}]))
    optimizer = BootstrapFewShotWithOptuna(metric=simple_metric, num_random_candidates=1)
    with pytest.raises(ValueError, match="single teacher Module"):
        asyncio.run(
            optimizer.compile(
                student,
                params=BootstrapOptunaCompileParams(trainset=trainset, teacher=[teacher_a, teacher_b], max_demos=1),
                run=run,
            )
        )
