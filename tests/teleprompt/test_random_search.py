import asyncio
from typing import Any, cast

from dspy.predict.predict import Predict
from dspy.primitives import Example, Module
from dspy.teleprompt.compile_params import RandomSearchCompileParams
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts


class SimpleModule(Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    async def _aforward_impl(self, *, run, options=None, **inputs):
        return await self.predictor(run=run, options=options, **inputs)


def simple_metric(example, prediction, trace=None):
    return example.output == prediction.output


def test_basic_workflow(make_run):
    student = SimpleModule(ts("input -> output"))
    teacher = SimpleModule(ts("input -> output"))
    lm = DummyLM(cast("Any", ["Initial thoughts", "Finish[blue]"]))
    run = make_run(lm=lm)
    optimizer = BootstrapFewShotWithRandomSearch(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    trainset = [
        Example.from_record({"input": "What is the color of the sky?", "output": "blue"}, input_keys=("input",)),
        Example.from_record(
            {"input": "What does the fox say?", "output": "Ring-ding-ding-ding-dingeringeding!"}, input_keys=("input",)
        ),
    ]
    asyncio.run(
        optimizer.compile(student, params=RandomSearchCompileParams(trainset=trainset, teacher=teacher), run=run)
    )
