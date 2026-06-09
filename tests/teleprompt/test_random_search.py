import asyncio
from typing import Any, cast

from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch
from dspy.utils.dummies import DummyLM
from tests.task_spec.helpers import ts


class SimpleModule(Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    async def aforward(self, **kwargs: object):
        return await self.predictor(**kwargs)


def simple_metric(example, prediction, trace=None):
    return example.output == prediction.output


def test_basic_workflow(make_run):
    student = SimpleModule(ts("input -> output"))
    teacher = SimpleModule(ts("input -> output"))
    lm = DummyLM(cast("Any", ["Initial thoughts", "Finish[blue]"]))
    run = make_run(lm=lm)
    optimizer = BootstrapFewShotWithRandomSearch(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    trainset = [
        Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
        Example(input="What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!").with_inputs("input"),
    ]
    asyncio.run(optimizer.compile(student, teacher=teacher, trainset=trainset, run=run))
