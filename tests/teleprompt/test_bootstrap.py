import asyncio
from typing import Any, cast

import pytest

from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.teleprompt.bootstrap import BootstrapFewShot
from dspy.teleprompt.compile_params import BootstrapFewShotCompileParams
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts


def simple_metric(example, prediction, trace=None):
    return example.output == prediction.output


examples = [
    Example.from_record({"input": "What is the color of the sky?", "output": "blue"}, input_keys=("input",)),
    Example.from_record({"input": "What does the fox say?", "output": "Ring-ding-ding-ding-dingeringeding!"}),
]
trainset = [examples[0]]
valset = [examples[1]]


def test_bootstrap_initialization(make_run):
    bootstrap = BootstrapFewShot(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    assert bootstrap.metric == simple_metric, "Metric not correctly initialized"


class SimpleModule(Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    async def aforward(self, *, run, options=None, **inputs):
        return await self.predictor(run=run, options=options, **inputs)


def test_compile_with_predict_instances(make_run):
    student = SimpleModule(ts("input -> output"))
    teacher = SimpleModule(ts("input -> output"))
    lm = DummyLM(cast("Any", ["Initial thoughts", "Finish[blue]"]))
    run = make_run(lm=lm)
    bootstrap = BootstrapFewShot(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    compiled_student = asyncio.run(
        bootstrap.compile(student, params=BootstrapFewShotCompileParams(trainset=trainset, teacher=teacher), run=run)
    )
    assert compiled_student is not None, "Failed to compile student"
    assert hasattr(compiled_student, "_compiled") and compiled_student._compiled, "Student compilation flag not set"


def test_bootstrap_effectiveness(make_run):
    student = SimpleModule(ts("input -> output"))
    teacher = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "blue"}, {"output": "Ring-ding-ding-ding-dingeringeding!"}], follow_examples=True)
    run = make_run(lm=lm, optimization_trace=[])
    bootstrap = BootstrapFewShot(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    compiled_student = asyncio.run(
        bootstrap.compile(student, params=BootstrapFewShotCompileParams(trainset=trainset, teacher=teacher), run=run)
    )
    assert len(compiled_student.predictor.demos) == 1
    assert compiled_student.predictor.demos[0].input == trainset[0].input
    assert compiled_student.predictor.demos[0].output == trainset[0].output
    prediction = asyncio.run(compiled_student(input=trainset[0].input, run=run))
    assert prediction.output == trainset[0].output


def test_error_handling_during_bootstrap(make_run):

    class BuggyModule(Module):
        def __init__(self, signature):
            super().__init__()
            self.predictor = Predict(signature)

        async def aforward(self, *, run, options=None, **inputs):
            raise RuntimeError("Simulated error")

    student = SimpleModule(ts("input -> output"))
    teacher = BuggyModule(ts("input -> output"))
    lm = DummyLM([{"output": "Initial thoughts"}])
    run = make_run(lm=lm)
    bootstrap = BootstrapFewShot(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1, max_errors=1)
    with pytest.raises(RuntimeError, match="Simulated error"):
        asyncio.run(
            bootstrap.compile(
                student, params=BootstrapFewShotCompileParams(trainset=trainset, teacher=teacher), run=run
            )
        )


def test_validation_set_usage(make_run):
    student = SimpleModule(ts("input -> output"))
    teacher = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "Initial thoughts"}, {"output": "Finish[blue]"}])
    run = make_run(lm=lm)
    bootstrap = BootstrapFewShot(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    compiled_student = asyncio.run(
        bootstrap.compile(student, params=BootstrapFewShotCompileParams(trainset=trainset, teacher=teacher), run=run)
    )
    assert len(compiled_student.predictor.demos) >= len(valset), "Validation set not used in compiled student demos"
