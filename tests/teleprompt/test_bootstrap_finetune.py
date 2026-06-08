import asyncio
from unittest.mock import patch

from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.teleprompt.bootstrap_finetune import BootstrapFinetune
from dspy.utils.dummies import DummyLM
from tests.task_spec.helpers import ts


def simple_metric(example, prediction, trace=None):
    return example.output == prediction.output


examples = [
    Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
    Example(input="What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!").with_inputs("input"),
]
trainset = [examples[0]]


def test_bootstrap_finetune_initialization(make_run):
    bootstrap = BootstrapFinetune(metric=simple_metric)
    assert bootstrap.metric == simple_metric, "Metric not correctly initialized"
    assert bootstrap.multitask, "Multitask should default to True"


class SimpleModule(Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    async def aforward(self, **kwargs: object):
        return await self.predictor(**kwargs)


def test_compile_with_predict_instances(make_run):
    student = SimpleModule(ts("input -> output"))
    teacher = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "blue"}, {"output": "Ring-ding-ding-ding-dingeringeding!"}])
    run = make_run(lm=lm)
    student.set_lm(lm)
    teacher.set_lm(lm)
    bootstrap = BootstrapFinetune(metric=simple_metric)
    with patch.object(bootstrap, "finetune_lms") as mock_finetune:
        mock_finetune.return_value = {(lm, None): lm}
        compiled_student = asyncio.run(bootstrap.compile(student, teacher=teacher, trainset=trainset, run=run))
        assert compiled_student is not None, "Failed to compile student"
        assert hasattr(compiled_student, "_compiled") and compiled_student._compiled, "Student compilation flag not set"
        mock_finetune.assert_called_once()


def test_error_handling_missing_lm(make_run):
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    student = SimpleModule(ts("input -> output"))
    bootstrap = BootstrapFinetune(metric=simple_metric)
    try:
        asyncio.run(bootstrap.compile(student, trainset=trainset, run=run))
        raise AssertionError("Should have raised ValueError for missing LM")
    except ValueError as e:
        assert "does not have an LM assigned" in str(e)
        assert "set_lm" in str(e)
