import pytest

from dspy.predict.predict import Predict
from dspy.primitives import Module
from dspy.teleprompt.bootstrap import BootstrapFewShot
from tests.task_spec.helpers import ts


class OnePredictorModule(Module):
    def __init__(self):
        super().__init__()
        self.predictor = Predict(ts("input -> output"))

    async def _aforward_impl(self, *, run, options=None, **inputs):
        return await self.predictor(run=run, options=options, **inputs)


class TwoPredictorModule(Module):
    def __init__(self):
        super().__init__()
        self.predictor_a = Predict(ts("input -> output"))
        self.predictor_b = Predict(ts("input -> output"))

    async def _aforward_impl(self, *, run, options=None, **inputs):
        return await self.predictor_a(run=run, options=options, **inputs)


def _always_true_metric(example, prediction, trace=None):
    del example, prediction, trace
    return True


def test_bootstrap_predictor_mapping_raises_on_structural_mismatch():
    bootstrap = BootstrapFewShot(metric=_always_true_metric)
    bootstrap.student = OnePredictorModule()
    bootstrap.teacher = TwoPredictorModule()
    with pytest.raises(AssertionError, match="same number of predictors"):
        bootstrap._prepare_predictor_mappings()
