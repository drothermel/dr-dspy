from typing import Any, cast

import pytest

from dspy.errors import ContextWindowExceededError
from dspy.predict.predict import Predict
from dspy.primitives import Example, Module
from dspy.teleprompt.infer_rules import InferRules
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts


class SimpleModule(Module):
    def __init__(self):
        super().__init__()
        self.predictor = Predict(ts("input -> output"))

    async def _aforward_impl(self, *, run, options=None, **inputs):
        return await self.predictor(run=run, options=options, **inputs)


def _unit_metric(example, prediction, trace=None):
    del example, prediction, trace
    return 1.0


@pytest.mark.asyncio
async def test_induce_rules_shrinks_demos_on_context_window_error(make_run):
    infer = InferRules(metric=_unit_metric, num_rules=1)
    calls = {"count": 0}

    async def flaky_rules_program(examples_text, *, run):
        del examples_text, run
        calls["count"] += 1
        if calls["count"] == 1:
            raise ContextWindowExceededError(message="too long")
        return "rule one"

    infer.rules_induction_program = cast("Any", flaky_rules_program)
    predictor = SimpleModule().predictor
    trainset = [
        Example.from_record({"input": "a", "output": "1"}, input_keys=("input",)),
        Example.from_record({"input": "b", "output": "2"}, input_keys=("input",)),
    ]
    run = make_run(lm=DummyLM([{}]))
    rules = await infer.induce_natural_language_rules(predictor=predictor, trainset=trainset, run=run)
    assert rules == "rule one"
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_induce_rules_propagates_non_recoverable_errors(make_run):
    infer = InferRules(metric=_unit_metric, num_rules=1)

    async def failing_rules_program(examples_text, *, run):
        del examples_text, run
        raise RuntimeError("unexpected")

    infer.rules_induction_program = cast("Any", failing_rules_program)
    predictor = SimpleModule().predictor
    trainset = [Example.from_record({"input": "a", "output": "1"}, input_keys=("input",))]
    run = make_run(lm=DummyLM([{}]))
    with pytest.raises(RuntimeError, match="unexpected"):
        await infer.induce_natural_language_rules(predictor=predictor, trainset=trainset, run=run)
