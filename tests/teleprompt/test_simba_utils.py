from dspy.adapters.json_adapter import JSONAdapter
from dspy.predict.predict import Predict
from dspy.primitives import Module
from dspy.runtime import CallLogMode, RunContext, TelemetryConfig
from dspy.teleprompt.simba_utils import prepare_models_for_resampling
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts


class SimpleModule(Module):
    def __init__(self, lm):
        super().__init__()
        self.predictor = Predict(ts("input -> output"))
        self.set_lm(lm)

    async def _aforward_impl(self, *, run, options=None, **inputs):
        return await self.predictor(run=run, options=options, **inputs)


def test_prepare_models_for_resampling_does_not_mutate_teacher_kwargs():
    base_lm = DummyLM([{}])
    original_kwargs = dict(base_lm.kwargs)
    teacher_run = RunContext.create(
        lm=base_lm,
        adapter=JSONAdapter(),
        telemetry=TelemetryConfig(call_log=CallLogMode.memory),
    )
    program = SimpleModule(base_lm)
    run = teacher_run

    models = prepare_models_for_resampling(program=program, n=2, run=run, teacher_run=teacher_run)

    assert dict(base_lm.kwargs) == original_kwargs
    assert len(models) == 2
    assert models[0] is not base_lm
    assert models[0].kwargs.get("temperature") == 1.0
