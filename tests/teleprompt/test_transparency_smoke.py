import pytest

from dspy.adapters.json_adapter import JSONAdapter
from dspy.predict.predict import Predict
from dspy.primitives import Example
from dspy.runtime import CallLogMode, TelemetryConfig, TransparencyMode
from dspy.task_spec import TaskSpec, input_field, output_field
from dspy.teleprompt.bootstrap import BootstrapFewShot
from dspy.teleprompt.compile_params import BootstrapFewShotCompileParams, COPROCompileParams, EvaluateCompileParams
from dspy.teleprompt.copro_optimizer import COPRO
from dspy.testing import DummyLM


class QATaskSpec(TaskSpec):
    name: str = "framework.smoke.qa"
    instructions: str = "Answer the question."
    inputs: tuple = (input_field("question", desc="The question to answer."),)
    outputs: tuple = (output_field("answer", desc="The answer."),)


@pytest.mark.asyncio
async def test_bootstrap_few_shot_smoke_strict(make_run):
    json_adapter = JSONAdapter()
    lm = DummyLM([{"answer": "4"}] * 5, adapter=json_adapter)
    run = make_run(
        lm=lm,
        adapter=json_adapter,
        telemetry=TelemetryConfig(transparency=TransparencyMode.strict, call_log=CallLogMode.memory),
    )
    student = Predict(QATaskSpec())
    trainset = [Example.from_record({"question": "2+2", "answer": "4"}, input_keys=("question",))]
    teleprompter = BootstrapFewShot(
        metric=lambda example, pred, _trace=None: pred.answer == example.answer,
        max_bootstrapped_demos=1,
        max_labeled_demos=0,
        teacher_run=run.fork(lm=lm, adapter=json_adapter),
    )
    compiled = await teleprompter.compile(student, params=BootstrapFewShotCompileParams(trainset=trainset), run=run)
    result = await compiled(question="2+2", run=run)
    assert result.answer == "4"


@pytest.mark.asyncio
async def test_copro_smoke_strict(make_run):
    json_adapter = JSONAdapter()
    copro_answer = {"proposed_instruction": "Answer carefully.", "proposed_prefix_for_output_field": "Answer:"}
    lm = DummyLM([copro_answer, copro_answer, copro_answer], adapter=json_adapter)
    run = make_run(
        lm=lm,
        adapter=json_adapter,
        telemetry=TelemetryConfig(transparency=TransparencyMode.strict, call_log=CallLogMode.memory),
    )
    student = Predict(QATaskSpec())
    teleprompter = COPRO(
        metric=lambda _example, _pred, _trace=None: 1.0, prompt_model=lm, breadth=2, depth=1, init_temperature=1.0
    )
    compiled = await teleprompter.compile(
        student,
        params=COPROCompileParams(
            trainset=[Example.from_record({"question": "2+2", "answer": "4"}, input_keys=("question",))],
            evaluate=EvaluateCompileParams(max_concurrency=1),
        ),
        run=run,
    )
    assert compiled is not None
