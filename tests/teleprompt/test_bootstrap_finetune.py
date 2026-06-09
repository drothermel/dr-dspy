import asyncio
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

from dspy.predict.predict import Predict
from dspy.primitives import Example, Module, Prediction
from dspy.teleprompt.bootstrap_finetune import BootstrapFinetune, filter_trace_data_for_finetune
from dspy.teleprompt.compile_params import BootstrapFewShotCompileParams
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts

if TYPE_CHECKING:
    from dspy.clients.lm import LM
    from dspy.teleprompt.bootstrap_trace import TraceData


def simple_metric(example, prediction, trace=None):
    return example.output == prediction.output


examples = [
    Example.from_record({"input": "What is the color of the sky?", "output": "blue"}, input_keys=("input",)),
    Example.from_record(
        {"input": "What does the fox say?", "output": "Ring-ding-ding-ding-dingeringeding!"}, input_keys=("input",)
    ),
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

    async def _aforward_impl(self, *, run, options=None, **inputs):
        return await self.predictor(run=run, options=options, **inputs)


def test_compile_with_predict_instances(make_run):
    student = SimpleModule(ts("input -> output"))
    teacher = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "blue"}, {"output": "Ring-ding-ding-ding-dingeringeding!"}])
    run = make_run(lm=lm)
    student.set_lm(lm)
    teacher.set_lm(lm)
    bootstrap = BootstrapFinetune(metric=simple_metric)
    with patch.object(bootstrap, "finetune_lms", new_callable=AsyncMock) as mock_finetune:
        mock_finetune.return_value = {(lm, None): lm}
        result = asyncio.run(
            bootstrap.compile(
                student, params=BootstrapFewShotCompileParams(trainset=trainset, teacher=teacher), run=run
            )
        )
        compiled_student = result.program
        assert compiled_student is not None, "Failed to compile student"
        assert hasattr(compiled_student, "_compiled") and compiled_student._compiled, "Student compilation flag not set"
        mock_finetune.assert_called_once()


def test_error_handling_missing_lm(make_run):
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    student = SimpleModule(ts("input -> output"))
    bootstrap = BootstrapFinetune(metric=simple_metric)
    try:
        asyncio.run(bootstrap.compile(student, params=BootstrapFewShotCompileParams(trainset=trainset), run=run))
        raise AssertionError("Should have raised ValueError for missing LM")
    except ValueError as e:
        assert "does not have an LM assigned" in str(e)
        assert "set_lm" in str(e)


def test_filter_trace_data_keeps_zero_scores():
    trace_data = cast(
        "list[TraceData]",
        [
            {"example_ind": 0, "example": examples[0], "prediction": None, "trace": [], "score": 0.0},
            {"example_ind": 1, "example": examples[1], "prediction": None, "trace": [], "score": False},
            {"example_ind": 2, "example": examples[0], "prediction": None, "trace": [], "score": None},
            {"example_ind": 3, "example": examples[1], "prediction": None, "trace": [], "score": 1.0},
        ],
    )
    filtered = filter_trace_data_for_finetune(trace_data, metric=simple_metric)
    assert len(filtered) == 3
    assert all(entry["score"] is not None for entry in filtered)


class TwoPredictorModule(Module):
    def __init__(self):
        super().__init__()
        self.pred_a = Predict(ts("input -> output_a"))
        self.pred_b = Predict(ts("input -> output_b"))


def test_prepare_finetune_data_multitask_false_filters_by_predictor(make_run):
    lm = DummyLM([{"output_a": "a", "output_b": "b"}])
    run = make_run(lm=lm)
    module = TwoPredictorModule()
    module.pred_a.set_lm(lm)
    module.pred_b.set_lm(lm)
    example = examples[0]
    trace_data = cast(
        "list[TraceData]",
        [
            {
                "example_ind": 0,
                "example": example,
                "prediction": Prediction(output_a="a", output_b="b"),
                "trace": [
                    (module.pred_a, {"input": example.input}, Prediction(output_a="a")),
                    (module.pred_b, {"input": example.input}, Prediction(output_b="b")),
                ],
                "score": 1.0,
            }
        ],
    )
    bootstrap = BootstrapFinetune(metric=None)
    lm_for_prepare = cast("LM", lm)

    with patch("dspy.teleprompt.bootstrap_finetune.build_call_data_from_trace") as mock_build:
        mock_build.return_value = {"messages": []}
        bootstrap._prepare_finetune_data(trace_data, lm_for_prepare, target_pred_ind=0, run=run)
        assert mock_build.call_count == 1
        assert mock_build.call_args.kwargs["pred_ind"] == 0

        mock_build.reset_mock()
        bootstrap._prepare_finetune_data(trace_data, lm_for_prepare, target_pred_ind=1, run=run)
        assert mock_build.call_count == 1
        assert mock_build.call_args.kwargs["pred_ind"] == 1

        mock_build.reset_mock()
        bootstrap._prepare_finetune_data(trace_data, lm_for_prepare, target_pred_ind=None, run=run)
        assert mock_build.call_count == 2
        assert {call.kwargs["pred_ind"] for call in mock_build.call_args_list} == {0, 1}


def test_finetune_lms_uses_asyncio_to_thread():
    lm = MagicMock()
    job = MagicMock()
    finetuned_lm = MagicMock()
    job.result.return_value = finetuned_lm
    job.thread = MagicMock()
    lm.finetune.return_value = job
    key = (lm, 0)
    finetune_dict = {
        key: {
            "lm": lm,
            "train_data": [],
            "train_data_format": "chat",
            "train_kwargs": {},
        }
    }

    async def fake_to_thread(fn, /, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch("dspy.teleprompt.bootstrap_finetune.asyncio.to_thread", side_effect=fake_to_thread) as mock_to_thread:
        result = asyncio.run(BootstrapFinetune.finetune_lms(finetune_dict))

    assert result[key] is finetuned_lm
    assert mock_to_thread.await_count == 2
    job.result.assert_called_once()
    job.thread.join.assert_called_once()
