import random

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.lm import LM
from dspy.predict.predict import Predict
from dspy.primitives import Example, Module, Prediction
from dspy.runtime.optimization_trace import FailedPrediction, TraceData
from dspy.teleprompt.grpo.rollout_groups import build_rollout_batches, validate_trace_data_and_log_issues
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts


class _SinglePredictModule(Module):
    def __init__(self, predictor: Predict) -> None:
        super().__init__()
        self.predictor = predictor

    async def _aforward_impl(self, *, run, options=None, **inputs):
        return await self.predictor(**inputs, run=run, options=options)


def _make_trace_data(
    *,
    example_ind: int,
    predictor: Predict,
    inputs: dict,
    outputs: Prediction | FailedPrediction,
    score: float = 1.0,
) -> TraceData:
    return {
        "example_ind": example_ind,
        "example": Example.from_record(inputs, input_keys=tuple(inputs)),
        "prediction": outputs,
        "trace": [(predictor, inputs, outputs)],
        "score": score,
    }


def test_validate_trace_data_accepts_matching_grid():
    task_spec = ts("question -> answer")
    predictor = Predict(task_spec)
    fingerprint = predictor.task_spec.fingerprint()
    trace_data = [
        [
            [
                _make_trace_data(
                    example_ind=0, predictor=predictor, inputs={"question": "q"}, outputs=Prediction(answer="a")
                )
            ]
        ]
    ]
    validate_trace_data_and_log_issues(
        trace_data,
        subsample_training_dataset=[Example.from_record({"question": "q"}, input_keys=("question",))],
        num_teachers=1,
        num_samples_per_input=1,
        pred_signature_hash_to_ind={fingerprint: 0},
    )


def test_build_rollout_batches_success_path(make_run):
    task_spec = ts("question -> answer")
    predictor = Predict(task_spec)
    predictor.set_lm(LM("openai/gpt-4o-mini"))
    student = _SinglePredictModule(predictor)
    fingerprint = predictor.task_spec.fingerprint()
    trace_data = [
        [
            [
                _make_trace_data(
                    example_ind=0,
                    predictor=predictor,
                    inputs={"question": "What is 2+2?"},
                    outputs=Prediction(answer="4"),
                    score=1.0,
                )
            ]
        ]
    ]
    run = make_run(lm=DummyLM([{}]), adapter=ChatAdapter())
    batches = build_rollout_batches(
        trace_data,
        student=student,
        pred_signature_hash_to_ind={fingerprint: 0},
        num_rollouts_per_grpo_step=1,
        adapter=ChatAdapter(),
        run=run,
        format_failure_score=-1.0,
        variably_invoked_predictor_grouping_mode="truncate",
        variably_invoked_predictor_fill_strategy=None,
        rng=random.Random(0),
    )
    assert len(batches) == 1
    assert len(batches[0]) == 1
    assert len(batches[0][0]) == 1
    assert batches[0][0][0].reward == 1.0


def test_build_rollout_batches_format_failure_path(make_run):
    task_spec = ts("question -> answer")
    predictor = Predict(task_spec)
    predictor.set_lm(LM("openai/gpt-4o-mini"))
    student = _SinglePredictModule(predictor)
    fingerprint = predictor.task_spec.fingerprint()
    failed = FailedPrediction(completion_text="not json", format_reward=-0.5)
    trace_data = [
        [
            [
                _make_trace_data(
                    example_ind=0,
                    predictor=predictor,
                    inputs={"question": "q"},
                    outputs=failed,
                    score=-0.5,
                )
            ]
        ]
    ]
    run = make_run(lm=DummyLM([{}]), adapter=ChatAdapter())
    batches = build_rollout_batches(
        trace_data,
        student=student,
        pred_signature_hash_to_ind={fingerprint: 0},
        num_rollouts_per_grpo_step=1,
        adapter=ChatAdapter(),
        run=run,
        format_failure_score=-1.0,
        variably_invoked_predictor_grouping_mode="truncate",
        variably_invoked_predictor_fill_strategy=None,
        rng=random.Random(0),
    )
    assert batches[0][0][0].reward == -0.5
    assert batches[0][0][0].completion.content == "not json"
