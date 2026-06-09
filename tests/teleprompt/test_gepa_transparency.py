from dspy.adapters.json_adapter import JSONAdapter
from dspy.runtime import CallLogMode, TelemetryConfig, TransparencyMode
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts


def test_gepa_reflection_uses_predict_path(make_run):
    student = __import__("dspy.predict.predict", fromlist=["Predict"]).Predict(
        ts("question -> answer", instructions="Answer.")
    )
    json_adapter = JSONAdapter()
    reflection_lm = DummyLM([{"new_instruction": "Better instruction."}], adapter=json_adapter)
    run = make_run(
        lm=reflection_lm,
        adapter=json_adapter,
        telemetry=TelemetryConfig(transparency=TransparencyMode.strict, call_log=CallLogMode.memory),
    )
    adapter = DspyAdapter(
        student_module=student,
        metric_fn=lambda _example, _prediction, **_kwargs: 1.0,
        feedback_map={},
        reflection_lm=reflection_lm,
        run=run,
    )
    result = adapter.propose_new_texts(
        candidate={"predict": "Answer the question."},
        reflective_dataset={
            "predict": [{"Inputs": {"question": "2+2"}, "Generated Outputs": {"answer": "5"}, "Feedback": "wrong"}]
        },
        components_to_update=["predict"],
    )
    assert result["predict"] == "Better instruction."
