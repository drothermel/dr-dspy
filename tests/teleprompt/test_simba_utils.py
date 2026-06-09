from dspy.predict.predict import Predict
from dspy.teleprompt.simba_utils import append_a_demo
from tests.task_spec.helpers import ts


def test_append_a_demo_does_not_mutate_trace_inputs():
    predictor = Predict(ts("question -> answer"))
    trace_inputs = {"question": "x" * 100}
    bucket_entry = {
        "score": 0.9,
        "trace": [(predictor, trace_inputs, {"answer": "ok"})],
    }
    append_demo = append_a_demo(demo_input_field_maxlen=10)
    append_demo(
        [bucket_entry],
        predictor,
        predictor2name={id(predictor): "predictor"},
        name2predictor={"predictor": predictor},
        batch_10p_score=0.1,
    )
    assert trace_inputs["question"] == "x" * 100


def test_append_a_rule_copy_does_not_mutate_original_bucket_scores():
    bucket = [
        {"score": 0.8, "trace": [], "prediction": {"answer": "a"}, "example": {}, "output_metadata": {}},
        {"score": 0.7, "trace": [], "prediction": {"answer": "b"}, "example": {}, "output_metadata": {}},
    ]
    good = {**bucket[0].copy(), "score": "N/A"}
    assert bucket[0]["score"] == 0.8
    assert good["score"] == "N/A"
