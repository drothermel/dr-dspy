from dspy.predict.predict import Predict
from dspy.propose.utils import create_predictor_level_history_string, strip_prefix
from tests.task_spec.helpers import ts


def test_strip_prefix_removes_label():
    assert strip_prefix('Assistant: "hello world"') == "hello world"


def test_create_predictor_level_history_string_empty_logs():
    program = Predict(ts("question -> answer"))
    history = create_predictor_level_history_string(
        base_program=program,
        predictor_i=0,
        trial_logs={},
        top_n=3,
    )
    assert history == ""
