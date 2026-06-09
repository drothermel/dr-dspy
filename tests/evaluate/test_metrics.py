from typing import Any, cast

from dspy.evaluate.metrics import answer_exact_match
from dspy.predict.predict import Predict
from dspy.primitives import Example
from tests.task_spec.helpers import ts


def test_answer_exact_match_string():
    example = Example.from_record({"question": "What is 1+1?", "answer": "2"}, input_keys=("question",))
    pred = cast("Any", Predict(ts("question -> answer")))
    pred.answer = "2"
    assert answer_exact_match(example, pred)


def test_answer_exact_match_list():
    example = Example.from_record({"question": "What is 1+1?", "answer": ["2", "two"]}, input_keys=("question",))
    pred = cast("Any", Predict(ts("question -> answer")))
    pred.answer = "2"
    assert answer_exact_match(example, pred)


def test_answer_exact_match_no_match():
    example = Example.from_record({"question": "What is 1+1?", "answer": "2"}, input_keys=("question",))
    pred = cast("Any", Predict(ts("question -> answer")))
    pred.answer = "3"
    assert not answer_exact_match(example, pred)
