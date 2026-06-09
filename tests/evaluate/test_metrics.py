from typing import Any, cast

from dspy.evaluate.metrics import answer_exact_match
from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from tests.task_spec.helpers import ts


def test_answer_exact_match_string():
    example = Example(question="What is 1+1?", answer="2").with_inputs("question")
    pred = cast("Any", Predict(ts("question -> answer")))
    pred.answer = "2"
    assert answer_exact_match(example, pred)


def test_answer_exact_match_list():
    example = Example(question="What is 1+1?", answer=["2", "two"]).with_inputs("question")
    pred = cast("Any", Predict(ts("question -> answer")))
    pred.answer = "2"
    assert answer_exact_match(example, pred)


def test_answer_exact_match_no_match():
    example = Example(question="What is 1+1?", answer="2").with_inputs("question")
    pred = cast("Any", Predict(ts("question -> answer")))
    pred.answer = "3"
    assert not answer_exact_match(example, pred)
