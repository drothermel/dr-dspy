from typing import Any, cast

import pytest

from dspy.evaluate.metrics import (
    answer_exact_match,
    em_score,
    hotpot_f1_score,
    max_hotpot_f1_score,
    normalize_text,
    token_f1_score,
)
from dspy.predict.predict import Predict
from dspy.primitives import Example
from tests.task_spec.helpers import ts


def test_normalize_text_strips_articles_and_punctuation():
    assert normalize_text(s="The U.S.A. is great!") == "usa is great"


def test_normalize_text_collapses_whitespace():
    assert normalize_text(s="  hello   world  ") == "hello world"


def test_em_score_exact_match_after_normalization():
    assert em_score(prediction="The answer", ground_truth="answer")


def test_token_f1_score_partial_overlap():
    assert token_f1_score(prediction="foo bar baz", ground_truth="bar baz qux") == pytest.approx(2 / 3)


def test_hotpot_f1_score_yes_no_mismatch():
    assert hotpot_f1_score(prediction="yes", ground_truth="no") == 0.0
    assert hotpot_f1_score(prediction="maybe", ground_truth="no") == 0.0


def test_max_hotpot_f1_score_over_answers():
    assert max_hotpot_f1_score(prediction="Paris", answers_list=["London", "Paris"]) == pytest.approx(1.0)


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


def test_answer_exact_match_frac_threshold():
    example = Example.from_record({"question": "Capital?", "answer": "Paris France"}, input_keys=("question",))
    pred = cast("Any", Predict(ts("question -> answer")))
    pred.answer = "Paris"
    assert not answer_exact_match(example, pred, frac=1.0)
    assert answer_exact_match(example, pred, frac=0.5)
