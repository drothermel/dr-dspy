import asyncio
import json
import signal
import tempfile
import threading

import pytest
from typing_extensions import override

from dspy.history import TurnLog
from dspy.evaluate.evaluate import Evaluate, EvaluationResult
from dspy.evaluate.metrics import answer_exact_match
from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.utils.callback import BaseCallback
from dspy.utils.dummies import DummyLM
from tests.task_spec.helpers import ts


def new_example(question, answer):
    return Example.from_record({"question": question, "answer": answer}, input_keys=("question",))


def test_evaluate_initialization(make_run):
    devset = [new_example("What is 1+1?", "2")]
    ev = Evaluate(devset=devset, metric=answer_exact_match, display_progress=False)
    assert ev.devset == devset
    assert ev.metric == answer_exact_match
    assert ev.num_threads is None
    assert not ev.display_progress


def test_evaluate_call(make_run):
    run = make_run(lm=DummyLM({"What is 1+1?": {"answer": "2"}, "What is 2+2?": {"answer": "4"}}))
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    program = Predict(ts("question -> answer"))
    assert asyncio.run(program(question="What is 1+1?", run=run)).answer == "2"
    ev = Evaluate(devset=devset, metric=answer_exact_match, display_progress=False)
    score = asyncio.run(ev(program, run=run))
    assert score.score == 100.0


def test_evaluate_single_thread_runs_in_main_thread(make_run):
    run = make_run(lm=DummyLM({"What is 1+1?": {"answer": "2"}, "What is 2+2?": {"answer": "4"}}))
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    execution_threads = []
    original_metric = answer_exact_match

    def tracking_metric(example, prediction, trace=None):
        execution_threads.append(threading.current_thread())
        return original_metric(example, prediction, trace)

    program = Predict(ts("question -> answer"))
    ev = Evaluate(devset=devset, metric=tracking_metric, display_progress=False, num_threads=1)
    result = asyncio.run(ev(program, run=run))
    assert result.score == 100.0
    assert all(t is threading.main_thread() for t in execution_threads)


@pytest.mark.extra
def test_construct_result_df(make_run):
    import pandas as pd

    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4"), new_example("What is 3+3?", "-1")]
    ev = Evaluate(devset=devset, metric=answer_exact_match)
    results = [
        (devset[0], Example.from_record({"answer": "2"}), 100.0),
        (devset[1], Example.from_record({"answer": "4"}), 100.0),
        (devset[2], Example.from_record({"answer": "-1"}), 0.0),
    ]
    result_df = ev._construct_result_table(results, answer_exact_match.__name__)
    pd.testing.assert_frame_equal(
        result_df,
        pd.DataFrame(
            {
                "question": ["What is 1+1?", "What is 2+2?", "What is 3+3?"],
                "example_answer": ["2", "4", "-1"],
                "pred_answer": ["2", "4", "-1"],
                "answer_exact_match": [100.0, 100.0, 0.0],
            }
        ),
    )


def test_multithread_evaluate_call(make_run):
    run = make_run(lm=DummyLM({"What is 1+1?": {"answer": "2"}, "What is 2+2?": {"answer": "4"}}))
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    program = Predict(ts("question -> answer"))
    assert asyncio.run(program(question="What is 1+1?", run=run)).answer == "2"
    ev = Evaluate(devset=devset, metric=answer_exact_match, display_progress=False, num_threads=2)
    result = asyncio.run(ev(program, run=run))
    assert result.score == 100.0


def test_multi_thread_evaluate_call_cancelled(monkeypatch, make_run):

    class SlowLM(DummyLM):
        @override
        def __call__(self, *args: object, **kwargs: object):
            import time

            time.sleep(1)
            return super().__call__(*args, **kwargs)

    run = make_run(lm=SlowLM({"What is 1+1?": {"answer": "2"}, "What is 2+2?": {"answer": "4"}}))
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    program = Predict(ts("question -> answer"))
    assert asyncio.run(program(question="What is 1+1?", run=run)).answer == "2"

    def sleep_then_interrupt():
        import time

        time.sleep(0.1)
        import os

        os.kill(os.getpid(), signal.SIGINT)

    input_thread = threading.Thread(target=sleep_then_interrupt)
    input_thread.start()
    ev = Evaluate(devset=devset, metric=answer_exact_match, display_progress=False, num_threads=2)
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(ev(program, run=run))


def test_evaluate_call_wrong_answer(make_run):
    run = make_run(lm=DummyLM({"What is 1+1?": {"answer": "0"}, "What is 2+2?": {"answer": "0"}}))
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    program = Predict(ts("question -> answer"))
    ev = Evaluate(devset=devset, metric=answer_exact_match, display_progress=False)
    result = asyncio.run(ev(program, run=run))
    assert result.score == 0.0


@pytest.mark.extra
@pytest.mark.parametrize(
    "program_with_example",
    [
        (Predict(ts("question -> answer")), new_example("What is 1+1?", "2")),
        (
            lambda text: asyncio.run(Predict(ts("text: str -> entities: list[str]"))(text=text)).entities,
            Example.from_record({"text": "United States", "entities": ["United States"]}, input_keys=("text",)),
        ),
        (
            lambda text: asyncio.run(Predict(ts("text: str -> entities: list[dict[str, str]]"))(text=text)).entities,
            Example.from_record(
                {"text": "United States", "entities": [{"name": "United States", "type": "location"}]},
                input_keys=("text",),
            ),
        ),
        (
            lambda text: asyncio.run(Predict(ts("text: str -> first_word: Tuple[str, int]"))(text=text)).words,
            Example.from_record({"text": "United States", "first_word": ("United", 6)}, input_keys=("text",)),
        ),
    ],
)
@pytest.mark.parametrize("display_table", [True, False, 1])
def test_evaluate_display_table(program_with_example, display_table, capfd, make_run):
    program, example = program_with_example
    example_input = next(iter(example.as_inputs().values()))
    example_output = {key: value for key, value in example.to_dict().items() if key not in example.as_inputs()}
    run = make_run(lm=DummyLM({example_input: example_output}))
    ev = Evaluate(
        devset=[example], metric=lambda example, pred, **_kwargs: example == pred, display_table=display_table
    )
    assert ev.display_table == display_table
    asyncio.run(ev(program, run=run))
    out, _ = capfd.readouterr()
    if display_table:
        example_input = next(iter(example.as_inputs().values()))
        assert example_input in out


def test_evaluate_callback(make_run):

    class TestCallback(BaseCallback):
        def __init__(self):
            self.start_call_inputs = None
            self.start_call_count = 0
            self.end_call_outputs = None
            self.end_call_count = 0

        @override
        def on_evaluate_start(self, call_id: str, instance, inputs):
            self.start_call_inputs = inputs
            self.start_call_count += 1

        @override
        def on_evaluate_end(self, call_id: str, outputs, exception=None):
            self.end_call_outputs = outputs
            self.end_call_count += 1

    callback = TestCallback()
    run = make_run(lm=DummyLM({"What is 1+1?": {"answer": "2"}, "What is 2+2?": {"answer": "4"}}), callbacks=[callback])
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    program = Predict(ts("question -> answer"))
    assert asyncio.run(program(question="What is 1+1?", run=run)).answer == "2"
    ev = Evaluate(devset=devset, metric=answer_exact_match, display_progress=False)
    result = asyncio.run(ev(program, run=run))
    assert result.score == 100.0
    assert callback.start_call_inputs is not None
    assert callback.start_call_inputs["program"] == program
    assert callback.start_call_count == 1
    assert callback.end_call_outputs is not None
    assert callback.end_call_outputs.score == 100.0
    assert callback.end_call_count == 1


def test_evaluation_result_repr(make_run):
    result = EvaluationResult(
        score=100.0, results=[(new_example("What is 1+1?", "2"), Example.from_record({"answer": "2"}), 100.0)]
    )
    assert repr(result) == "EvaluationResult(score=100.0, results=<list of 1 results>)"


def test_evaluate_save_as_json_with_history(make_run):
    run = make_run(lm=DummyLM({"What is 1+1?": {"answer": "2"}, "What is 2+2?": {"answer": "4"}}))
    history1 = TurnLog(turns=({"question": "Previous Q1", "answer": "Previous A1"}))
    history2 = TurnLog(turns=(
            {"question": "Previous Q2", "answer": "Previous A2"},
            {"question": "Previous Q3", "answer": "Previous A3"},
        ))
    devset = [
        Example.from_record({"question": "What is 1+1?", "answer": "2", "history": history1}, input_keys=("question",)),
        Example.from_record({"question": "What is 2+2?", "answer": "4", "history": history2}, input_keys=("question",)),
    ]
    program = Predict(ts("question -> answer"))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_json = f.name
    try:
        evaluator = Evaluate(devset=devset, metric=answer_exact_match, display_progress=False, save_as_json=temp_json)
        result = asyncio.run(evaluator(program, run=run))
        assert result.score == 100.0
        with open(temp_json) as f:
            data = json.load(f)
        assert len(data) == 2
        assert "history" in data[0]
        assert isinstance(data[0]["history"], dict)
        assert "messages" in data[0]["history"]
        assert len(data[0]["history"]["messages"]) == 1
        assert data[0]["history"]["messages"][0] == {"question": "Previous Q1", "answer": "Previous A1"}
        assert "history" in data[1]
        assert isinstance(data[1]["history"], dict)
        assert "messages" in data[1]["history"]
        assert len(data[1]["history"]["messages"]) == 2
        assert data[1]["history"]["messages"][0] == {"question": "Previous Q2", "answer": "Previous A2"}
        assert data[1]["history"]["messages"][1] == {"question": "Previous Q3", "answer": "Previous A3"}
    finally:
        import os

        if os.path.exists(temp_json):
            os.unlink(temp_json)


def test_evaluate_save_as_csv_with_history(make_run):
    run = make_run(lm=DummyLM({"What is 1+1?": {"answer": "2"}}))
    history = TurnLog(turns=({"question": "Previous Q", "answer": "Previous A"}))
    devset = [
        Example.from_record({"question": "What is 1+1?", "answer": "2", "history": history}, input_keys=("question",))
    ]
    program = Predict(ts("question -> answer"))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        temp_csv = f.name
    try:
        evaluator = Evaluate(devset=devset, metric=answer_exact_match, display_progress=False, save_as_csv=temp_csv)
        result = asyncio.run(evaluator(program, run=run))
        assert result.score == 100.0
        import csv

        with open(temp_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert "history" in rows[0]
        assert "messages" in rows[0]["history"]
    finally:
        import os

        if os.path.exists(temp_csv):
            os.unlink(temp_csv)
