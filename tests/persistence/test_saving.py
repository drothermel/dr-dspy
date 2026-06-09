import asyncio
import logging
from unittest.mock import patch

import pytest
from typing_extensions import override

from dspy.persistence import load
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.predict import Predict
from dspy.primitives import Example, Module
from dspy.task_spec import default_task_instructions, input_field, make_task_spec, output_field
from dspy.teleprompt.bootstrap import BootstrapFewShot
from dspy.teleprompt.compile_params import BootstrapFewShotCompileParams
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts

QA_TASK_SPEC = ts("question->answer", instructions=default_task_instructions(inputs=("question",), outputs=("answer",)))


def test_save_predict(tmp_path, make_run):
    predict = Predict(QA_TASK_SPEC)
    predict.save(tmp_path, save_program=True)
    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "program.pkl").exists()
    loaded_predict = load(tmp_path, allow_pickle=True)
    assert isinstance(loaded_predict, Predict)
    assert predict.task_spec == loaded_predict.task_spec


def test_save_custom_model(tmp_path, make_run):

    class CustomModel(Module):
        def __init__(self):
            self.cot1 = ChainOfThought(ts("question->refined_question"))
            self.cot2 = ChainOfThought(ts("refined_question->answer"))

    model = CustomModel()
    model.save(tmp_path, save_program=True)
    loaded_model = load(tmp_path, allow_pickle=True)
    assert isinstance(loaded_model, CustomModel)
    assert len(model.predictors()) == len(loaded_model.predictors())
    for predictor, loaded_predictor in zip(model.predictors(), loaded_model.predictors(), strict=False):
        assert predictor.task_spec == loaded_predictor.task_spec


def test_save_model_with_custom_signature(tmp_path, make_run):
    import datetime

    MySignature = make_task_spec(
        {
            "current_date": input_field("current_date", type_=datetime.date, desc="The current date."),
            "target_date": input_field("target_date", type_=datetime.date, desc="The target date."),
            "date_diff": output_field(
                "date_diff", type_=int, desc="The difference in days between the current_date and the target_date"
            ),
        },
        instructions="Compute date difference.",
        name="MySignature",
    )
    predict = Predict(MySignature)
    predict.task_spec = predict.task_spec.with_instructions("You are a helpful assistant.")
    predict.save(tmp_path, save_program=True)
    loaded_predict = load(tmp_path, allow_pickle=True)
    assert isinstance(loaded_predict, Predict)
    assert predict.task_spec == loaded_predict.task_spec


@pytest.mark.extra
def test_save_compiled_model(tmp_path, make_run):
    predict = Predict(QA_TASK_SPEC)
    run = make_run(lm=DummyLM([{"answer": "blue"}, {"answer": "white"}] * 10))
    trainset = [
        {"question": "What is the color of the sky?", "answer": "blue"},
        {"question": "What is the color of the ocean?", "answer": "blue"},
        {"question": "What is the color of the milk?", "answer": "white"},
        {"question": "What is the color of the coffee?", "answer": "black"},
    ]
    trainset = [Example.from_record(example).with_input_keys("question") for example in trainset]

    def dummy_metric(example, pred, trace=None):
        return True

    optimizer = BootstrapFewShot(max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=5, metric=dummy_metric)
    compiled_predict = asyncio.run(
        optimizer.compile(predict, params=BootstrapFewShotCompileParams(trainset=trainset), run=run)
    )
    compiled_predict.save(tmp_path, save_program=True)
    loaded_predict = load(tmp_path, allow_pickle=True)
    assert compiled_predict.demos == loaded_predict.demos
    assert compiled_predict.task_spec == loaded_predict.task_spec


def test_load_with_version_mismatch(tmp_path):
    from dspy.persistence import logger

    save_versions = {"python": "3.9", "dspy": "2.4.0", "cloudpickle": "2.0"}
    load_versions = {"python": "3.10", "dspy": "2.5.0", "cloudpickle": "2.1"}
    predict = Predict(QA_TASK_SPEC)

    class ListHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages = []

        @override
        def emit(self, record):
            self.messages.append(record.getMessage())

    handler = ListHandler()
    original_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        with patch("dspy.persistence.get_dependency_versions", return_value=save_versions):
            predict.save(tmp_path, save_program=True)
        handler.messages.clear()
        with patch("dspy.persistence.get_dependency_versions", return_value=load_versions):
            loaded_predict = load(tmp_path, allow_pickle=True)
        mismatch_messages = [msg for msg in handler.messages if "There is a mismatch of" in msg]
        assert len(mismatch_messages) == 3
        assert isinstance(loaded_predict, Predict)
        assert predict.task_spec == loaded_predict.task_spec
    finally:
        logger.setLevel(original_level)
        logger.removeHandler(handler)


def test_pickle_loading_requires_explicit_permission(tmp_path):
    predict = Predict(QA_TASK_SPEC)
    predict.save(tmp_path, save_program=True)
    with pytest.raises(ValueError, match="Loading with pickle is not allowed"):
        load(tmp_path)
    loaded_predict = load(tmp_path, allow_pickle=True)
    assert isinstance(loaded_predict, Predict)


def test_pkl_file_loading_requires_explicit_permission(tmp_path):
    predict = Predict(QA_TASK_SPEC)
    pkl_path = tmp_path / "model.pkl"
    predict.save(pkl_path)
    new_predict = Predict(QA_TASK_SPEC)
    with pytest.raises(ValueError, match=r"Loading \.pkl files can run arbitrary code"):
        new_predict.load(pkl_path)
    new_predict.load(pkl_path, allow_pickle=True)
    assert new_predict.dump_state() == predict.dump_state()


def test_json_file_loading_works_without_permission(tmp_path):
    predict = Predict(QA_TASK_SPEC)
    json_path = tmp_path / "model.json"
    predict.save(json_path)
    new_predict = Predict(QA_TASK_SPEC)
    new_predict.load(json_path)
    assert new_predict.dump_state() == predict.dump_state()
