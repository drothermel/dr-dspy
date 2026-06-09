import asyncio
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typing_extensions import override

from dspy.persistence import logger
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.predict import Predict
from dspy.primitives import Example, Module
from dspy.task_spec import default_task_instructions, input_field, make_task_spec, output_field
from dspy.teleprompt.bootstrap import BootstrapFewShot
from dspy.teleprompt.compile_params import BootstrapFewShotCompileParams
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts

QA_TASK_SPEC = ts("question->answer", instructions=default_task_instructions(inputs=("question",), outputs=("answer",)))


def test_load_state_is_transactional():
    Sig = ts("question -> answer")

    class Prog(Module):
        def __init__(self):
            super().__init__()
            self.a = ChainOfThought(Sig)
            self.b = ChainOfThought(Sig)

    source = Prog()
    sentinel = Example.from_record({"question": "q1", "answer": "a1"}, input_keys=("question",))
    source.a.predict.demos = [sentinel]
    source.b.predict.demos = [sentinel]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "state.json"
        source.save(str(path), save_program=False)
        raw = json.loads(path.read_text())
        corrupted = {k: v for k, v in raw.items() if "b." not in k}
        path.write_text(json.dumps(corrupted))
        template = Prog()
        assert template.a.predict.demos == []
        with pytest.raises(KeyError):
            template.load(str(path))
        assert template.a.predict.demos == [], "load_state partially mutated module before failing"


def test_save_and_load_with_json(tmp_path, make_run):
    model = ChainOfThought(ts("q -> a"))
    model.predict.task_spec = model.predict.task_spec.with_instructions("You are a helpful assistant.")
    model.predict.demos = [
        Example.from_record(
            {"q": "What is the capital of France?", "a": "Paris", "reasoning": "n/a"}, input_keys=("q",)
        ),
        Example.from_record(
            {
                "q": [
                    Example.from_record({"q": "What is the capital of France?"}),
                    Example.from_record({"q": "What is actually the capital of France?"}),
                ],
                "a": "Paris",
                "reasoning": "n/a",
            },
            input_keys=("q",),
        ),
    ]
    save_path = tmp_path / "model.json"
    model.save(save_path)
    new_model = ChainOfThought(ts("q -> a"))
    new_model.load(save_path)
    assert new_model.predict.task_spec == model.predict.task_spec
    assert new_model.predict.demos[0] == model.predict.demos[0].to_dict()
    assert new_model.predict.demos[1] == model.predict.demos[1].to_dict()


@pytest.mark.extra
def test_save_and_load_with_pkl(tmp_path, make_run):
    import datetime

    MySignature = make_task_spec(
        {
            "current_date": input_field("current_date", type_=datetime.date, desc="The current date."),
            "target_date": input_field("target_date", type_=datetime.date, desc="The target date."),
            "date_diff": output_field(
                "date_diff", type_=int, desc="The difference in days between the current_date and the target_date"
            ),
        },
        instructions="Just a custom task spec.",
        name="MySignature",
    )
    trainset = [
        {"current_date": datetime.date(2024, 1, 1), "target_date": datetime.date(2024, 1, 2), "date_diff": 1},
        {"current_date": datetime.date(2024, 1, 1), "target_date": datetime.date(2024, 1, 3), "date_diff": 2},
        {"current_date": datetime.date(2024, 1, 1), "target_date": datetime.date(2024, 1, 4), "date_diff": 3},
        {"current_date": datetime.date(2024, 1, 1), "target_date": datetime.date(2024, 1, 5), "date_diff": 4},
        {"current_date": datetime.date(2024, 1, 1), "target_date": datetime.date(2024, 1, 6), "date_diff": 5},
    ]
    trainset = [Example.from_record(example).with_input_keys("current_date", "target_date") for example in trainset]
    run = make_run(lm=DummyLM([{"date_diff": "1", "reasoning": "n/a"}, {"date_diff": "2", "reasoning": "n/a"}] * 10))
    cot = ChainOfThought(MySignature)
    asyncio.run(cot(current_date=datetime.date(2024, 1, 1), target_date=datetime.date(2024, 1, 2), run=run))

    def dummy_metric(example, pred, trace=None):
        return True

    optimizer = BootstrapFewShot(max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=5, metric=dummy_metric)
    compile_result = asyncio.run(
        optimizer.compile(cot, params=BootstrapFewShotCompileParams(trainset=trainset), run=run)
    )
    compiled_cot = compile_result.program
    compiled_cot.predict.task_spec = compiled_cot.predict.task_spec.with_instructions("You are a helpful assistant.")
    save_path = tmp_path / "program.pkl"
    compiled_cot.save(save_path)
    new_cot = ChainOfThought(MySignature)
    new_cot.load(save_path, allow_pickle=True)
    assert str(new_cot.predict.task_spec) == str(compiled_cot.predict.task_spec)
    assert new_cot.predict.demos == compiled_cot.predict.demos


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


def test_load_state_with_version_mismatch(tmp_path, make_run):
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
        save_path = tmp_path / "program.pkl"
        with patch("dspy.persistence.metadata.get_dependency_versions", return_value=save_versions):
            predict.save(save_path)
        with patch("dspy.persistence.metadata.get_dependency_versions", return_value=load_versions):
            loaded_predict = Predict(QA_TASK_SPEC)
            loaded_predict.load(save_path, allow_pickle=True)
        assert len(handler.messages) == 4
        assert "Saving state to .pkl" in handler.messages[0]
        for msg in handler.messages[1:]:
            assert "There is a mismatch of" in msg
        assert isinstance(loaded_predict, Predict)
        assert predict.task_spec == loaded_predict.task_spec
    finally:
        logger.setLevel(original_level)
        logger.removeHandler(handler)


def test_save_pkl_emits_save_warning(tmp_path):
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
        json_path = tmp_path / "module.json"
        predict.save(json_path)
        assert not any("Saving state to .pkl" in msg for msg in handler.messages)

        handler.messages.clear()
        pkl_path = tmp_path / "module.pkl"
        predict.save(pkl_path)
        assert any("Saving state to .pkl" in msg for msg in handler.messages)
    finally:
        logger.setLevel(original_level)
        logger.removeHandler(handler)


def test_load_warns_when_saved_metadata_missing_dependency_keys(tmp_path):
    save_versions = {"python": "3.9"}
    load_versions = {"python": "3.9", "dspy": "2.5.0", "cloudpickle": "2.1"}
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
        save_path = tmp_path / "program.pkl"
        with patch("dspy.persistence.metadata.get_dependency_versions", return_value=save_versions):
            predict.save(save_path)
        handler.messages.clear()
        with patch("dspy.persistence.metadata.get_dependency_versions", return_value=load_versions):
            loaded_predict = Predict(QA_TASK_SPEC)
            loaded_predict.load(save_path, allow_pickle=True)
        missing_key_messages = [
            msg
            for msg in handler.messages
            if "does not include `dspy`" in msg or "does not include `cloudpickle`" in msg
        ]
        assert len(missing_key_messages) == 2
        assert isinstance(loaded_predict, Predict)
    finally:
        logger.setLevel(original_level)
        logger.removeHandler(handler)
