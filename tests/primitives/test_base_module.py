import asyncio
import logging
import os
import threading
from typing import cast
from unittest.mock import patch

import pytest
from typing_extensions import override

from dspy.runtime import CallLogMode, TelemetryConfig

try:
    from litellm import Choices, Message, ModelResponse
    from litellm.types.utils import Usage
except ImportError:
    pytest.skip(reason="litellm is not installed", allow_module_level=True)
from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.lm import LM
from dspy.persistence import load
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.parallel import Parallel
from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.task_spec import default_task_instructions, input_field, make_task_spec, output_field
from dspy.teleprompt.bootstrap import BootstrapFewShot
from dspy.teleprompt.compile_params import BootstrapFewShotCompileParams
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts

QA_TASK_SPEC = ts("question->answer", instructions=default_task_instructions(inputs=("question",), outputs=("answer",)))
QUESTION_ANSWER_TASK_SPEC = ts(
    "question -> answer", instructions=default_task_instructions(inputs=("question",), outputs=("answer",))
)


def test_deepcopy_basic():
    cot = ChainOfThought(ts("q -> a"))
    cot_copy = cot.deepcopy()
    assert len(cot.parameters()) == len(cot_copy.parameters())
    assert id(cot.parameters()[0]) != id(cot_copy.parameters()[0])
    assert cot.parameters()[0].__dict__ == cot_copy.parameters()[0].__dict__


def test_deepcopy_with_uncopyable_modules(make_run):

    class CustomClass(Module):
        def __init__(self):
            self.lock = threading.Lock()
            self.cot = ChainOfThought(ts("q -> a"))

    model = CustomClass()
    model_copy = model.deepcopy()
    assert len(model.parameters()) == len(model_copy.parameters())
    assert id(model.lock) == id(model_copy.lock)
    assert id(model.parameters()[0]) != id(model_copy.parameters()[0])
    assert model.parameters()[0].__dict__ == model_copy.parameters()[0].__dict__


def test_deepcopy_with_nested_modules(make_run):

    class CustomClass1(Module):
        def __init__(self):
            self.lock = threading.Lock()
            self.cot = ChainOfThought(ts("q -> a"))

    class CustomClass2(Module):
        def __init__(self):
            self.submodel = CustomClass1()

    model = CustomClass2()
    model_copy = model.deepcopy()
    assert len(model.parameters()) == len(model_copy.parameters())
    assert id(model.submodel.lock) == id(model_copy.submodel.lock)
    assert id(model.parameters()[0]) != id(model_copy.parameters()[0])
    assert model.parameters()[0].__dict__ == model_copy.parameters()[0].__dict__


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
    compiled_cot = asyncio.run(optimizer.compile(cot, params=BootstrapFewShotCompileParams(trainset=trainset), run=run))
    compiled_cot.predict.task_spec = compiled_cot.predict.task_spec.with_instructions("You are a helpful assistant.")
    save_path = tmp_path / "program.pkl"
    compiled_cot.save(save_path)
    new_cot = ChainOfThought(MySignature)
    new_cot.load(save_path, allow_pickle=True)
    assert str(new_cot.predict.task_spec) == str(compiled_cot.predict.task_spec)
    assert new_cot.predict.demos == compiled_cot.predict.demos


def test_save_with_extra_modules(tmp_path, make_run):
    import sys

    custom_module_path = tmp_path / "custom_module.py"
    with open(custom_module_path, "w") as f:
        f.write(
            '\nfrom dspy.predict.chain_of_thought import ChainOfThought\nfrom dspy.primitives.module import Module\nfrom dspy.task_spec import make_task_spec\n\nclass MyModule(Module):\n    def __init__(self):\n        self.cot = ChainOfThought(make_task_spec("q -> a", instructions="Answer the question."))\n\n    async def aforward(self, q):\n        return await self.cot(q=q)\n'
        )
    sys.path.insert(0, str(tmp_path))
    try:
        import custom_module

        cot = custom_module.MyModule()
        cot.save(tmp_path, save_program=True)
        sys.modules.pop("custom_module", None)
        sys.path.remove(str(tmp_path))
        del custom_module
        with pytest.raises(ModuleNotFoundError):
            load(tmp_path, allow_pickle=True)
        sys.path.insert(0, str(tmp_path))
        import custom_module

        cot.save(tmp_path, modules_to_serialize=[custom_module], save_program=True)
        sys.modules.pop("custom_module", None)
        sys.path.remove(str(tmp_path))
        del custom_module
        loaded_module = load(tmp_path, allow_pickle=True)
        assert loaded_module.cot.predict.task_spec == cot.cot.predict.task_spec
    finally:
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))


def test_load_with_version_mismatch(tmp_path, make_run):
    from dspy.primitives.base_module import logger

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
        with patch("dspy.primitives.base_module.get_dependency_versions", return_value=save_versions):
            predict.save(save_path)
        with patch("dspy.primitives.base_module.get_dependency_versions", return_value=load_versions):
            loaded_predict = Predict(QA_TASK_SPEC)
            loaded_predict.load(save_path, allow_pickle=True)
        assert len(handler.messages) == 4
        assert ".pkl" in handler.messages[0]
        for msg in handler.messages[1:]:
            assert "There is a mismatch of" in msg
        assert isinstance(loaded_predict, Predict)
        assert predict.task_spec == loaded_predict.task_spec
    finally:
        logger.setLevel(original_level)
        logger.removeHandler(handler)


@pytest.mark.llm_call
def test_single_module_call_with_usage_tracker(lm_for_test, make_run):
    make_run(lm=LM(lm_for_test, temperature=0.0), telemetry=TelemetryConfig(track_usage=True))
    predict = ChainOfThought(ts("question -> answer"))
    output = predict(question="What is the capital of France?")
    lm_usage = output.get_lm_usage()
    assert len(lm_usage) == 1
    assert lm_usage[lm_for_test]["prompt_tokens"] > 0
    assert lm_usage[lm_for_test]["completion_tokens"] > 0
    assert lm_usage[lm_for_test]["total_tokens"] > 0


@pytest.mark.llm_call
def test_multi_module_call_with_usage_tracker(lm_for_test, make_run):
    make_run(lm=LM(lm_for_test, temperature=0.0), telemetry=TelemetryConfig(track_usage=True))

    class MyProgram(Module):
        def __init__(self):
            self.predict1 = ChainOfThought(ts("question -> answer"))
            self.predict2 = ChainOfThought(ts("question, answer -> score"))

        @override
        def __call__(self, question: str) -> Prediction:
            answer = self.predict1(question=question)
            return self.predict2(question=question, answer=answer)

    program = MyProgram()
    output = program(question="What is the capital of France?")
    lm_usage = output.get_lm_usage()
    assert len(lm_usage) == 1
    assert lm_usage[lm_for_test]["prompt_tokens"] > 0
    assert lm_usage[lm_for_test]["prompt_tokens"] > 0
    assert lm_usage[lm_for_test]["completion_tokens"] > 0
    assert lm_usage[lm_for_test]["total_tokens"] > 0


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Skip the test if OPENAI_API_KEY is not set.")
def test_usage_tracker_in_parallel(make_run):

    class MyProgram(Module):
        def __init__(self, lm):
            self.lm = lm
            self.predict1 = ChainOfThought(ts("question -> answer"))
            self.predict2 = ChainOfThought(ts("question, answer -> score"))

        async def aforward(self, question: str, *, run) -> Prediction:
            answer = await self.predict1(question=question, run=run)
            return await self.predict2(question=question, answer=answer, run=run)

    program1 = MyProgram(lm=LM("openai/gpt-4o-mini"))
    program2 = MyProgram(lm=LM("openai/gpt-3.5-turbo"))
    parallelizer = Parallel()
    run = make_run(lm=LM("openai/gpt-4o-mini"))
    results = asyncio.run(
        parallelizer(
            [
                (program1, {"question": "What is the meaning of life?"}),
                (program2, {"question": "why did a chicken cross the kitchen?"}),
            ],
            run=run,
        )
    )
    typed_results = cast("list[Prediction]", results)
    assert typed_results[0].get_lm_usage() is not None
    assert typed_results[1].get_lm_usage() is not None
    assert typed_results[0].get_lm_usage().keys() == {"openai/gpt-4o-mini"}
    assert typed_results[1].get_lm_usage().keys() == {"openai/gpt-3.5-turbo"}


@pytest.mark.asyncio
async def test_usage_tracker_async_parallel(make_run):
    program = Predict(QUESTION_ANSWER_TASK_SPEC)
    with patch("litellm.acompletion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'answer': 'Paris'}"))],
            usage=Usage(
                prompt_tokens=1117,
                completion_tokens=46,
                total_tokens=1163,
                prompt_tokens_details={"cached_tokens": 0, "audio_tokens": 0},
                completion_tokens_details={
                    "reasoning_tokens": 0,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            ),
            model="openai/gpt-4o-mini",
        )
        run = make_run(lm=LM("openai/gpt-4o-mini"), adapter=JSONAdapter(), telemetry=TelemetryConfig(track_usage=True))
        coroutines = [
            program(question="What is the capital of France?", run=run),
            program(question="What is the capital of France?", run=run),
            program(question="What is the capital of France?", run=run),
            program(question="What is the capital of France?", run=run),
        ]
        results = await asyncio.gather(*coroutines)
        assert results[0].get_lm_usage() is not None
        assert results[1].get_lm_usage() is not None
        lm_usage0 = results[0].get_lm_usage()["openai/gpt-4o-mini"]
        lm_usage1 = results[1].get_lm_usage()["openai/gpt-4o-mini"]
        assert lm_usage0["prompt_tokens"] == 1117
        assert lm_usage1["prompt_tokens"] == 1117
        assert lm_usage0["completion_tokens"] == 46
        assert lm_usage1["completion_tokens"] == 46
        assert lm_usage0["total_tokens"] == 1163
        assert lm_usage1["total_tokens"] == 1163


def test_usage_tracker_no_side_effect(make_run):

    class MyProgram(Module):
        def __init__(self):
            self.predict = Predict(QUESTION_ANSWER_TASK_SPEC)

        async def aforward(self, question: str, **kwargs: object) -> str:
            run = kwargs["run"]
            return (await self.predict(question=question, run=run)).answer

    program = MyProgram()
    run = make_run(lm=DummyLM([{"answer": "Paris"}]), telemetry=TelemetryConfig(track_usage=True))
    result = asyncio.run(program(question="What is the capital of France?", run=run))
    assert result == "Paris"


def test_module_history(make_run):

    class MyProgram(Module):
        def __init__(self):
            super().__init__()
            self.cot = ChainOfThought(ts("question -> answer"))

        async def aforward(self, question: str, **kwargs: object) -> Prediction:
            run = kwargs["run"]
            return await self.cot(question=question, run=run)

    with patch("litellm.acompletion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(message=Message(content="{'reasoning': 'Paris is the capital of France', 'answer': 'Paris'}"))
            ],
            model="openai/gpt-4o-mini",
        )
        run = make_run(lm=LM("openai/gpt-4o-mini"), adapter=JSONAdapter())
        program = MyProgram()
        asyncio.run(program(question="What is the capital of France?", run=run))
        asyncio.run(program.cot(question="What is the capital of France?", run=run))
        assert len(program.call_log) == 1
        assert len(program.cot.call_log) == 2
        assert len(program.cot.predict.call_log) == 2
        assert id(program.call_log[0]) == id(program.cot.call_log[0])
        assert program.call_log[0].outputs == ["{'reasoning': 'Paris is the capital of France', 'answer': 'Paris'}"]
        asyncio.run(program(question="What is the capital of France?", run=run))
        assert len(program.call_log) == 2
        assert len(program.cot.call_log) == 3
        assert len(program.cot.predict.call_log) == 3
        asyncio.run(program(question="What is the capital of France?", run=run))
        assert len(program.call_log) == 3
        assert len(program.cot.call_log) == 4
        assert len(program.cot.predict.call_log) == 4


def test_module_history_with_concurrency(make_run):

    class MyProgram(Module):
        def __init__(self):
            super().__init__()
            self.cot = ChainOfThought(ts("question -> answer"))

        async def aforward(self, question: str, **kwargs: object) -> Prediction:
            run = kwargs["run"]
            return await self.cot(question=question, run=run)

    with patch("litellm.acompletion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'reasoning': 'N/A', 'answer': 'Holy crab!'}"))],
            model="openai/gpt-4o-mini",
        )
        run = make_run(lm=LM("openai/gpt-4o-mini"), adapter=JSONAdapter())
        program = MyProgram()

        async def run_concurrent():
            await asyncio.gather(
                program(question="What is the meaning of life?", run=run),
                program(question="why did a chicken cross the kitchen?", run=run),
            )

        asyncio.run(run_concurrent())
        assert len(program.call_log) == 2
        assert len(program.cot.call_log) == 2
        assert len(program.cot.predict.call_log) == 2


@pytest.mark.asyncio
async def test_module_history_async(make_run):

    class MyProgram(Module):
        def __init__(self):
            super().__init__()
            self.cot = ChainOfThought(ts("question -> answer"))

        async def aforward(self, question: str, **kwargs: object) -> Prediction:
            run = kwargs["run"]
            return await self.cot(question=question, run=run)

    with patch("litellm.acompletion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(message=Message(content="{'reasoning': 'Paris is the capital of France', 'answer': 'Paris'}"))
            ],
            model="openai/gpt-4o-mini",
        )
        program = MyProgram()
        run = make_run(lm=LM("openai/gpt-4o-mini"), adapter=JSONAdapter())
        await program(question="What is the capital of France?", run=run)
        await program.cot(question="What is the capital of France?", run=run)
        assert len(program.call_log) == 1
        assert len(program.cot.call_log) == 2
        assert len(program.cot.predict.call_log) == 2
        assert id(program.call_log[0]) == id(program.cot.call_log[0])
        assert program.call_log[0].outputs == ["{'reasoning': 'Paris is the capital of France', 'answer': 'Paris'}"]
        run = make_run(
            lm=LM("openai/gpt-4o-mini"), adapter=JSONAdapter(), telemetry=TelemetryConfig(call_log=CallLogMode.off)
        )
        await program(question="What is the capital of France?", run=run)
        assert len(program.call_log) == 1
        assert len(program.cot.call_log) == 2
        assert len(program.cot.predict.call_log) == 2
        run = make_run(
            lm=LM("openai/gpt-4o-mini"),
            adapter=JSONAdapter(),
            telemetry=TelemetryConfig(call_log=CallLogMode.memory),
            init_run_log=False,
        )
        fresh_program = MyProgram()
        await fresh_program(question="What is the capital of France?", run=run)
        assert len(fresh_program.call_log) == 1
        assert len(fresh_program.cot.call_log) == 1
        assert len(fresh_program.cot.predict.call_log) == 1


def test_forward_direct_call_warning(caplog, make_run):

    class TestModule(Module):
        async def aforward(self, x, **kwargs: object):
            return x

    module = TestModule()
    with caplog.at_level(logging.WARNING, logger="dspy.primitives.module"):
        asyncio.run(module.aforward("test"))
    assert "directly is discouraged" in caplog.text


def test_forward_through_call_no_warning(capsys, make_run):

    class TestModule(Module):
        async def aforward(self, x, **kwargs: object):
            return x

    module = TestModule()
    run = make_run(lm=DummyLM([{}]))
    asyncio.run(module(x="test", run=run))
    captured = capsys.readouterr()
    assert "directly is discouraged" not in captured.err
