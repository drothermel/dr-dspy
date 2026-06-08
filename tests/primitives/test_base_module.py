import asyncio
import logging
import os
import threading
from unittest.mock import patch

import pytest
from typing_extensions import override

try:
    from litellm import Choices, Message, ModelResponse
    from litellm.types.utils import Usage
except ImportError:
    pytest.skip("litellm is not installed", allow_module_level=True)  # ty: ignore[too-many-positional-arguments]

from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.lm import LM
from dspy.dsp.utils.settings import settings
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.parallel import Parallel
from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature
from dspy.task_spec import default_task_instructions
from dspy.task_spec.bridge import task_spec_from_signature
from dspy.teleprompt.bootstrap import BootstrapFewShot
from dspy.utils.dummies import DummyLM
from dspy.utils.saving import load
from tests.task_spec.helpers import ts

QA_TASK_SPEC = ts("question->answer", instructions=default_task_instructions(inputs=("question",), outputs=("answer",)))
QUESTION_ANSWER_TASK_SPEC = ts(
    "question -> answer",
    instructions=default_task_instructions(inputs=("question",), outputs=("answer",)),
)


def test_deepcopy_basic():
    cot = ChainOfThought(ts("q -> a"))  # ty:ignore[invalid-argument-type, too-many-positional-arguments]
    cot_copy = cot.deepcopy()
    assert len(cot.parameters()) == len(cot_copy.parameters())
    # Parameters should be different objects with the same values.
    assert id(cot.parameters()[0]) != id(cot_copy.parameters()[0])
    assert cot.parameters()[0].__dict__ == cot_copy.parameters()[0].__dict__


def test_deepcopy_with_uncopyable_modules():
    class CustomClass(Module):
        def __init__(self):
            self.lock = threading.Lock()  # Non-copyable object.
            self.cot = ChainOfThought(ts("q -> a"))  # ty:ignore[invalid-argument-type, too-many-positional-arguments]

    model = CustomClass()
    model_copy = model.deepcopy()
    assert len(model.parameters()) == len(model_copy.parameters())
    # The lock should be refer to the same object (shallow copy).
    assert id(model.lock) == id(model_copy.lock)
    # Parameters should be different objects with the same values.
    assert id(model.parameters()[0]) != id(model_copy.parameters()[0])
    assert model.parameters()[0].__dict__ == model_copy.parameters()[0].__dict__


def test_deepcopy_with_nested_modules():
    class CustomClass1(Module):
        def __init__(self):
            self.lock = threading.Lock()  # Non-copyable object.
            self.cot = ChainOfThought(ts("q -> a"))  # ty:ignore[invalid-argument-type, too-many-positional-arguments]

    class CustomClass2(Module):
        def __init__(self):
            self.submodel = CustomClass1()

    model = CustomClass2()
    model_copy = model.deepcopy()
    assert len(model.parameters()) == len(model_copy.parameters())
    # The lock should be refer to the same object (shallow copy).
    assert id(model.submodel.lock) == id(model_copy.submodel.lock)
    # Parameters should be different objects with the same values.
    assert id(model.parameters()[0]) != id(model_copy.parameters()[0])
    assert model.parameters()[0].__dict__ == model_copy.parameters()[0].__dict__


def test_save_and_load_with_json(tmp_path):
    model = ChainOfThought(ts("q -> a"))  # ty:ignore[invalid-argument-type, too-many-positional-arguments]
    model.predict.task_spec = model.predict.task_spec.with_instructions("You are a helpful assistant.")
    model.predict.demos = [
        Example(q="What is the capital of France?", a="Paris", reasoning="n/a").with_inputs("q"),
        # Nested example
        Example(
            q=[
                Example(q="What is the capital of France?"),
                Example(q="What is actually the capital of France?"),
            ],
            a="Paris",
            reasoning="n/a",
        ).with_inputs("q"),
    ]
    save_path = tmp_path / "model.json"
    model.save(save_path)
    new_model = ChainOfThought(ts("q -> a"))  # ty:ignore[invalid-argument-type, too-many-positional-arguments]
    new_model.load(save_path)

    assert new_model.predict.task_spec.equals(model.predict.task_spec)
    assert new_model.predict.demos[0] == model.predict.demos[0].to_dict()
    assert new_model.predict.demos[1] == model.predict.demos[1].to_dict()


@pytest.mark.extra
def test_save_and_load_with_pkl(tmp_path):
    import datetime

    # `datetime.date` is not json serializable, so we need to save with pickle.
    class MySignature(Signature):
        """Just a custom signature."""

        current_date: datetime.date = InputField()
        target_date: datetime.date = InputField()
        date_diff: int = OutputField(desc="The difference in days between the current_date and the target_date")

    trainset = [
        {"current_date": datetime.date(2024, 1, 1), "target_date": datetime.date(2024, 1, 2), "date_diff": 1},
        {"current_date": datetime.date(2024, 1, 1), "target_date": datetime.date(2024, 1, 3), "date_diff": 2},
        {"current_date": datetime.date(2024, 1, 1), "target_date": datetime.date(2024, 1, 4), "date_diff": 3},
        {"current_date": datetime.date(2024, 1, 1), "target_date": datetime.date(2024, 1, 5), "date_diff": 4},
        {"current_date": datetime.date(2024, 1, 1), "target_date": datetime.date(2024, 1, 6), "date_diff": 5},
    ]
    trainset = [Example(**example).with_inputs("current_date", "target_date") for example in trainset]

    settings.configure(
        lm=DummyLM([{"date_diff": "1", "reasoning": "n/a"}, {"date_diff": "2", "reasoning": "n/a"}] * 10)
    )

    cot = ChainOfThought(task_spec_from_signature(MySignature))
    asyncio.run(cot(current_date=datetime.date(2024, 1, 1), target_date=datetime.date(2024, 1, 2)))

    def dummy_metric(example, pred, trace=None):
        return True

    optimizer = BootstrapFewShot(max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=5, metric=dummy_metric)
    compiled_cot = asyncio.run(optimizer.compile(cot, trainset=trainset))
    compiled_cot.predict.task_spec = compiled_cot.predict.task_spec.with_instructions("You are a helpful assistant.")

    save_path = tmp_path / "program.pkl"
    compiled_cot.save(save_path)

    new_cot = ChainOfThought(task_spec_from_signature(MySignature))
    new_cot.load(save_path, allow_pickle=True)

    assert str(new_cot.predict.task_spec) == str(compiled_cot.predict.task_spec)
    assert new_cot.predict.demos == compiled_cot.predict.demos


def test_save_with_extra_modules(tmp_path):
    import sys

    # Create a temporary Python file with our custom module
    custom_module_path = tmp_path / "custom_module.py"
    with open(custom_module_path, "w") as f:
        f.write("""
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.primitives.module import Module
from dspy.signatures.signature import Signature

class MyModule(Module):
    def __init__(self):
        self.cot = ChainOfThought(ts("q -> a"))

    async def aforward(self, q):
        return await self.cot(q=q)
""")

    # Add the tmp_path to Python path so we can import the module
    sys.path.insert(0, str(tmp_path))
    try:
        import custom_module  # ty:ignore[unresolved-import]

        cot = custom_module.MyModule()

        cot.save(tmp_path, save_program=True)
        # Remove the custom module from sys.modules to simulate it not being available
        sys.modules.pop("custom_module", None)
        # Also remove it from sys.path
        sys.path.remove(str(tmp_path))
        del custom_module

        # Test the loading fails without using `modules_to_serialize`
        with pytest.raises(ModuleNotFoundError):
            load(tmp_path, allow_pickle=True)

        sys.path.insert(0, str(tmp_path))
        import custom_module  # ty:ignore[unresolved-import]

        cot.save(
            tmp_path,
            modules_to_serialize=[custom_module],
            save_program=True,
        )

        # Remove the custom module from sys.modules to simulate it not being available
        sys.modules.pop("custom_module", None)
        # Also remove it from sys.path
        sys.path.remove(str(tmp_path))
        del custom_module

        loaded_module = load(tmp_path, allow_pickle=True)
        assert loaded_module.cot.predict.task_spec == cot.cot.predict.task_spec

    finally:
        # Only need to clean up sys.path
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))


def test_load_with_version_mismatch(tmp_path):
    from dspy.primitives.base_module import logger

    # Mock versions during save
    save_versions = {"python": "3.9", "dspy": "2.4.0", "cloudpickle": "2.0"}

    # Mock versions during load
    load_versions = {"python": "3.10", "dspy": "2.5.0", "cloudpickle": "2.1"}

    predict = Predict(QA_TASK_SPEC)

    # Create a custom handler to capture log messages
    class ListHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages = []

        @override
        def emit(self, record):
            self.messages.append(record.getMessage())

    # Add handler and set level
    handler = ListHandler()
    original_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

    try:
        save_path = tmp_path / "program.pkl"
        # Mock version during save
        with patch("dspy.primitives.base_module.get_dependency_versions", return_value=save_versions):
            predict.save(save_path)

        # Mock version during load
        with patch("dspy.primitives.base_module.get_dependency_versions", return_value=load_versions):
            loaded_predict = Predict(QA_TASK_SPEC)
            loaded_predict.load(save_path, allow_pickle=True)

        # Assert warnings were logged: 1 for pickle loading + 3 for version mismatches
        assert len(handler.messages) == 4

        # First message is about pickle loading
        assert ".pkl" in handler.messages[0]

        # Rest are version mismatch warnings
        for msg in handler.messages[1:]:
            assert "There is a mismatch of" in msg

        # Verify the model still loads correctly despite version mismatches
        assert isinstance(loaded_predict, Predict)
        assert predict.task_spec.equals(loaded_predict.task_spec)

    finally:
        # Clean up: restore original level and remove handler
        logger.setLevel(original_level)
        logger.removeHandler(handler)


@pytest.mark.llm_call
def test_single_module_call_with_usage_tracker(lm_for_test):
    settings.configure(lm=LM(lm_for_test, cache=False, temperature=0.0), track_usage=True)

    predict = ChainOfThought(ts("question -> answer"))
    output = predict(question="What is the capital of France?")

    lm_usage = output.get_lm_usage()
    assert len(lm_usage) == 1
    assert lm_usage[lm_for_test]["prompt_tokens"] > 0
    assert lm_usage[lm_for_test]["completion_tokens"] > 0
    assert lm_usage[lm_for_test]["total_tokens"] > 0

    # Test no usage being tracked when cache is enabled
    settings.configure(lm=LM(lm_for_test, cache=True, temperature=0.0), track_usage=True)
    for _ in range(2):
        output = predict(question="What is the capital of France?")

    assert len(output.get_lm_usage()) == 0


@pytest.mark.llm_call
def test_multi_module_call_with_usage_tracker(lm_for_test):
    settings.configure(lm=LM(lm_for_test, cache=False, temperature=0.0), track_usage=True)

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


# TODO: Replace the live OpenAI dependency with a deterministic two-LM fixture before enabling this in CI.
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Skip the test if OPENAI_API_KEY is not set.")
def test_usage_tracker_in_parallel():
    class MyProgram(Module):
        def __init__(self, lm):
            self.lm = lm
            self.predict1 = ChainOfThought(ts("question -> answer"))
            self.predict2 = ChainOfThought(ts("question, answer -> score"))

        async def aforward(self, question: str) -> Prediction:
            with settings.context(lm=self.lm):
                answer = await self.predict1(question=question)
                return await self.predict2(question=question, answer=answer)

    settings.configure(track_usage=True)
    program1 = MyProgram(lm=LM("openai/gpt-4o-mini", cache=False))
    program2 = MyProgram(lm=LM("openai/gpt-3.5-turbo", cache=False))

    parallelizer = Parallel()

    results = asyncio.run(
        parallelizer(
            [
                (program1, {"question": "What is the meaning of life?"}),
                (program2, {"question": "why did a chicken cross the kitchen?"}),
            ]
        )
    )

    assert results[0].get_lm_usage() is not None
    assert results[1].get_lm_usage() is not None

    assert results[0].get_lm_usage().keys() == {"openai/gpt-4o-mini"}
    assert results[1].get_lm_usage().keys() == {"openai/gpt-3.5-turbo"}


@pytest.mark.asyncio
async def test_usage_tracker_async_parallel():
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

        coroutines = [
            program.acall(question="What is the capital of France?"),
            program.acall(question="What is the capital of France?"),
            program.acall(question="What is the capital of France?"),
            program.acall(question="What is the capital of France?"),
        ]
        with settings.context(lm=LM("openai/gpt-4o-mini", cache=False), track_usage=True, adapter=JSONAdapter()):
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


def test_usage_tracker_no_side_effect():
    class MyProgram(Module):
        def __init__(self):
            self.predict = Predict(QUESTION_ANSWER_TASK_SPEC)

        async def aforward(self, question: str, **kwargs: object) -> str:
            return (await self.predict(question=question)).answer

    program = MyProgram()
    with settings.context(lm=DummyLM([{"answer": "Paris"}]), track_usage=True):
        result = asyncio.run(program(question="What is the capital of France?"))
    assert result == "Paris"


def test_module_history():
    class MyProgram(Module):
        def __init__(self, **kwargs: object):
            super().__init__(**kwargs)
            self.cot = ChainOfThought(ts("question -> answer"))

        async def aforward(self, question: str, **kwargs: object) -> Prediction:
            return await self.cot(question=question)

    with patch("litellm.acompletion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(message=Message(content="{'reasoning': 'Paris is the capital of France', 'answer': 'Paris'}"))
            ],
            model="openai/gpt-4o-mini",
        )
        settings.configure(lm=LM("openai/gpt-4o-mini", cache=False), adapter=JSONAdapter())
        program = MyProgram()
        asyncio.run(program(question="What is the capital of France?"))

        # Second call only call the submodule.
        asyncio.run(program.cot(question="What is the capital of France?"))

        # The LM history entity exists in all the ancestor callers.
        assert len(program.history) == 1
        assert len(program.cot.history) == 2
        assert len(program.cot.predict.history) == 2

        # The same history entity is shared across all the ancestor callers to reduce memory usage.
        assert id(program.history[0]) == id(program.cot.history[0])

        assert program.history[0].outputs == ["{'reasoning': 'Paris is the capital of France', 'answer': 'Paris'}"]

        settings.configure(disable_history=True)

        asyncio.run(program(question="What is the capital of France?"))
        # No history is recorded when history is disabled.
        assert len(program.history) == 1
        assert len(program.cot.history) == 2
        assert len(program.cot.predict.history) == 2

        settings.configure(disable_history=False)

        asyncio.run(program(question="What is the capital of France?"))
        # History is recorded again when history is enabled.
        assert len(program.history) == 2
        assert len(program.cot.history) == 3
        assert len(program.cot.predict.history) == 3


def test_module_history_with_concurrency():
    class MyProgram(Module):
        def __init__(self):
            super().__init__()
            self.cot = ChainOfThought(ts("question -> answer"))

        async def aforward(self, question: str, **kwargs: object) -> Prediction:
            return await self.cot(question=question)

    with patch("litellm.acompletion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'reasoning': 'N/A', 'answer': 'Holy crab!'}"))],
            model="openai/gpt-4o-mini",
        )
        settings.configure(lm=LM("openai/gpt-4o-mini", cache=False), adapter=JSONAdapter())
        program = MyProgram()

        async def run_concurrent():
            await asyncio.gather(
                program(question="What is the meaning of life?"),
                program(question="why did a chicken cross the kitchen?"),
            )

        asyncio.run(run_concurrent())
        assert len(program.history) == 2
        assert len(program.cot.history) == 2
        assert len(program.cot.predict.history) == 2


@pytest.mark.asyncio
async def test_module_history_async():
    class MyProgram(Module):
        def __init__(self, **kwargs: object):
            super().__init__(**kwargs)
            self.cot = ChainOfThought(ts("question -> answer"))

        async def aforward(self, question: str, **kwargs: object) -> Prediction:
            return await self.cot.acall(question=question)

    with patch("litellm.acompletion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(message=Message(content="{'reasoning': 'Paris is the capital of France', 'answer': 'Paris'}"))
            ],
            model="openai/gpt-4o-mini",
        )
        program = MyProgram()
        with settings.context(lm=LM("openai/gpt-4o-mini", cache=False), adapter=JSONAdapter()):
            await program.acall(question="What is the capital of France?")

            # Second call only call the submodule.
            await program.cot.acall(question="What is the capital of France?")

        # The LM history entity exists in all the ancestor callers.
        assert len(program.history) == 1
        assert len(program.cot.history) == 2
        assert len(program.cot.predict.history) == 2

        # The same history entity is shared across all the ancestor callers to reduce memory usage.
        assert id(program.history[0]) == id(program.cot.history[0])

        assert program.history[0].outputs == ["{'reasoning': 'Paris is the capital of France', 'answer': 'Paris'}"]

        with settings.context(disable_history=True, lm=LM("openai/gpt-4o-mini", cache=False), adapter=JSONAdapter()):
            await program.acall(question="What is the capital of France?")

        # No history is recorded when history is disabled.
        assert len(program.history) == 1
        assert len(program.cot.history) == 2
        assert len(program.cot.predict.history) == 2

        with settings.context(disable_history=False, lm=LM("openai/gpt-4o-mini", cache=False), adapter=JSONAdapter()):
            await program.acall(question="What is the capital of France?")
        # History is recorded again when history is enabled.
        assert len(program.history) == 2
        assert len(program.cot.history) == 3
        assert len(program.cot.predict.history) == 3


def test_forward_direct_call_warning(caplog):
    class TestModule(Module):
        async def aforward(self, x):
            return x

    module = TestModule()
    with caplog.at_level(logging.WARNING, logger="dspy.primitives.module"):
        asyncio.run(module.aforward("test"))
    assert "directly is discouraged" in caplog.text


def test_forward_through_call_no_warning(capsys):
    class TestModule(Module):
        async def aforward(self, x):
            return x

    module = TestModule()
    asyncio.run(module(x="test"))
    captured = capsys.readouterr()
    assert "directly is discouraged" not in captured.err
