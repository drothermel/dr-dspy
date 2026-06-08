import asyncio
import copy
import enum
import logging
import time
import types
from datetime import datetime
from unittest.mock import patch

import orjson
import pydantic
import pytest

try:
    from litellm import ModelResponse
except ImportError:
    pytest.skip("litellm is not installed", allow_module_level=True)  # ty: ignore[too-many-positional-arguments]
from pydantic import BaseModel, HttpUrl

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types.history import History
from dspy.adapters.types.image import Image
from dspy.clients.base_lm import LM_CLASS_STATE_KEY, BaseLM
from dspy.clients.lm import LM
from dspy.dsp.utils.settings import settings
from dspy.predict.parallel import Parallel
from dspy.predict.predict import Predict, serialize_object
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature
from dspy.utils.dummies import DummyLM


class CustomStateLM(BaseLM):
    def __init__(self, model, *, deployment: str, **kwargs: object):
        super().__init__(model=model, **kwargs)  # ty:ignore[invalid-argument-type]
        self.deployment = deployment

    def dump_state(self):
        state = super().dump_state()
        state["deployment"] = self.deployment
        return state

    @classmethod
    def load_state(cls, state):  # ty:ignore[invalid-method-override]
        state = dict(state)
        state.pop(LM_CLASS_STATE_KEY, None)
        return cls(**state)


class OuterLMContainer:
    class InnerLM(BaseLM):
        pass


def test_initialization_with_string_signature():
    signature_string = "input1, input2 -> output"
    predict = Predict(signature_string)
    expected_instruction = "Given the fields `input1`, `input2`, produce the fields `output`."
    assert predict.signature.instructions == expected_instruction
    assert predict.signature.instructions == Signature(signature_string).instructions  # ty:ignore[too-many-positional-arguments, unresolved-attribute]


def test_reset_method():
    predict_instance = Predict("input -> output")
    predict_instance.lm = "modified"  # ty:ignore[invalid-assignment]
    predict_instance.traces = ["trace"]
    predict_instance.train = ["train"]
    predict_instance.demos = ["demo"]
    predict_instance.reset()
    assert predict_instance.lm is None
    assert predict_instance.traces == []
    assert predict_instance.train == []
    assert predict_instance.demos == []


def test_lm_after_dump_and_load_state():
    predict_instance = Predict("input -> output")
    lm = LM(
        model="openai/gpt-4o-mini",
        model_type="chat",
        temperature=1,
        max_tokens=100,
        num_retries=10,
    )
    predict_instance.lm = lm
    expected_lm_state = {
        LM_CLASS_STATE_KEY: "dspy.clients.lm.LM",
        "model": "openai/gpt-4o-mini",
        "model_type": "chat",
        "temperature": 1,
        "max_tokens": 100,
        "num_retries": 10,
        "cache": True,
        "finetuning_model": None,
        "launch_kwargs": {},
        "train_kwargs": {},
    }
    assert lm.dump_state() == expected_lm_state
    dumped_state = predict_instance.dump_state()
    new_instance = Predict("input -> output")
    new_instance.load_state(dumped_state)
    assert new_instance.lm.dump_state() == expected_lm_state  # ty:ignore[unresolved-attribute]


def test_base_lm_dump_state_ignores_internal_class_marker_kwarg():
    lm = CustomStateLM(model="custom-model", deployment="prod", **{LM_CLASS_STATE_KEY: "malicious.module.LM"})

    assert lm.dump_state()[LM_CLASS_STATE_KEY] == f"{CustomStateLM.__module__}.{CustomStateLM.__qualname__}"


def test_legacy_lm_state_without_class_marker_loads_as_lm():
    predict_instance = Predict("input -> output")
    predict_instance.lm = LM(model="openai/gpt-4o-mini", temperature=1, max_tokens=100)
    dumped_state = predict_instance.dump_state()
    dumped_state["lm"].pop(LM_CLASS_STATE_KEY)

    loaded_instance = Predict("input -> output").load_state(dumped_state)

    assert isinstance(loaded_instance.lm, LM)
    assert loaded_instance.lm.model == "openai/gpt-4o-mini"
    assert LM_CLASS_STATE_KEY in loaded_instance.lm.dump_state()


def test_custom_lm_load_state_requires_trusted_opt_in():
    predict_instance = Predict("input -> output")
    predict_instance.lm = CustomStateLM(model="custom-model", deployment="prod")
    dumped_state = predict_instance.dump_state()

    with pytest.raises(ValueError, match="Refusing to import custom serialized LM class"):
        Predict("input -> output").load_state(dumped_state)

    loaded_instance = Predict("input -> output").load_state(dumped_state, allow_unsafe_lm_state=True)

    assert isinstance(loaded_instance.lm, CustomStateLM)
    assert loaded_instance.lm.model == "custom-model"
    assert loaded_instance.lm.deployment == "prod"


def test_nested_custom_lm_class_path_loads_for_trusted_state():
    predict_instance = Predict("input -> output")
    predict_instance.lm = OuterLMContainer.InnerLM(model="nested-model")
    dumped_state = predict_instance.dump_state()

    loaded_instance = Predict("input -> output").load_state(dumped_state, allow_unsafe_lm_state=True)

    assert isinstance(loaded_instance.lm, OuterLMContainer.InnerLM)
    assert loaded_instance.lm.model == "nested-model"


def test_call_method():
    predict_instance = Predict("input -> output")
    lm = DummyLM([{"output": "test output"}])
    settings.configure(lm=lm)
    result = predict_instance(input="test input")
    assert result.output == "test output"


def test_instructions_after_dump_and_load_state():
    predict_instance = Predict(Signature("input -> output", "original instructions"))  # ty:ignore[invalid-argument-type, too-many-positional-arguments]
    dumped_state = predict_instance.dump_state()
    new_instance = Predict(Signature("input -> output", "new instructions"))  # ty:ignore[invalid-argument-type, too-many-positional-arguments]
    new_instance.load_state(dumped_state)
    assert new_instance.signature.instructions == "original instructions"


def test_demos_after_dump_and_load_state():
    class TranslateToEnglish(Signature):
        """Translate content from a language to English."""

        content: str = InputField()
        language: str = InputField()
        translation: str = OutputField()

    original_instance = Predict(TranslateToEnglish)
    original_instance.demos = [
        Example(
            content="¿Qué tal?",
            language="SPANISH",
            translation="Hello there",
        ).with_inputs("content", "language"),
    ]

    dumped_state = original_instance.dump_state()
    assert len(dumped_state["demos"]) == len(original_instance.demos)
    assert dumped_state["demos"][0]["content"] == original_instance.demos[0].content

    saved_state = orjson.dumps(dumped_state).decode()
    loaded_state = orjson.loads(saved_state)

    new_instance = Predict(TranslateToEnglish)
    new_instance.load_state(loaded_state)
    assert len(new_instance.demos) == len(original_instance.demos)
    # Demos don't need to keep the same types after saving and loading the state.
    assert new_instance.demos[0]["content"] == original_instance.demos[0].content


def test_typed_demos_after_dump_and_load_state():
    class Item(pydantic.BaseModel):
        name: str
        quantity: int

    class InventorySignature(Signature):
        """Handle inventory items and their translations."""

        items: list[Item] = InputField()
        language: str = InputField()
        translated_items: list[Item] = OutputField()
        total_quantity: int = OutputField()

    original_instance = Predict(InventorySignature)
    original_instance.demos = [
        Example(
            items=[Item(name="apple", quantity=5), Item(name="banana", quantity=3)],
            language="SPANISH",
            translated_items=[Item(name="manzana", quantity=5), Item(name="plátano", quantity=3)],
            total_quantity=8,
        ).with_inputs("items", "language"),
    ]

    # Test dump_state
    dumped_state = original_instance.dump_state()
    assert len(dumped_state["demos"]) == len(original_instance.demos)
    # Verify the input items were properly serialized
    assert isinstance(dumped_state["demos"][0]["items"], list)
    assert len(dumped_state["demos"][0]["items"]) == 2
    assert dumped_state["demos"][0]["items"][0] == {"name": "apple", "quantity": 5}

    # Test serialization/deserialization
    saved_state = orjson.dumps(dumped_state).decode()
    loaded_state = orjson.loads(saved_state)

    # Test load_state
    new_instance = Predict(InventorySignature)
    new_instance.load_state(loaded_state)
    assert len(new_instance.demos) == len(original_instance.demos)

    # Verify the structure is maintained after loading
    loaded_demo = new_instance.demos[0]
    assert isinstance(loaded_demo["items"], list)
    assert len(loaded_demo["items"]) == 2
    assert loaded_demo["items"][0]["name"] == "apple"
    assert loaded_demo["items"][0]["quantity"] == 5
    assert loaded_demo["items"][1]["name"] == "banana"
    assert loaded_demo["items"][1]["quantity"] == 3

    # Verify output items were also properly maintained
    assert isinstance(loaded_demo["translated_items"], list)
    assert len(loaded_demo["translated_items"]) == 2
    assert loaded_demo["translated_items"][0]["name"] == "manzana"
    assert loaded_demo["translated_items"][1]["name"] == "plátano"


def test_signature_fields_after_dump_and_load_state(tmp_path):
    class CustomSignature(Signature):
        """I am just an instruction."""

        sentence = InputField(desc="I am an innocent input!")
        sentiment = OutputField()

    file_path = tmp_path / "tmp.json"
    original_instance = Predict(CustomSignature)
    original_instance.save(file_path)

    class CustomSignature2(Signature):
        """I am not a pure instruction."""

        sentence = InputField(desc="I am a malicious input!")
        sentiment = OutputField(desc="I am a malicious output!", prefix="I am a prefix!")

    new_instance = Predict(CustomSignature2)
    assert new_instance.signature.dump_state() != original_instance.signature.dump_state()
    # After loading, the fields should be the same.
    new_instance.load(file_path)
    assert new_instance.signature.dump_state() == original_instance.signature.dump_state()


@pytest.mark.parametrize("filename", ["model.json", "model.pkl"])
def test_lm_field_after_dump_and_load_state(tmp_path, filename):
    file_path = tmp_path / filename
    lm = LM(
        model="openai/gpt-4o-mini",
        model_type="chat",
        temperature=1,
        max_tokens=100,
        num_retries=10,
    )
    original_predict = Predict("q->a")
    original_predict.lm = lm

    original_predict.save(file_path)

    assert file_path.exists()

    loaded_predict = Predict("q->a")
    loaded_predict.load(file_path, allow_pickle=True)

    assert original_predict.dump_state() == loaded_predict.dump_state()


@pytest.mark.parametrize("endpoint_override_key", ["api_base", "base_url"])
def test_load_ignores_serialized_endpoint_override_by_default(tmp_path, endpoint_override_key):
    file_path = tmp_path / "model.json"
    override_url = "http://override.local/v1"
    original_predict = Predict("q->a")
    original_predict.lm = LM(model="openai/gpt-4o-mini")
    original_predict.save(file_path)

    with open(file_path, "rb") as f:
        saved_state = orjson.loads(f.read())
    saved_state["lm"][endpoint_override_key] = override_url
    with open(file_path, "wb") as f:
        f.write(orjson.dumps(saved_state))

    with patch("dspy.predict.predict.logger.warning") as warning_mock:
        loaded_predict = Predict("q->a")
        loaded_predict.load(file_path)

    assert loaded_predict.lm is not None
    assert endpoint_override_key not in loaded_predict.lm.kwargs
    warning_mock.assert_called_once()
    assert warning_mock.call_args.args[1] == [endpoint_override_key]


@pytest.mark.parametrize("endpoint_override_key", ["api_base", "base_url"])
def test_load_allows_serialized_endpoint_override_with_opt_in(tmp_path, endpoint_override_key):
    file_path = tmp_path / "model.json"
    override_url = "http://override.local/v1"
    original_predict = Predict("q->a")
    original_predict.lm = LM(model="openai/gpt-4o-mini")
    original_predict.save(file_path)

    with open(file_path, "rb") as f:
        saved_state = orjson.loads(f.read())
    saved_state["lm"][endpoint_override_key] = override_url
    with open(file_path, "wb") as f:
        f.write(orjson.dumps(saved_state))

    with patch("dspy.predict.predict.logger.warning") as warning_mock:
        loaded_predict = Predict("q->a")
        loaded_predict.load(file_path, allow_unsafe_lm_state=True)

    assert loaded_predict.lm is not None
    assert loaded_predict.lm.kwargs[endpoint_override_key] == override_url
    warning_mock.assert_not_called()


@pytest.mark.parametrize("endpoint_override_key", ["api_base", "base_url"])
def test_load_state_ignores_serialized_endpoint_override_by_default(endpoint_override_key):
    override_url = "http://override.local/v1"
    original_predict = Predict("q->a")
    original_predict.lm = LM(model="openai/gpt-4o-mini")
    saved_state = copy.deepcopy(original_predict.dump_state())
    saved_state["lm"][endpoint_override_key] = override_url

    with patch("dspy.predict.predict.logger.warning") as warning_mock:
        loaded_predict = Predict("q->a")
        loaded_predict.load_state(saved_state)

    assert loaded_predict.lm is not None
    assert endpoint_override_key not in loaded_predict.lm.kwargs
    warning_mock.assert_called_once()
    assert warning_mock.call_args.args[1] == [endpoint_override_key]


@pytest.mark.parametrize("endpoint_override_key", ["api_base", "base_url"])
def test_load_state_allows_serialized_endpoint_override_with_opt_in(endpoint_override_key):
    override_url = "http://override.local/v1"
    original_predict = Predict("q->a")
    original_predict.lm = LM(model="openai/gpt-4o-mini")
    saved_state = copy.deepcopy(original_predict.dump_state())
    saved_state["lm"][endpoint_override_key] = override_url

    with patch("dspy.predict.predict.logger.warning") as warning_mock:
        loaded_predict = Predict("q->a")
        loaded_predict.load_state(saved_state, allow_unsafe_lm_state=True)

    assert loaded_predict.lm is not None
    assert loaded_predict.lm.kwargs[endpoint_override_key] == override_url
    warning_mock.assert_not_called()


def test_load_state_ignores_serialized_model_list_endpoint_override_by_default():
    override_url = "http://override.local/v1"
    original_predict = Predict("q->a")
    original_predict.lm = LM(model="openai/gpt-4o-mini")
    saved_state = copy.deepcopy(original_predict.dump_state())
    saved_state["lm"]["model_list"] = [
        {
            "model_name": "openai/gpt-4o-mini",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_base": override_url,
            },
        }
    ]

    with patch("dspy.predict.predict.logger.warning") as warning_mock:
        loaded_predict = Predict("q->a")
        loaded_predict.load_state(saved_state)

    assert loaded_predict.lm is not None
    assert "model_list" not in loaded_predict.lm.kwargs
    warning_mock.assert_called_once()
    assert "model_list" in warning_mock.call_args.args[1]


@pytest.mark.parametrize("endpoint_override_key", ["api_base", "base_url"])
def test_load_prevents_serialized_endpoint_override_reaching_litellm(tmp_path, endpoint_override_key):
    file_path = tmp_path / "model.json"
    override_url = "http://override.local/v1"
    original_predict = Predict("q->a")
    original_predict.lm = LM(model="openai/gpt-4o-mini")
    original_predict.save(file_path)

    with open(file_path, "rb") as f:
        saved_state = orjson.loads(f.read())
    saved_state["lm"][endpoint_override_key] = override_url
    with open(file_path, "wb") as f:
        f.write(orjson.dumps(saved_state))

    loaded_predict = Predict("q->a")
    loaded_predict.load(file_path)

    class FakeResp(dict):
        cache_hit = False
        usage = {}  # noqa: RUF012

        def __init__(self):
            super().__init__({"choices": []})

    with patch("litellm.completion", return_value=FakeResp()) as completion_mock:
        loaded_predict.lm.forward(prompt="hello", cache=False)  # ty:ignore[unresolved-attribute]

    assert completion_mock.call_count == 1
    assert completion_mock.call_args.kwargs.get(endpoint_override_key) != override_url


def test_load_blocks_serialized_model_list_unless_opted_in(tmp_path):
    file_path = tmp_path / "model.json"
    override_url = "http://override.local/v1"
    original_predict = Predict("q->a")
    original_predict.lm = LM(model="openai/gpt-4o-mini")
    original_predict.save(file_path)

    with open(file_path, "rb") as f:
        saved_state = orjson.loads(f.read())
    saved_state["lm"]["model_list"] = [
        {
            "model_name": "openai/gpt-4o-mini",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_base": override_url,
            },
        }
    ]
    with open(file_path, "wb") as f:
        f.write(orjson.dumps(saved_state))

    class FakeResp(dict):
        cache_hit = False
        usage = {}  # noqa: RUF012

        def __init__(self):
            super().__init__({"choices": []})

    safe_loaded_predict = Predict("q->a")
    safe_loaded_predict.load(file_path)
    with patch("litellm.batch_completion_models", return_value=FakeResp()) as batch_completion_mock:  # noqa: SIM117
        with patch("litellm.completion", return_value=FakeResp()) as completion_mock:
            safe_loaded_predict.lm.forward(prompt="hello", cache=False)  # ty:ignore[unresolved-attribute]

    assert completion_mock.called
    assert not batch_completion_mock.called

    opt_in_loaded_predict = Predict("q->a")
    opt_in_loaded_predict.load(file_path, allow_unsafe_lm_state=True)
    with patch("litellm.batch_completion_models", return_value=FakeResp()) as batch_completion_mock:
        opt_in_loaded_predict.lm.forward(prompt="hello", cache=False)  # ty:ignore[unresolved-attribute]

    opt_in_deployments = batch_completion_mock.call_args.kwargs["deployments"]
    assert opt_in_deployments[0]["api_base"] == override_url


def test_load_uses_env_api_key_without_honoring_serialized_endpoint_override(tmp_path, monkeypatch):
    file_path = tmp_path / "model.json"
    override_url = "http://override.local/v1"
    env_api_key = "sk-live-test-secret"

    original_predict = Predict("q->a")
    original_predict.lm = LM(model="openai/gpt-4o-mini", model_type="text")
    original_predict.save(file_path)

    with open(file_path, "rb") as f:
        saved_state = orjson.loads(f.read())
    assert "api_key" not in saved_state["lm"]
    saved_state["lm"]["api_base"] = override_url
    with open(file_path, "wb") as f:
        f.write(orjson.dumps(saved_state))

    monkeypatch.setenv("openai_API_KEY", env_api_key)

    class FakeResp(dict):
        cache_hit = False
        usage = {}  # noqa: RUF012

        def __init__(self):
            super().__init__({"choices": []})

    # Simulates legacy behavior by allowing serialized endpoint overrides.
    opt_in_loaded_predict = Predict("q->a")
    opt_in_loaded_predict.load(file_path, allow_unsafe_lm_state=True)
    with patch("litellm.text_completion", return_value=FakeResp()) as text_completion_mock:
        opt_in_loaded_predict.lm.forward(prompt="hello", cache=False)  # ty:ignore[unresolved-attribute]

    assert text_completion_mock.call_args.kwargs["api_base"] == override_url
    assert text_completion_mock.call_args.kwargs["api_key"] == env_api_key

    safe_loaded_predict = Predict("q->a")
    safe_loaded_predict.load(file_path)
    with patch("litellm.text_completion", return_value=FakeResp()) as text_completion_mock:
        safe_loaded_predict.lm.forward(prompt="hello", cache=False)  # ty:ignore[unresolved-attribute]

    # In the safe path, the key still comes from the environment, but the serialized endpoint override does not.
    assert text_completion_mock.call_args.kwargs["api_key"] == env_api_key
    assert text_completion_mock.call_args.kwargs["api_base"] != override_url


def test_forward_method():
    program = Predict("question -> answer")
    settings.configure(lm=DummyLM([{"answer": "No more responses"}]))
    result = program(question="What is 1+1?").answer
    assert result == "No more responses"


def test_forward_method2():
    program = Predict("question -> answer1, answer2")
    settings.configure(lm=DummyLM([{"answer1": "my first answer", "answer2": "my second answer"}]))
    result = program(question="What is 1+1?")
    assert result.answer1 == "my first answer"
    assert result.answer2 == "my second answer"


def test_config_management():
    predict_instance = Predict("input -> output")
    predict_instance.update_config(new_key="value")
    config = predict_instance.get_config()
    assert "new_key" in config
    assert config["new_key"] == "value"


def test_multi_output():
    program = Predict("question -> answer", n=2)
    settings.configure(lm=DummyLM([{"answer": "my first answer"}, {"answer": "my second answer"}]))
    results = program(question="What is 1+1?")
    assert results.completions.answer[0] == "my first answer"
    assert results.completions.answer[1] == "my second answer"


def test_multi_output2():
    program = Predict("question -> answer1, answer2", n=2)
    settings.configure(
        lm=DummyLM(
            [
                {"answer1": "my 0 answer", "answer2": "my 2 answer"},
                {"answer1": "my 1 answer", "answer2": "my 3 answer"},
            ],
        )
    )
    results = program(question="What is 1+1?")
    assert results.completions.answer1[0] == "my 0 answer"
    assert results.completions.answer1[1] == "my 1 answer"
    assert results.completions.answer2[0] == "my 2 answer"
    assert results.completions.answer2[1] == "my 3 answer"


def test_datetime_inputs_and_outputs():
    # Define a model for datetime inputs and outputs
    class TimedEvent(pydantic.BaseModel):
        event_name: str
        event_time: datetime

    class TimedSignature(Signature):
        events: list[TimedEvent] = InputField()
        summary: str = OutputField()
        next_event_time: datetime = OutputField()

    program = Predict(TimedSignature)

    lm = DummyLM(
        [
            {
                "reasoning": "Processed datetime inputs",
                "summary": "All events are processed",
                "next_event_time": "2024-11-27T14:00:00",
            }
        ]
    )
    settings.configure(lm=lm)

    output = program(
        events=[
            TimedEvent(event_name="Event 1", event_time=datetime(2024, 11, 25, 10, 0, 0)),
            TimedEvent(event_name="Event 2", event_time=datetime(2024, 11, 25, 15, 30, 0)),
        ]
    )
    assert output.summary == "All events are processed"
    assert output.next_event_time == datetime(2024, 11, 27, 14, 0, 0)


def test_explicitly_valued_enum_inputs_and_outputs():
    class Status(enum.Enum):
        PENDING = "pending"
        IN_PROGRESS = "in_progress"
        COMPLETED = "completed"

    class StatusSignature(Signature):
        current_status: Status = InputField()
        next_status: Status = OutputField()

    program = Predict(StatusSignature)

    lm = DummyLM(
        [
            {
                "reasoning": "The current status is 'PENDING', advancing to 'IN_PROGRESS'.",
                "next_status": "in_progress",
            }
        ]
    )
    settings.configure(lm=lm)

    output = program(current_status=Status.PENDING)
    assert output.next_status == Status.IN_PROGRESS


def test_enum_inputs_and_outputs_with_shared_names_and_values():
    class TicketStatus(enum.Enum):
        OPEN = "CLOSED"
        CLOSED = "RESOLVED"
        RESOLVED = "OPEN"

    class TicketStatusSignature(Signature):
        current_status: TicketStatus = InputField()
        next_status: TicketStatus = OutputField()

    program = Predict(TicketStatusSignature)

    # Mock reasoning and output
    lm = DummyLM(
        [
            {
                "reasoning": "The ticket is currently 'OPEN', transitioning to 'CLOSED'.",
                "next_status": "RESOLVED",  # Refers to TicketStatus.CLOSED by value
            }
        ]
    )
    settings.configure(lm=lm)

    output = program(current_status=TicketStatus.OPEN)
    assert output.next_status == TicketStatus.CLOSED  # By value


def test_auto_valued_enum_inputs_and_outputs():
    Status = enum.Enum("Status", ["PENDING", "IN_PROGRESS", "COMPLETED"])

    class StatusSignature(Signature):
        current_status: Status = InputField()
        next_status: Status = OutputField()

    program = Predict(StatusSignature)

    lm = DummyLM(
        [
            {
                "reasoning": "The current status is 'PENDING', advancing to 'IN_PROGRESS'.",
                "next_status": "IN_PROGRESS",  # Use the auto-assigned value for IN_PROGRESS
            }
        ]
    )
    settings.configure(lm=lm)

    output = program(current_status=Status.PENDING)
    assert output.next_status == Status.IN_PROGRESS


def test_named_predictors():
    class MyModule(Module):
        def __init__(self):
            super().__init__()
            self.inner = Predict("question -> answer")

    program = MyModule()
    assert program.named_predictors() == [("inner", program.inner)]

    # Check that it also works the second time.
    program2 = copy.deepcopy(program)
    assert program2.named_predictors() == [("inner", program2.inner)]


def test_output_only():
    class OutputOnlySignature(Signature):
        output = OutputField()

    predictor = Predict(OutputOnlySignature)

    lm = DummyLM([{"output": "short answer"}])
    settings.configure(lm=lm)
    assert predictor().output == "short answer"


def test_load_state_chaining():
    """Test that load_state returns self for chaining."""
    original = Predict("question -> answer")
    original.demos = [{"question": "test", "answer": "response"}]
    state = original.dump_state()

    new_instance = Predict("question -> answer").load_state(state)
    assert new_instance is not None
    assert new_instance.demos == original.demos


@pytest.mark.parametrize("adapter_type", ["chat", "json"])
def test_call_predict_with_chat_history(adapter_type):
    class SpyLM(LM):
        def __init__(self, *args: object, return_json=False, **kwargs: object):
            super().__init__(*args, **kwargs)  # ty:ignore[invalid-argument-type]
            self.calls = []
            self.return_json = return_json

        def __call__(self, prompt=None, messages=None, **kwargs: object):
            self.calls.append({"prompt": prompt, "messages": messages, "kwargs": kwargs})
            if self.return_json:
                return ["{'answer':'100%'}"]
            return ["[[ ## answer ## ]]\n100%!"]

    class MySignature(Signature):
        question: str = InputField()
        history: History = InputField()
        answer: str = OutputField()

    program = Predict(MySignature)

    if adapter_type == "chat":
        lm = SpyLM("dummy_model")
        settings.configure(adapter=ChatAdapter(), lm=lm)
    else:
        lm = SpyLM("dummy_model", return_json=True)
        settings.configure(adapter=JSONAdapter(), lm=lm)

    program(
        question="are you sure that's correct?",
        history=History(messages=[{"question": "what's the capital of france?", "answer": "paris"}]),
    )

    # Verify the LM was called with correct messages
    assert len(lm.calls) == 1
    messages = lm.calls[0]["messages"]

    assert len(messages) == 4

    assert "what's the capital of france?" in messages[1]["content"]
    assert "paris" in messages[2]["content"]
    assert "are you sure that's correct" in messages[3]["content"]


def test_lm_usage():
    program = Predict("question -> answer")
    settings.configure(lm=LM("openai/gpt-4o-mini", cache=False), track_usage=True)
    with patch(
        "dspy.clients.lm.litellm_completion",
        return_value=ModelResponse(
            choices=[{"message": {"content": "[[ ## answer ## ]]\nParis"}}],
            usage={"total_tokens": 10},
        ),
    ):
        result = program(question="What is the capital of France?")
        assert result.answer == "Paris"
        assert result.get_lm_usage()["openai/gpt-4o-mini"]["total_tokens"] == 10


def test_lm_usage_with_parallel():
    program = Predict("question -> answer")

    def program_wrapper(question):
        # Sleep to make it possible to cause a race condition
        time.sleep(0.5)
        return program(question=question)

    settings.configure(lm=LM("openai/gpt-4o-mini", cache=False), track_usage=True)
    with patch(
        "dspy.clients.lm.litellm_completion",
        return_value=ModelResponse(
            choices=[{"message": {"content": "[[ ## answer ## ]]\nParis"}}],
            usage={"total_tokens": 10},
        ),
    ):
        parallelizer = Parallel()
        input_pairs = [
            (program_wrapper, {"question": "What is the capital of France?"}),
            (program_wrapper, {"question": "What is the capital of France?"}),
        ]
        results = parallelizer(input_pairs)
        assert results[0].answer == "Paris"
        assert results[1].answer == "Paris"
        assert results[0].get_lm_usage()["openai/gpt-4o-mini"]["total_tokens"] == 10
        assert results[1].get_lm_usage()["openai/gpt-4o-mini"]["total_tokens"] == 10


@pytest.mark.asyncio
async def test_lm_usage_with_async():
    program = Predict("question -> answer")

    original_aforward = program.aforward

    async def patched_aforward(self, **kwargs: object):
        await asyncio.sleep(1)
        return await original_aforward(**kwargs)

    program.aforward = types.MethodType(patched_aforward, program)  # ty:ignore[invalid-assignment]

    with (
        settings.context(lm=LM("openai/gpt-4o-mini", cache=False), track_usage=True),
        patch(
            "litellm.acompletion",
            return_value=ModelResponse(
                choices=[{"message": {"content": "[[ ## answer ## ]]\nParis"}}],
                usage={"total_tokens": 10},
            ),
        ),
    ):
        coroutines = [
            program.acall(question="What is the capital of France?"),
            program.acall(question="What is the capital of France?"),
            program.acall(question="What is the capital of France?"),
            program.acall(question="What is the capital of France?"),
        ]
        results = await asyncio.gather(*coroutines)
        assert results[0].answer == "Paris"
        assert results[1].answer == "Paris"
        assert results[0].get_lm_usage()["openai/gpt-4o-mini"]["total_tokens"] == 10
        assert results[1].get_lm_usage()["openai/gpt-4o-mini"]["total_tokens"] == 10
        assert results[2].get_lm_usage()["openai/gpt-4o-mini"]["total_tokens"] == 10
        assert results[3].get_lm_usage()["openai/gpt-4o-mini"]["total_tokens"] == 10


def test_positional_arguments():
    program = Predict("question -> answer")
    with pytest.raises(ValueError) as e:  # noqa: PT011
        program("What is the capital of France?")
    assert str(e.value) == (
        "Positional arguments are not allowed when calling `dspy.predict.predict.Predict`, must use keyword arguments "
        "that match "
        "your signature input fields: 'question'. For example: `predict(question=input_value, ...)`."
    )


def test_error_message_on_invalid_lm_setup():
    # No LM is loaded.
    with pytest.raises(ValueError, match="No LM is loaded"):
        Predict("question -> answer")(question="Why did a chicken cross the kitchen?")

    # LM is a string.
    settings.configure(lm="openai/gpt-4o-mini")
    with pytest.raises(ValueError) as e:  # noqa: PT011
        Predict("question -> answer")(question="Why did a chicken cross the kitchen?")

    assert "LM must be an instance of `dspy.clients.base_lm.BaseLM`, not a string." in str(e.value)

    def dummy_lm():
        pass

    # LM is not an instance of BaseLM.
    settings.configure(lm=dummy_lm)
    with pytest.raises(ValueError) as e:  # noqa: PT011
        Predict("question -> answer")(question="Why did a chicken cross the kitchen?")
    assert "LM must be an instance of `dspy.clients.base_lm.BaseLM`, not <class 'function'>." in str(e.value)


@pytest.mark.parametrize("adapter_type", ["chat", "json"])
def test_field_constraints(adapter_type):
    class SpyLM(LM):
        def __init__(self, *args: object, return_json=False, **kwargs: object):
            super().__init__(*args, **kwargs)  # ty:ignore[invalid-argument-type]
            self.calls = []
            self.return_json = return_json

        def __call__(self, prompt=None, messages=None, **kwargs: object):
            self.calls.append({"prompt": prompt, "messages": messages, "kwargs": kwargs})
            if self.return_json:
                return ["{'score':'0.5', 'count':'2'}"]
            return ["[[ ## score ## ]]\n0.5\n[[ ## count ## ]]\n2"]

    class ConstrainedSignature(Signature):
        """Test signature with constrained fields."""

        # Input with length and value constraints
        text: str = InputField(min_length=5, max_length=100, desc="Input text")
        number: int = InputField(gt=0, lt=10, desc="A number between 0 and 10")

        # Output with multiple constraints
        score: float = OutputField(ge=0.0, le=1.0, desc="Score between 0 and 1")
        count: int = OutputField(multiple_of=2, desc="Even number count")

    program = Predict(ConstrainedSignature)
    lm = SpyLM("dummy_model")
    if adapter_type == "chat":
        lm = SpyLM("dummy_model")
        settings.configure(adapter=ChatAdapter(), lm=lm)
    else:
        lm = SpyLM("dummy_model", return_json=True)
        settings.configure(adapter=JSONAdapter(), lm=lm)

    # Call the predictor to trigger instruction generation
    program(text="hello world", number=5)

    # Get the system message containing the instructions
    system_message = lm.calls[0]["messages"][0]["content"]

    # Verify constraints are included in the field descriptions
    assert "minimum length: 5" in system_message
    assert "maximum length: 100" in system_message
    assert "greater than: 0" in system_message
    assert "less than: 10" in system_message
    assert "greater than or equal to: 0.0" in system_message
    assert "less than or equal to: 1.0" in system_message
    assert "a multiple of the given number: 2" in system_message


@pytest.mark.asyncio
async def test_async_predict():
    program = Predict("question -> answer")
    with settings.context(lm=DummyLM([{"answer": "Paris"}])):
        result = await program.acall(question="What is the capital of France?")
        assert result.answer == "Paris"


def test_predicted_outputs_piped_from_predict_to_lm_call():
    program = Predict("question -> answer")
    settings.configure(lm=LM("openai/gpt-4o-mini"))

    with patch("litellm.completion") as mock_completion:
        program(
            question="Why did a chicken cross the kitchen?",
            prediction={"type": "content", "content": "A chicken crossing the kitchen"},
        )

        assert mock_completion.call_args[1]["prediction"] == {
            "type": "content",
            "content": "A chicken crossing the kitchen",
        }

    # If the signature has prediction as an input field, and the prediction is not set as the standard predicted output
    # format, it should not be passed to the LM.
    program = Predict("question, prediction -> judgement")
    with patch("litellm.completion") as mock_completion:
        program(question="Why did a chicken cross the kitchen?", prediction="To get to the other side!")

    assert "prediction" not in mock_completion.call_args[1]


def test_dump_state_pydantic_non_primitive_types():
    class WebsiteInfo(BaseModel):
        name: str
        url: HttpUrl
        description: str | None = None
        created_at: datetime

    class TestSignature(Signature):
        website_info: WebsiteInfo = InputField()
        summary: str = OutputField()

    website_info = WebsiteInfo(
        name="Example",
        url="https://www.example.com",  # ty:ignore[invalid-argument-type]
        description="Test website",
        created_at=datetime(2021, 1, 1, 12, 0, 0),
    )

    serialized = serialize_object(website_info)

    assert serialized["url"] == "https://www.example.com/"
    assert serialized["created_at"] == "2021-01-01T12:00:00"

    json_str = orjson.dumps(serialized).decode()
    reloaded = orjson.loads(json_str)
    assert reloaded == serialized

    predictor = Predict(TestSignature)
    demo = {"website_info": website_info, "summary": "This is a test website."}
    predictor.demos = [demo]

    state = predictor.dump_state()
    json_str = orjson.dumps(state).decode()
    reloaded_state = orjson.loads(json_str)

    demo_data = reloaded_state["demos"][0]
    assert demo_data["website_info"]["url"] == "https://www.example.com/"
    assert demo_data["website_info"]["created_at"] == "2021-01-01T12:00:00"


def test_trace_size_limit():
    program = Predict("question -> answer")
    settings.configure(lm=DummyLM([{"answer": "Paris"}]), max_trace_size=3)

    for _ in range(10):
        program(question="What is the capital of France?")

    assert len(settings.trace) == 3


def test_disable_trace():
    program = Predict("question -> answer")
    settings.configure(lm=DummyLM([{"answer": "Paris"}]), trace=None)

    for _ in range(10):
        program(question="What is the capital of France?")

    assert settings.trace is None


def test_per_module_history_size_limit():
    program = Predict("question -> answer")
    settings.configure(lm=DummyLM([{"answer": "Paris"}]), max_history_size=5)

    for _ in range(10):
        program(question="What is the capital of France?")
    assert len(program.history) == 5


def test_per_module_history_disabled():
    program = Predict("question -> answer")
    settings.configure(lm=DummyLM([{"answer": "Paris"}]), disable_history=True)

    for _ in range(10):
        program(question="What is the capital of France?")
    assert len(program.history) == 0


def test_input_field_default_value():
    class SpyLM(LM):
        def __init__(self):
            super().__init__("dummy")
            self.calls = []

        def __call__(self, prompt=None, messages=None, **kwargs: object):
            self.calls.append({"messages": messages})
            return ["[[ ## answer ## ]]\ntest"]

    class SignatureWithDefault(Signature):
        context: str = InputField(default="DEFAULT_CONTEXT")
        question: str = InputField()
        answer: str = OutputField()

    lm = SpyLM()
    settings.configure(lm=lm)
    predictor = Predict(SignatureWithDefault)
    predictor(question="test")

    user_message = lm.calls[0]["messages"][-1]["content"]
    assert "DEFAULT_CONTEXT" in user_message


def log_test_helper():
    lm = DummyLM([{"answer": "test output"}])
    settings.configure(lm=lm)
    dspy_logger = logging.getLogger("dspy")
    dspy_logger.propagate = True


def test_extra_fields_warning(caplog):
    """Test that extra fields not in signature generate a warning."""
    log_test_helper()

    predict_instance = Predict("question -> answer")

    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(question="test", extra_field="should warn", another="also warn")

    # Check that warning was logged about extra fields
    assert "not in signature" in caplog.text
    assert "extra_field" in caplog.text


def test_missing_optional_input_field_no_warning(caplog):
    log_test_helper()

    class OptionalInputSignature(Signature):
        question: str = InputField()
        context: str | None = InputField()
        answer: str = OutputField()

    predict_instance = Predict(OptionalInputSignature)

    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(question="test")

    assert "Not all input fields were provided" not in caplog.text


def test_missing_required_input_field_still_warns(caplog):
    log_test_helper()

    class OptionalInputSignature(Signature):
        question: str = InputField()
        context: str | None = InputField()
        answer: str = OutputField()

    predict_instance = Predict(OptionalInputSignature)

    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance()

    assert "Not all input fields were provided" in caplog.text
    assert "Missing: ['question']" in caplog.text


def test_warning_images(caplog):
    """Test whether type mismatch for images generates a warning."""
    log_test_helper()

    predict_instance = Predict("question:Image -> answer")

    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(question=Image("https://example.com/image1.jpg"))

    assert "Type mismatch" not in caplog.text

    caplog.clear()

    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(question="dog_image")

    assert "Type mismatch for field 'question': expected Image" in caplog.text


def test_type_mismatch_warning(caplog):
    """Test that type mismatches in input fields generate a warning."""
    log_test_helper()

    class TypedSignature(Signature):
        count: int = InputField()
        name: str = InputField()
        result: str = OutputField()

    predict_instance = Predict(TypedSignature)
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)

    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        # Pass a string where int is expected
        predict_instance(count="not an int", name="test")

    assert "Type mismatch for field 'count': expected int" in caplog.text


def test_correct_types_no_warning(caplog):
    """Test that correct types don't generate warnings."""
    log_test_helper()

    class TypedSignature(Signature):
        count: int = InputField()
        name: str = InputField()
        result: str = OutputField()

    predict_instance = Predict(TypedSignature)
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        # Pass correct types
        predict_instance(count=42, name="test")

    assert "not in signature" not in caplog.text
    assert "Type mismatch" not in caplog.text


def test_list_type_validation(caplog):
    """Test type validation with list[str] types."""
    log_test_helper()

    class ComplexSignature(Signature):
        items: list[str] = InputField()
        result: str = OutputField()

    predict_instance = Predict(ComplexSignature)
    lm = DummyLM([{"result": "test output 1"}, {"result": "test output 2"}])
    settings.configure(lm=lm)

    # Test with wrong type
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(items="not a list")

    assert "Type mismatch for field 'items': expected list" in caplog.text

    # Test with correct type
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(items=["a", "b", "c"])

    assert "Type mismatch for field 'items'" not in caplog.text


def test_literal_type_validation(caplog):
    """Test type validation with Literal types."""
    from typing import Literal

    log_test_helper()

    class LiteralSignature(Signature):
        status: Literal["pending", "approved", "rejected"] = InputField()
        priority: Literal[1, 2, 3] = InputField()
        result: str = OutputField()

    predict_instance = Predict(LiteralSignature)

    # Test with correct literal values
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(status="approved", priority=2)

    assert "Type mismatch" not in caplog.text

    # Test with incorrect literal value for string
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(status="invalid", priority=2)

    assert "Type mismatch for field 'status': expected Literal['pending', 'approved', 'rejected']" in caplog.text

    # Test with incorrect literal value for int
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(status="approved", priority=5)

    assert "Type mismatch for field 'priority': expected Literal[1, 2, 3]" in caplog.text


def test_literal_union_type_validation(caplog):
    """Test type validation with Literal types in Union."""
    from typing import Literal

    log_test_helper()

    class UnionLiteralSignature(Signature):
        mode: Literal["auto", "manual"] | None = InputField()
        result: str = OutputField()

    predict_instance = Predict(UnionLiteralSignature)

    # Test with valid literal value
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(mode="auto")

    assert "Type mismatch" not in caplog.text

    # Test with None
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(mode=None)

    assert "Type mismatch" not in caplog.text

    # Test with invalid value
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(mode="invalid")

    assert "Type mismatch for field 'mode'" in caplog.text


def test_list_string(caplog):
    """Test passing list of strings."""
    log_test_helper()

    class TypedSignature(Signature):
        nameList: list[str] = InputField()  # noqa: N815
        result: str = OutputField()

    predict_instance = Predict(TypedSignature)
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)

    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        # Pass list of strings
        predict_instance(nameList=["Alice", "Bob", "Charlie"])

    assert "Type mismatch" not in caplog.text

    caplog.clear()

    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        # Pass list of non strings
        predict_instance(nameList=[1, 2, 3, None])

    assert "Type mismatch for field 'nameList': expected list[str]" in caplog.text


def test_nested_list_type_validation(caplog):
    """Test type validation with list element types."""
    log_test_helper()

    class NestedListSignature(Signature):
        numbers: list[int] = InputField()
        names: list[str] = InputField()
        result: str = OutputField()

    predict_instance = Predict(NestedListSignature)

    # Test with correct element types
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(numbers=[1, 2, 3], names=["alice", "bob"])

    # Should not have type warnings for correct element types
    assert "Type mismatch" not in caplog.text

    # Test with incorrect element types in numbers
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(numbers=["1", "2", "3"], names=["alice", "bob"])

    assert "Type mismatch for field 'numbers': expected list[int]" in caplog.text

    # Test with incorrect element types in names
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(numbers=[1, 2, 3], names=[1, 2, 3])

    assert "Type mismatch for field 'names': expected list[str]" in caplog.text

    # Test with empty list (should be valid)
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(numbers=[], names=[])

    assert "Type mismatch" not in caplog.text


def test_nested_dict_type_validation(caplog):
    """Test type validation with dict key and value types."""
    log_test_helper()

    class DictSignature(Signature):
        mapping: dict[str, int] = InputField()
        result: str = OutputField()

    predict_instance = Predict(DictSignature)

    # Test with correct key-value types
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(mapping={"a": 1, "b": 2, "c": 3})

    assert "Type mismatch" not in caplog.text

    # Test with incorrect value types
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(mapping={"a": "1", "b": "2", "c": "3"})

    assert "Type mismatch for field 'mapping': expected dict[str, int]" in caplog.text

    # Test with incorrect key types
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(mapping={1: 1, 2: 2, 3: 3})

    assert "Type mismatch for field 'mapping': expected dict[str, int]" in caplog.text


def test_nested_tuple_type_validation(caplog):
    """Test type validation with tuple types."""
    log_test_helper()

    class TupleSignature(Signature):
        fixed_tuple: tuple[str, int, bool] = InputField()
        var_tuple: tuple[int, ...] = InputField()
        result: str = OutputField()

    predict_instance = Predict(TupleSignature)

    # Test with correct tuple types
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(fixed_tuple=("hello", 42, True), var_tuple=(1, 2, 3, 4))

    assert "Type mismatch" not in caplog.text

    # Test with incorrect element types in fixed tuple
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(fixed_tuple=(123, 42, True), var_tuple=(1, 2, 3))

    assert "Type mismatch for field 'fixed_tuple': expected tuple[str, int, bool]" in caplog.text

    # Test with wrong length fixed tuple
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(fixed_tuple=("hello", 42), var_tuple=(1, 2, 3))

    assert "Type mismatch for field 'fixed_tuple': expected tuple[str, int, bool]" in caplog.text

    # Test with incorrect element types in variable tuple
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(fixed_tuple=("hello", 42, True), var_tuple=("a", "b", "c"))

    assert "Type mismatch for field 'var_tuple': expected tuple[int, ...]" in caplog.text


def test_literal_type_validation_string_signature(caplog):
    """Test type validation with Literal types using string signatures."""
    log_test_helper()

    # Use string signature with type annotations
    predict_instance = Predict("status:Literal['pending','approved','rejected'], priority:Literal[1,2,3] -> result")

    # Test with correct literal values
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(status="approved", priority=2)

    assert "Type mismatch" not in caplog.text

    # Test with incorrect literal value for string
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(status="invalid", priority=2)

    assert "Type mismatch for field 'status': expected Literal['pending', 'approved', 'rejected']" in caplog.text

    # Test with incorrect literal value for int
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(status="approved", priority=5)

    assert "Type mismatch for field 'priority': expected Literal[1, 2, 3]" in caplog.text


def test_list_type_validation_string_signature(caplog):
    """Test type validation with list element types using string signatures."""
    log_test_helper()

    # Use string signature with type annotations
    predict_instance = Predict("numbers:list[int], names:list[str] -> result")

    # Test with correct element types
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(numbers=[1, 2, 3], names=["alice", "bob"])

    assert "Type mismatch" not in caplog.text

    # Test with incorrect element types in numbers
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(numbers=["1", "2", "3"], names=["alice", "bob"])

    assert "Type mismatch for field 'numbers': expected list[int]" in caplog.text

    # Test with incorrect element types in names
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(numbers=[1, 2, 3], names=[1, 2, 3])

    assert "Type mismatch for field 'names': expected list[str]" in caplog.text

    # Test with empty list (should be valid)
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(numbers=[], names=[])

    assert "Type mismatch" not in caplog.text


def test_dict_type_validation_string_signature(caplog):
    """Test type validation with dict key and value types using string signatures."""
    log_test_helper()

    # Use string signature with type annotations
    predict_instance = Predict("mapping:dict[str,int] -> result")

    # Test with correct key-value types
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(mapping={"a": 1, "b": 2, "c": 3})

    assert "Type mismatch" not in caplog.text

    # Test with incorrect value types
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(mapping={"a": "1", "b": "2", "c": "3"})

    assert "Type mismatch for field 'mapping': expected dict[str, int]" in caplog.text

    # Test with incorrect key types
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(mapping={1: 1, 2: 2, 3: 3})

    assert "Type mismatch for field 'mapping': expected dict[str, int]" in caplog.text


def test_tuple_type_validation_string_signature(caplog):
    """Test type validation with tuple types using string signatures."""
    log_test_helper()

    # Use string signature with type annotations
    predict_instance = Predict("fixed_tuple:tuple[str,int,bool], var_tuple:tuple[int,...] -> result")

    # Test with correct tuple types
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(fixed_tuple=("hello", 42, True), var_tuple=(1, 2, 3, 4))

    assert "Type mismatch" not in caplog.text

    # Test with incorrect element types in fixed tuple
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(fixed_tuple=(123, 42, True), var_tuple=(1, 2, 3))

    assert "Type mismatch for field 'fixed_tuple': expected tuple[str, int, bool]" in caplog.text

    # Test with wrong length fixed tuple
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(fixed_tuple=("hello", 42), var_tuple=(1, 2, 3))

    assert "Type mismatch for field 'fixed_tuple': expected tuple[str, int, bool]" in caplog.text

    # Test with incorrect element types in variable tuple
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(fixed_tuple=("hello", 42, True), var_tuple=("a", "b", "c"))

    assert "Type mismatch for field 'var_tuple': expected tuple[int, ...]" in caplog.text


def test_union_type_validation_string_signature(caplog):
    """Test type validation with Union types using string signatures."""
    log_test_helper()

    # Use string signature with type annotations
    predict_instance = Predict("mode:Literal['auto','manual']|None -> result")

    # Test with valid literal value
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(mode="auto")

    assert "Type mismatch" not in caplog.text

    # Test with None
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(mode=None)

    assert "Type mismatch" not in caplog.text

    # Test with invalid value
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(mode="invalid")

    assert "Type mismatch for field 'mode'" in caplog.text


@pytest.mark.parametrize("enable_type_warnings", [False, True])
def test_basic_types_string_signature(caplog, enable_type_warnings):
    """Test type validation with basic types using string signatures."""
    log_test_helper()
    settings.configure(warn_on_type_mismatch=enable_type_warnings)
    # Use string signature with type annotations
    predict_instance = Predict("count:int, name:str -> result")

    # Test with correct types
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(count=42, name="test")

    assert "Type mismatch" not in caplog.text

    # Test with incorrect type for count
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(count="not an int", name="test")

    if enable_type_warnings:
        assert "Type mismatch for field 'count': expected int" in caplog.text
    else:
        assert "Type mismatch" not in caplog.text


def test_untyped_string_signature(caplog):
    """Test type validation with basic types using string signatures without type."""
    log_test_helper()

    # Use string signature without annotations
    predict_instance = Predict("count, name -> result")

    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        # Test with incorrect type for count and name
        predict_instance(count="abc", name=123)

    assert "Type mismatch" not in caplog.text


def test_untyped_class_signature(caplog):
    """Test type validation with basic types using class signature without type."""
    log_test_helper()

    # Use class signature with type annotations
    class TestSignature(Signature):
        count = InputField()
        name = InputField()
        result = OutputField()

    predict_instance = Predict(TestSignature)

    # Test with correct types
    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        # Test with "unexpected" type for count and name
        predict_instance(count="abc", name=123)

    assert "Type mismatch" not in caplog.text


def test_string_to_list_signature(caplog):
    """Test type validation with string input field type where the module gets called with a list."""
    log_test_helper()

    # Use class signature with type annotations
    class TestSignature(Signature):
        name: str = InputField()
        count = InputField()
        result = OutputField()

    predict_instance = Predict(TestSignature)

    caplog.clear()
    lm = DummyLM([{"result": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        # Test with a list of strings
        predict_instance(name=["abc", "def", "geh"], count=123)

    assert "Type mismatch" not in caplog.text


@pytest.mark.parametrize("enable_type_warnings", [False, True])
def test_custom_signature_types(caplog, enable_type_warnings):
    """Test type validation with custom signature types."""
    log_test_helper()
    settings.configure(warn_on_type_mismatch=enable_type_warnings)

    class MyContainer:
        class Query(pydantic.BaseModel):
            text: str

    signature = Signature("query: MyContainer.Query -> answer")  # ty:ignore[too-many-positional-arguments]
    predict_instance = Predict(signature)  # ty:ignore[invalid-argument-type]

    # Create an instance of the Query model
    query_instance = MyContainer.Query(text="What is the capital of France?")

    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        predict_instance(query=query_instance)

    assert "Type mismatch" not in caplog.text

    caplog.clear()
    lm = DummyLM([{"answer": "test output"}])
    settings.configure(lm=lm)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        # Test with an incorrect type
        predict_instance(query="What is the capital of France?")

    if enable_type_warnings:
        assert "Type mismatch for field 'query': expected Query" in caplog.text
    else:
        assert "Type mismatch" not in caplog.text
