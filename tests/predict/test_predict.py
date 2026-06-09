import asyncio
import copy
import enum
import logging
import types
from datetime import datetime
from typing import Any, ClassVar, cast
from unittest.mock import AsyncMock, patch

import orjson
import pydantic
import pytest
from typing_extensions import override

from dspy.runtime import CallLogMode, TelemetryConfig

try:
    from litellm import ModelResponse
except ImportError:
    pytest.skip(reason="litellm is not installed", allow_module_level=True)
from pydantic import BaseModel, HttpUrl

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types.image import Image
from dspy.clients.base_lm import LM_CLASS_STATE_KEY, PROVIDER_OPTIONS_STATE_KEY, BaseLM
from dspy.clients.lm import LM
from dspy.core.types import LMConfig, LMRequest, PredictOptions
from dspy.history import TurnLog
from dspy.predict.parallel import Parallel
from dspy.predict.predict import Predict
from dspy.primitives import Example, Module
from dspy.serialization.json import to_jsonable
from dspy.task_spec import TaskSpec, default_task_instructions, input_field, make_task_spec, output_field
from dspy.testing import DummyLM
from tests.test_utils.spy_lm import SpyLM


def _field_names(spec_part: str) -> tuple[str, ...]:
    names: list[str] = []
    for chunk in spec_part.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        names.append(chunk.split(":")[0].strip())
    return tuple(names)


def pspec(spec: str, *, instructions: str | None = None, **kwargs) -> TaskSpec:
    if instructions is None:
        inputs_str, outputs_str = spec.split("->", 1)
        instructions = default_task_instructions(inputs=_field_names(inputs_str), outputs=_field_names(outputs_str))
    return make_task_spec(spec, instructions=instructions, **kwargs)


class InventoryItem(pydantic.BaseModel):
    name: str
    quantity: int


class CustomStateLM(BaseLM):
    def __init__(self, model: str, *, deployment: str, **kwargs: Any):
        super().__init__(model=model, **cast("Any", kwargs))
        self.deployment = deployment

    @override
    def dump_state(self):
        state = super().dump_state()
        state["deployment"] = self.deployment
        return state

    @classmethod
    @override
    def load_state(cls, state: dict[str, Any], *, allow_custom_lm_class: bool = False):
        state = dict(state)
        state.pop(LM_CLASS_STATE_KEY, None)
        state.pop(PROVIDER_OPTIONS_STATE_KEY, None)
        _ = allow_custom_lm_class
        deployment = state.pop("deployment")
        return cls(model=state.pop("model"), deployment=deployment, **state)


class OuterLMContainer:
    class InnerLM(BaseLM):
        pass


def test_initialization_with_string_signature():
    signature_string = "input1, input2 -> output"
    with pytest.raises(TypeError, match="TaskSpec instance, not a string"):
        Predict(cast("Any", signature_string))
    expected_instruction = "Given the fields `input1`, `input2`, produce the fields `output`."
    predict = Predict(pspec(signature_string))
    assert predict.task_spec.instructions == expected_instruction


def test_reset_method(make_run):
    predict_instance = Predict(pspec("input -> output"))
    cast("Any", predict_instance).lm = "modified"
    predict_instance.demos = ["demo"]
    predict_instance.reset()
    assert predict_instance.lm is None
    assert predict_instance.demos == []


def test_lm_after_dump_and_load_state(make_run):
    predict_instance = Predict(pspec("input -> output"))
    lm = LM(model="openai/gpt-4o-mini", model_type="chat", temperature=1, max_tokens=100, num_retries=10)
    predict_instance.lm = lm
    expected_lm_state = {
        LM_CLASS_STATE_KEY: "dspy.clients.lm.LM",
        "model": "openai/gpt-4o-mini",
        "model_type": "chat",
        "temperature": 1,
        "max_tokens": 100,
        "num_retries": 10,
        PROVIDER_OPTIONS_STATE_KEY: {"extensions": {}},
    }
    assert lm.dump_state() == expected_lm_state
    dumped_state = predict_instance.dump_state()
    new_instance = Predict(pspec("input -> output"))
    new_instance.load_state(dumped_state)
    assert new_instance.lm is not None
    assert new_instance.lm.dump_state() == expected_lm_state


def test_base_lm_dump_state_ignores_internal_class_marker_kwarg(make_run):
    lm = CustomStateLM(model="custom-model", deployment="prod")
    lm.kwargs[LM_CLASS_STATE_KEY] = "malicious.module.LM"
    assert lm.dump_state()[LM_CLASS_STATE_KEY] == f"{CustomStateLM.__module__}.{CustomStateLM.__qualname__}"


def test_legacy_lm_state_without_class_marker_loads_as_lm(make_run):
    predict_instance = Predict(pspec("input -> output"))
    predict_instance.lm = LM(model="openai/gpt-4o-mini", temperature=1, max_tokens=100)
    dumped_state = predict_instance.dump_state()
    dumped_state["lm"].pop(LM_CLASS_STATE_KEY)
    loaded_instance = Predict(pspec("input -> output")).load_state(dumped_state)
    assert isinstance(loaded_instance.lm, LM)
    assert loaded_instance.lm.model == "openai/gpt-4o-mini"
    assert LM_CLASS_STATE_KEY in loaded_instance.lm.dump_state()


def test_custom_lm_load_state_requires_trusted_opt_in(make_run):
    predict_instance = Predict(pspec("input -> output"))
    predict_instance.lm = CustomStateLM(model="custom-model", deployment="prod")
    dumped_state = predict_instance.dump_state()
    with pytest.raises(ValueError, match="Refusing to import custom serialized LM class"):
        Predict(pspec("input -> output")).load_state(dumped_state)
    loaded_instance = Predict(pspec("input -> output")).load_state(dumped_state, allow_unsafe_lm_state=True)
    assert isinstance(loaded_instance.lm, CustomStateLM)
    assert loaded_instance.lm.model == "custom-model"
    assert loaded_instance.lm.deployment == "prod"


def test_nested_custom_lm_class_path_loads_for_trusted_state(make_run):
    predict_instance = Predict(pspec("input -> output"))
    predict_instance.lm = OuterLMContainer.InnerLM(model="nested-model")
    dumped_state = predict_instance.dump_state()
    loaded_instance = Predict(pspec("input -> output")).load_state(dumped_state, allow_unsafe_lm_state=True)
    assert isinstance(loaded_instance.lm, OuterLMContainer.InnerLM)
    assert loaded_instance.lm.model == "nested-model"


def test_call_method(make_run):
    predict_instance = Predict(pspec("input -> output"))
    lm = DummyLM([{"output": "test output"}])
    run = make_run(lm=lm)
    result = asyncio.run(predict_instance(input="test input", run=run))
    assert result.output == "test output"


def test_instructions_after_dump_and_load_state():
    predict_instance = Predict(pspec("input -> output", instructions="original instructions"))
    dumped_state = predict_instance.dump_state()
    new_instance = Predict(pspec("input -> output", instructions="new instructions"))
    new_instance.load_state(dumped_state)
    assert new_instance.task_spec.instructions == "original instructions"


def test_demos_after_dump_and_load_state():
    TranslateToEnglish = make_task_spec(
        {
            "content": input_field("content", desc="The content."),
            "language": input_field("language", desc="The language."),
            "translation": output_field("translation", desc="The translation."),
        },
        instructions="Translate content from a language to English.",
        name="TranslateToEnglish",
    )
    original_instance = Predict(TranslateToEnglish)
    original_instance.demos = [
        Example.from_record(
            {"content": "¿Qué tal?", "language": "SPANISH", "translation": "Hello there"},
            input_keys=("content", "language"),
        )
    ]
    dumped_state = original_instance.dump_state()
    assert len(dumped_state["demos"]) == len(original_instance.demos)
    assert dumped_state["demos"][0]["content"] == original_instance.demos[0].content
    saved_state = orjson.dumps(dumped_state).decode()
    loaded_state = orjson.loads(saved_state)
    new_instance = Predict(TranslateToEnglish)
    new_instance.load_state(loaded_state)
    assert len(new_instance.demos) == len(original_instance.demos)
    assert new_instance.demos[0]["content"] == original_instance.demos[0].content


def test_typed_demos_after_dump_and_load_state():
    InventorySignature = make_task_spec(
        {
            "items": input_field("items", type_=list[InventoryItem], desc="The items."),
            "language": input_field("language", desc="The language."),
            "translated_items": output_field(
                "translated_items", type_=list[InventoryItem], desc="The translated items."
            ),
            "total_quantity": output_field("total_quantity", type_=int, desc="The total quantity."),
        },
        instructions="Handle inventory items and their translations.",
        name="InventorySignature",
    )
    original_instance = Predict(InventorySignature)
    original_instance.demos = [
        Example.from_record(
            {
                "items": [InventoryItem(name="apple", quantity=5), InventoryItem(name="banana", quantity=3)],
                "language": "SPANISH",
                "translated_items": [
                    InventoryItem(name="manzana", quantity=5),
                    InventoryItem(name="plátano", quantity=3),
                ],
                "total_quantity": 8,
            },
            input_keys=("items", "language"),
        )
    ]
    dumped_state = original_instance.dump_state()
    assert len(dumped_state["demos"]) == len(original_instance.demos)
    assert isinstance(dumped_state["demos"][0]["items"], list)
    assert len(dumped_state["demos"][0]["items"]) == 2
    assert dumped_state["demos"][0]["items"][0] == {"name": "apple", "quantity": 5}
    saved_state = orjson.dumps(dumped_state).decode()
    loaded_state = orjson.loads(saved_state)
    new_instance = Predict(InventorySignature)
    item_type_key = f"{InventoryItem.__module__}.{InventoryItem.__qualname__}"
    new_instance.load_state(loaded_state, custom_types={item_type_key: InventoryItem})
    assert len(new_instance.demos) == len(original_instance.demos)
    loaded_demo = new_instance.demos[0]
    assert isinstance(loaded_demo["items"], list)
    assert len(loaded_demo["items"]) == 2
    assert loaded_demo["items"][0]["name"] == "apple"
    assert loaded_demo["items"][0]["quantity"] == 5
    assert loaded_demo["items"][1]["name"] == "banana"
    assert loaded_demo["items"][1]["quantity"] == 3
    assert isinstance(loaded_demo["translated_items"], list)
    assert len(loaded_demo["translated_items"]) == 2
    assert loaded_demo["translated_items"][0]["name"] == "manzana"
    assert loaded_demo["translated_items"][1]["name"] == "plátano"


def test_signature_fields_after_dump_and_load_state(tmp_path):
    CustomSignature = make_task_spec(
        {
            "sentence": input_field("sentence", desc="I am an innocent input!"),
            "sentiment": output_field("sentiment", desc="The sentiment."),
        },
        instructions="I am just an instruction.",
        name="CustomSignature",
    )
    file_path = tmp_path / "tmp.json"
    original_instance = Predict(CustomSignature)
    original_instance.save(file_path)
    CustomSignature2 = make_task_spec(
        {
            "sentence": input_field("sentence", desc="I am a malicious input!"),
            "sentiment": output_field("sentiment", desc="I am a malicious output!"),
        },
        instructions="I am not a pure instruction.",
        name="CustomSignature2",
    )
    new_instance = Predict(CustomSignature2)
    assert new_instance.task_spec.to_dict() != original_instance.task_spec.to_dict()
    new_instance.load(file_path)
    assert new_instance.task_spec.to_dict() == original_instance.task_spec.to_dict()


@pytest.mark.parametrize("filename", ["model.json", "model.pkl"])
def test_lm_field_after_dump_and_load_state(tmp_path, filename):
    file_path = tmp_path / filename
    lm = LM(model="openai/gpt-4o-mini", model_type="chat", temperature=1, max_tokens=100, num_retries=10)
    original_predict = Predict(pspec("q->a"))
    original_predict.lm = lm
    original_predict.save(file_path)
    assert file_path.exists()
    loaded_predict = Predict(pspec("q->a"))
    loaded_predict.load(file_path, allow_pickle=True)
    assert original_predict.dump_state() == loaded_predict.dump_state()


@pytest.mark.parametrize("endpoint_override_key", ["api_base", "base_url"])
def test_load_ignores_serialized_endpoint_override_by_default(tmp_path, endpoint_override_key):
    file_path = tmp_path / "model.json"
    override_url = "http://override.local/v1"
    original_predict = Predict(pspec("q->a"))
    original_predict.lm = LM(model="openai/gpt-4o-mini")
    original_predict.save(file_path)
    with open(file_path, "rb") as f:
        saved_state = orjson.loads(f.read())
    saved_state["lm"][endpoint_override_key] = override_url
    with open(file_path, "wb") as f:
        f.write(orjson.dumps(saved_state))
    with patch("dspy.predict.predict.logger.warning") as warning_mock:
        loaded_predict = Predict(pspec("q->a"))
        loaded_predict.load(file_path)
    assert loaded_predict.lm is not None
    assert endpoint_override_key not in loaded_predict.lm.kwargs
    warning_mock.assert_called_once()
    assert warning_mock.call_args.args[1] == [endpoint_override_key]


@pytest.mark.parametrize("endpoint_override_key", ["api_base", "base_url"])
def test_load_allows_serialized_endpoint_override_with_opt_in(tmp_path, endpoint_override_key):
    file_path = tmp_path / "model.json"
    override_url = "http://override.local/v1"
    original_predict = Predict(pspec("q->a"))
    original_predict.lm = LM(model="openai/gpt-4o-mini")
    original_predict.save(file_path)
    with open(file_path, "rb") as f:
        saved_state = orjson.loads(f.read())
    saved_state["lm"][endpoint_override_key] = override_url
    with open(file_path, "wb") as f:
        f.write(orjson.dumps(saved_state))
    with patch("dspy.predict.predict.logger.warning") as warning_mock:
        loaded_predict = Predict(pspec("q->a"))
        loaded_predict.load(file_path, allow_unsafe_lm_state=True)
    assert loaded_predict.lm is not None
    assert loaded_predict.lm.kwargs[endpoint_override_key] == override_url
    warning_mock.assert_not_called()


@pytest.mark.parametrize("endpoint_override_key", ["api_base", "base_url"])
def test_load_state_ignores_serialized_endpoint_override_by_default(endpoint_override_key, make_run):
    override_url = "http://override.local/v1"
    original_predict = Predict(pspec("q->a"))
    original_predict.lm = LM(model="openai/gpt-4o-mini")
    saved_state = copy.deepcopy(original_predict.dump_state())
    saved_state["lm"][endpoint_override_key] = override_url
    with patch("dspy.predict.predict.logger.warning") as warning_mock:
        loaded_predict = Predict(pspec("q->a"))
        loaded_predict.load_state(saved_state)
    assert loaded_predict.lm is not None
    assert endpoint_override_key not in loaded_predict.lm.kwargs
    warning_mock.assert_called_once()
    assert warning_mock.call_args.args[1] == [endpoint_override_key]


@pytest.mark.parametrize("endpoint_override_key", ["api_base", "base_url"])
def test_load_state_allows_serialized_endpoint_override_with_opt_in(endpoint_override_key, make_run):
    override_url = "http://override.local/v1"
    original_predict = Predict(pspec("q->a"))
    original_predict.lm = LM(model="openai/gpt-4o-mini")
    saved_state = copy.deepcopy(original_predict.dump_state())
    saved_state["lm"][endpoint_override_key] = override_url
    with patch("dspy.predict.predict.logger.warning") as warning_mock:
        loaded_predict = Predict(pspec("q->a"))
        loaded_predict.load_state(saved_state, allow_unsafe_lm_state=True)
    assert loaded_predict.lm is not None
    assert loaded_predict.lm.kwargs[endpoint_override_key] == override_url
    warning_mock.assert_not_called()


def test_load_state_ignores_serialized_model_list_endpoint_override_by_default(make_run):
    override_url = "http://override.local/v1"
    original_predict = Predict(pspec("q->a"))
    original_predict.lm = LM(model="openai/gpt-4o-mini")
    saved_state = copy.deepcopy(original_predict.dump_state())
    saved_state["lm"]["model_list"] = [
        {
            "model_name": "openai/gpt-4o-mini",
            "litellm_params": {"model": "openai/gpt-4o-mini", "api_base": override_url},
        }
    ]
    with patch("dspy.predict.predict.logger.warning") as warning_mock:
        loaded_predict = Predict(pspec("q->a"))
        loaded_predict.load_state(saved_state)
    assert loaded_predict.lm is not None
    assert "model_list" not in loaded_predict.lm.kwargs
    warning_mock.assert_called_once()
    assert "model_list" in warning_mock.call_args.args[1]


@pytest.mark.parametrize("endpoint_override_key", ["api_base", "base_url"])
def test_load_prevents_serialized_endpoint_override_reaching_litellm(tmp_path, endpoint_override_key, make_run):
    file_path = tmp_path / "model.json"
    override_url = "http://override.local/v1"
    original_predict = Predict(pspec("q->a"))
    original_predict.lm = LM(model="openai/gpt-4o-mini")
    original_predict.save(file_path)
    with open(file_path, "rb") as f:
        saved_state = orjson.loads(f.read())
    saved_state["lm"][endpoint_override_key] = override_url
    with open(file_path, "wb") as f:
        f.write(orjson.dumps(saved_state))
    loaded_predict = Predict(pspec("q->a"))
    loaded_predict.load(file_path)

    class FakeResp(dict):
        usage: ClassVar[dict] = {}

        def __init__(self):
            super().__init__({"choices": []})

    with patch(
        "dspy.clients.lm.transport.alitellm_completion", new_callable=AsyncMock, return_value=FakeResp()
    ) as completion_mock:
        lm = loaded_predict.lm
        assert lm is not None
        run = make_run(lm=lm)
        asyncio.run(lm(LMRequest.from_call(model=lm.model, prompt="hello"), run=run))
    assert completion_mock.call_count == 1
    assert completion_mock.call_args.kwargs["request"].get(endpoint_override_key) != override_url


def test_load_blocks_serialized_model_list_unless_opted_in(tmp_path, make_run):
    file_path = tmp_path / "model.json"
    override_url = "http://override.local/v1"
    original_predict = Predict(pspec("q->a"))
    original_predict.lm = LM(model="openai/gpt-4o-mini")
    original_predict.save(file_path)
    with open(file_path, "rb") as f:
        saved_state = orjson.loads(f.read())
    saved_state["lm"]["model_list"] = [
        {
            "model_name": "openai/gpt-4o-mini",
            "litellm_params": {"model": "openai/gpt-4o-mini", "api_base": override_url},
        }
    ]
    with open(file_path, "wb") as f:
        f.write(orjson.dumps(saved_state))

    class FakeResp(dict):
        usage: ClassVar[dict] = {}

        def __init__(self):
            super().__init__({"choices": []})

    safe_loaded_predict = Predict(pspec("q->a"))
    safe_loaded_predict.load(file_path)
    with (
        patch("litellm.batch_completion_models", return_value=FakeResp()) as batch_completion_mock,
        patch(
            "dspy.clients.lm.transport.alitellm_completion", new_callable=AsyncMock, return_value=FakeResp()
        ) as completion_mock,
    ):
        lm = safe_loaded_predict.lm
        assert lm is not None
        run = make_run(lm=lm)
        asyncio.run(lm(LMRequest.from_call(model=lm.model, prompt="hello"), run=run))
    assert completion_mock.called
    assert not batch_completion_mock.called
    opt_in_loaded_predict = Predict(pspec("q->a"))
    opt_in_loaded_predict.load(file_path, allow_unsafe_lm_state=True)
    with patch(
        "litellm.batch_completion_models", new_callable=AsyncMock, return_value=FakeResp()
    ) as batch_completion_mock:
        lm = opt_in_loaded_predict.lm
        assert lm is not None
        run = make_run(lm=lm)
        asyncio.run(lm(LMRequest.from_call(model=lm.model, prompt="hello"), run=run))
    opt_in_deployments = batch_completion_mock.call_args.kwargs["deployments"]
    assert opt_in_deployments[0]["api_base"] == override_url


def test_load_uses_env_api_key_without_honoring_serialized_endpoint_override(tmp_path, monkeypatch, make_run):
    file_path = tmp_path / "model.json"
    override_url = "http://override.local/v1"
    env_api_key = "sk-live-test-secret"
    original_predict = Predict(pspec("q->a"))
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
        usage: ClassVar[dict] = {}

        def __init__(self):
            super().__init__({"choices": []})

    opt_in_loaded_predict = Predict(pspec("q->a"))
    opt_in_loaded_predict.load(file_path, allow_unsafe_lm_state=True)
    with patch("litellm.atext_completion", new_callable=AsyncMock, return_value=FakeResp()) as text_completion_mock:
        lm = opt_in_loaded_predict.lm
        assert lm is not None
        run = make_run(lm=lm)
        asyncio.run(lm(LMRequest.from_call(model=lm.model, prompt="hello"), run=run))
    assert text_completion_mock.call_args.kwargs["api_base"] == override_url
    assert text_completion_mock.call_args.kwargs["api_key"] == env_api_key
    safe_loaded_predict = Predict(pspec("q->a"))
    safe_loaded_predict.load(file_path)
    with patch("litellm.atext_completion", new_callable=AsyncMock, return_value=FakeResp()) as text_completion_mock:
        lm = safe_loaded_predict.lm
        assert lm is not None
        run = make_run(lm=lm)
        asyncio.run(lm(LMRequest.from_call(model=lm.model, prompt="hello"), run=run))
    assert text_completion_mock.call_args.kwargs["api_key"] == env_api_key
    assert text_completion_mock.call_args.kwargs["api_base"] != override_url


def test_forward_method(make_run):
    program = Predict(pspec("question -> answer"))
    run = make_run(lm=DummyLM([{"answer": "No more responses"}]))
    result = asyncio.run(program(question="What is 1+1?", run=run)).answer
    assert result == "No more responses"


def test_forward_method2(make_run):
    program = Predict(pspec("question -> answer1, answer2"))
    run = make_run(lm=DummyLM([{"answer1": "my first answer", "answer2": "my second answer"}]))
    result = asyncio.run(program(question="What is 1+1?", run=run))
    assert result.answer1 == "my first answer"
    assert result.answer2 == "my second answer"


def test_config_management(make_run):
    predict_instance = Predict(pspec("input -> output"))
    predict_instance.update_config(LMConfig(extensions={"new_key": "value"}))
    config = predict_instance.get_config()
    assert "new_key" in config.extensions
    assert config.extensions["new_key"] == "value"


def test_multi_output(make_run):
    program = Predict(pspec("question -> answer"), config=LMConfig(n=2))
    run = make_run(lm=DummyLM([{"answer": "my first answer"}, {"answer": "my second answer"}]))
    results = asyncio.run(program(question="What is 1+1?", run=run))
    assert results.completions.answer[0] == "my first answer"
    assert results.completions.answer[1] == "my second answer"


def test_multi_output2(make_run):
    program = Predict(pspec("question -> answer1, answer2"), config=LMConfig(n=2))
    run = make_run(
        lm=DummyLM(
            [{"answer1": "my 0 answer", "answer2": "my 2 answer"}, {"answer1": "my 1 answer", "answer2": "my 3 answer"}]
        )
    )
    results = asyncio.run(program(question="What is 1+1?", run=run))
    assert results.completions.answer1[0] == "my 0 answer"
    assert results.completions.answer1[1] == "my 1 answer"
    assert results.completions.answer2[0] == "my 2 answer"
    assert results.completions.answer2[1] == "my 3 answer"


def test_datetime_inputs_and_outputs(make_run):

    class TimedEvent(pydantic.BaseModel):
        event_name: str
        event_time: datetime

    TimedSignature = make_task_spec(
        {
            "events": input_field("events", type_=list[TimedEvent], desc="The events."),
            "summary": output_field("summary", desc="The summary."),
            "next_event_time": output_field("next_event_time", type_=datetime, desc="The next event time."),
        },
        instructions="Process timed events.",
        name="TimedSignature",
    )
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
    run = make_run(lm=lm)
    output = asyncio.run(
        program(
            events=[
                TimedEvent(event_name="Event 1", event_time=datetime(2024, 11, 25, 10, 0, 0)),
                TimedEvent(event_name="Event 2", event_time=datetime(2024, 11, 25, 15, 30, 0)),
            ],
            run=run,
        )
    )
    assert output.summary == "All events are processed"
    assert output.next_event_time == datetime(2024, 11, 27, 14, 0, 0)


def test_explicitly_valued_enum_inputs_and_outputs(make_run):

    class Status(enum.Enum):
        PENDING = "pending"
        IN_PROGRESS = "in_progress"
        COMPLETED = "completed"

    StatusSignature = make_task_spec(
        {
            "current_status": input_field("current_status", type_=Status, desc="The current status."),
            "next_status": output_field("next_status", type_=Status, desc="The next status."),
        },
        instructions="Advance status.",
        name="StatusSignature",
    )
    program = Predict(StatusSignature)
    lm = DummyLM(
        [{"reasoning": "The current status is 'PENDING', advancing to 'IN_PROGRESS'.", "next_status": "in_progress"}]
    )
    run = make_run(lm=lm)
    output = asyncio.run(program(current_status=Status.PENDING, run=run))
    assert output.next_status == Status.IN_PROGRESS


def test_enum_inputs_and_outputs_with_shared_names_and_values(make_run):

    class TicketStatus(enum.Enum):
        OPEN = "CLOSED"
        CLOSED = "RESOLVED"
        RESOLVED = "OPEN"

    TicketStatusSignature = make_task_spec(
        {
            "current_status": input_field("current_status", type_=TicketStatus, desc="The current status."),
            "next_status": output_field("next_status", type_=TicketStatus, desc="The next status."),
        },
        instructions="Advance ticket status.",
        name="TicketStatusSignature",
    )
    program = Predict(TicketStatusSignature)
    lm = DummyLM(
        [{"reasoning": "The ticket is currently 'OPEN', transitioning to 'CLOSED'.", "next_status": "RESOLVED"}]
    )
    run = make_run(lm=lm)
    output = asyncio.run(program(current_status=TicketStatus.OPEN, run=run))
    assert output.next_status == TicketStatus.CLOSED


def test_auto_valued_enum_inputs_and_outputs(make_run):
    Status = enum.Enum("Status", ["PENDING", "IN_PROGRESS", "COMPLETED"])
    StatusSignature = make_task_spec(
        {
            "current_status": input_field("current_status", type_=Status, desc="The current status."),
            "next_status": output_field("next_status", type_=Status, desc="The next status."),
        },
        instructions="Advance auto-valued status.",
        name="StatusSignature",
    )
    program = Predict(StatusSignature)
    lm = DummyLM(
        [{"reasoning": "The current status is 'PENDING', advancing to 'IN_PROGRESS'.", "next_status": "IN_PROGRESS"}]
    )
    run = make_run(lm=lm)
    output = asyncio.run(program(current_status=Status.PENDING, run=run))
    assert output.next_status == Status.IN_PROGRESS


def test_named_predictors(make_run):

    class MyModule(Module):
        def __init__(self):
            super().__init__()
            self.inner = Predict(pspec("question -> answer"))

    program = MyModule()
    assert program.named_predictors() == [("self.inner", program.inner)]
    program2 = copy.deepcopy(program)
    assert program2.named_predictors() == [("self.inner", program2.inner)]


def test_output_only(make_run):
    OutputOnlySignature = make_task_spec(
        {"output": output_field("output", desc="The output.")},
        instructions="Produce output.",
        name="OutputOnlySignature",
    )
    predictor = Predict(OutputOnlySignature)
    lm = DummyLM([{"output": "short answer"}])
    run = make_run(lm=lm)
    assert asyncio.run(predictor(run=run)).output == "short answer"


def test_load_state_chaining(make_run):
    original = Predict(pspec("question -> answer"))
    original.demos = [Example.from_record({"question": "test", "answer": "response"}, input_keys=("question",))]
    state = original.dump_state()
    new_instance = Predict(pspec("question -> answer")).load_state(state)
    assert new_instance is not None
    assert len(new_instance.demos) == len(original.demos)
    assert new_instance.demos[0]["question"] == original.demos[0]["question"]
    assert new_instance.demos[0]["answer"] == original.demos[0]["answer"]


@pytest.mark.parametrize("adapter_type", ["chat", "json"])
def test_call_predict_with_chat_history(adapter_type, make_run):
    MySignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "turn_log": input_field("turn_log", type_=TurnLog, desc="The history."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Answer with chat history.",
        name="MySignature",
    )
    program = Predict(MySignature)
    if adapter_type == "chat":
        lm = SpyLM("dummy_model")
        run = make_run(lm=lm, adapter=ChatAdapter())
    else:
        lm = SpyLM("dummy_model", return_json=True)
        run = make_run(lm=lm, adapter=JSONAdapter())
    asyncio.run(
        program(
            question="are you sure that's correct?",
            turn_log=TurnLog.model_validate(
                {"turns": [{"question": "what's the capital of france?", "answer": "paris"}]}
            ),
            run=run,
        )
    )
    assert len(lm.calls) == 1
    messages = lm.calls[0]["messages"]
    assert len(messages) == 4
    assert "what's the capital of france?" in messages[1]["content"]
    assert "paris" in messages[2]["content"]
    assert "are you sure that's correct" in messages[3]["content"]


def test_lm_usage(make_run):
    program = Predict(pspec("question -> answer"))
    run = make_run(
        lm=LM("openai/gpt-4o-mini"),
        adapter=ChatAdapter(),
        telemetry=TelemetryConfig(track_usage=True),
    )
    with patch(
        "dspy.clients.lm.transport.alitellm_completion",
        return_value=ModelResponse(
            choices=[{"message": {"content": "[[ ## answer ## ]]\nParis"}}], usage={"total_tokens": 10}
        ),
    ):
        result = asyncio.run(program(question="What is the capital of France?", run=run))
        assert result.answer == "Paris"
        assert result.get_lm_usage()["openai/gpt-4o-mini"]["total_tokens"] == 10


def test_lm_usage_with_parallel(make_run):
    program = Predict(pspec("question -> answer"))
    run = make_run(
        lm=LM("openai/gpt-4o-mini"),
        adapter=ChatAdapter(),
        telemetry=TelemetryConfig(track_usage=True),
    )
    with patch(
        "dspy.clients.lm.transport.alitellm_completion",
        return_value=ModelResponse(
            choices=[{"message": {"content": "[[ ## answer ## ]]\nParis"}}], usage={"total_tokens": 10}
        ),
    ):
        parallelizer = Parallel(run=run)
        input_pairs = [
            (program, {"question": "What is the capital of France?"}),
            (program, {"question": "What is the capital of France?"}),
        ]
        results = asyncio.run(parallelizer(input_pairs, run=run)).results
        assert results[0].answer == "Paris"
        assert results[1].answer == "Paris"
        assert results[0].get_lm_usage()["openai/gpt-4o-mini"]["total_tokens"] == 10
        assert results[1].get_lm_usage()["openai/gpt-4o-mini"]["total_tokens"] == 10


@pytest.mark.asyncio
async def test_lm_usage_with_async(make_run):
    from dspy.runtime.usage_tracker import UsageTracker

    program = Predict(pspec("question -> answer"))
    original_aforward_impl = program._aforward_impl

    async def patched_aforward_impl(self, *, run, options=None, **inputs):
        await asyncio.sleep(1)
        return await original_aforward_impl(run=run, options=options, **inputs)

    cast("Any", program)._aforward_impl = types.MethodType(patched_aforward_impl, program)
    run = make_run(
        lm=LM("openai/gpt-4o-mini"),
        adapter=ChatAdapter(),
        telemetry=TelemetryConfig(track_usage=True),
    )
    with patch(
        "litellm.acompletion",
        return_value=ModelResponse(
            choices=[{"message": {"content": "[[ ## answer ## ]]\nParis"}}], usage={"total_tokens": 10}
        ),
    ):
        coroutines = [
            program(
                question="What is the capital of France?",
                run=run.fork(usage_tracker=UsageTracker()),
            )
            for _ in range(4)
        ]
        results = await asyncio.gather(*coroutines)
        assert results[0].answer == "Paris"
        assert results[1].answer == "Paris"
        assert results[0].get_lm_usage()["openai/gpt-4o-mini"]["total_tokens"] == 10
        assert results[1].get_lm_usage()["openai/gpt-4o-mini"]["total_tokens"] == 10
        assert results[2].get_lm_usage()["openai/gpt-4o-mini"]["total_tokens"] == 10
        assert results[3].get_lm_usage()["openai/gpt-4o-mini"]["total_tokens"] == 10


def test_positional_arguments(make_run):
    program = Predict(pspec("question -> answer"))
    run = make_run(lm=DummyLM([{"answer": "Paris"}]))
    with pytest.raises(TypeError):
        asyncio.run(program("What is the capital of France?", run=run))


def test_error_message_on_invalid_lm_setup(make_run):
    with pytest.raises(TypeError, match="run"):
        asyncio.run(Predict(pspec("question -> answer"))(question="Why did a chicken cross the kitchen?"))

    predictor = Predict(pspec("question -> answer"))
    run = make_run(lm=DummyLM([{"answer": "test"}]))
    with pytest.raises(ValueError, match="options=PredictOptions"):
        asyncio.run(
            predictor(
                question="Why did a chicken cross the kitchen?",
                prediction="flat kwarg",
                run=run,
            )
        )

    predictor = Predict(pspec("question -> answer"))
    cast("Any", predictor).lm = "openai/gpt-4o-mini"
    run = make_run(lm=LM("openai/gpt-4o-mini"))
    with pytest.raises(
        ValueError, match=r"LM must be an instance of `dspy\.clients\.base_lm\.BaseLM`, not a string\."
    ) as e:
        asyncio.run(predictor(question="Why did a chicken cross the kitchen?", run=run))
    assert "LM must be an instance of `dspy.clients.base_lm.BaseLM`, not a string." in str(e.value)

    predictor = Predict(pspec("question -> answer"))

    def dummy_lm():
        pass

    cast("Any", predictor).lm = dummy_lm
    run = make_run(lm=LM("openai/gpt-4o-mini"))
    with pytest.raises(
        ValueError, match=r"LM must be an instance of `dspy\.clients\.base_lm\.BaseLM`, not <class 'function'>\."
    ) as e:
        asyncio.run(predictor(question="Why did a chicken cross the kitchen?", run=run))
    assert "LM must be an instance of `dspy.clients.base_lm.BaseLM`, not <class 'function'>." in str(e.value)


@pytest.mark.parametrize("adapter_type", ["chat", "json"])
def test_field_constraints(adapter_type, make_run):
    ConstrainedSignature = make_task_spec(
        {
            "text": input_field("text", desc="Input text", constraints="minimum length: 5, maximum length: 100"),
            "number": input_field(
                "number", type_=int, desc="A number between 0 and 10", constraints="greater than: 0, less than: 10"
            ),
            "score": output_field(
                "score",
                type_=float,
                desc="Score between 0 and 1",
                constraints="greater than or equal to: 0.0, less than or equal to: 1.0",
            ),
            "count": output_field(
                "count", type_=int, desc="Even number count", constraints="a multiple of the given number: 2"
            ),
        },
        instructions="Test signature with constrained fields.",
        name="ConstrainedSignature",
    )
    program = Predict(ConstrainedSignature)
    if adapter_type == "chat":
        lm = SpyLM("dummy_model", response_text="[[ ## score ## ]]\n0.5\n[[ ## count ## ]]\n2")
        run = make_run(lm=lm, adapter=ChatAdapter())
    else:
        lm = SpyLM("dummy_model", return_json=True, response_text="{'score':'0.5', 'count':'2'}")
        run = make_run(lm=lm, adapter=JSONAdapter())
    asyncio.run(program(text="hello world", number=5, run=run))
    system_message = lm.calls[0]["messages"][0]["content"]
    assert "minimum length: 5" in system_message
    assert "maximum length: 100" in system_message
    assert "greater than: 0" in system_message
    assert "less than: 10" in system_message
    assert "greater than or equal to: 0.0" in system_message
    assert "less than or equal to: 1.0" in system_message
    assert "a multiple of the given number: 2" in system_message


@pytest.mark.asyncio
async def test_async_predict(make_run):
    program = Predict(pspec("question -> answer"))
    run = make_run(lm=DummyLM([{"answer": "Paris"}]))
    result = await program(question="What is the capital of France?", run=run)
    assert result.answer == "Paris"


def test_predicted_outputs_piped_from_predict_to_lm_call(make_run):
    program = Predict(pspec("question -> answer"))
    run = make_run(lm=LM("openai/gpt-4o-mini"))
    mock_response = ModelResponse(choices=[{"message": {"content": "[[ ## answer ## ]]\nParis"}}])
    with patch("litellm.acompletion", return_value=mock_response) as mock_completion:
        asyncio.run(
            program(
                question="Why did a chicken cross the kitchen?",
                options=PredictOptions(prediction={"type": "content", "content": "A chicken crossing the kitchen"}),
                run=run,
            )
        )
        assert mock_completion.call_args[1]["prediction"] == {
            "type": "content",
            "content": "A chicken crossing the kitchen",
        }
    program = Predict(pspec("question, candidate -> judgement"))
    judgement_response = ModelResponse(choices=[{"message": {"content": "[[ ## judgement ## ]]\nFair"}}])
    with patch("litellm.acompletion", return_value=judgement_response) as mock_completion:
        asyncio.run(
            program(
                question="Why did a chicken cross the kitchen?",
                candidate="To get to the other side!",
                run=run,
            )
        )
    assert "prediction" not in mock_completion.call_args[1]


def test_dump_state_pydantic_non_primitive_types(make_run):

    class WebsiteInfo(BaseModel):
        name: str
        url: HttpUrl
        description: str | None = None
        created_at: datetime

    TestSignature = make_task_spec(
        {
            "website_info": input_field("website_info", type_=WebsiteInfo, desc="The website info."),
            "summary": output_field("summary", desc="The summary."),
        },
        instructions="Summarize website info.",
        name="TestSignature",
    )
    website_info = WebsiteInfo(
        name="Example",
        url=cast("HttpUrl", "https://www.example.com"),
        description="Test website",
        created_at=datetime(2021, 1, 1, 12, 0, 0),
    )
    serialized = to_jsonable(website_info)
    assert serialized["url"] == "https://www.example.com/"
    assert serialized["created_at"] == "2021-01-01T12:00:00"
    json_str = orjson.dumps(serialized).decode()
    reloaded = orjson.loads(json_str)
    assert reloaded == serialized
    predictor = Predict(TestSignature)
    demo = {"website_info": website_info, "summary": "This is a test website."}
    predictor.demos = [Example.from_record(demo, input_keys=("website_info",))]
    state = predictor.dump_state()
    json_str = orjson.dumps(state).decode()
    reloaded_state = orjson.loads(json_str)
    demo_data = reloaded_state["demos"][0]
    assert demo_data["website_info"]["url"] == "https://www.example.com/"
    assert demo_data["website_info"]["created_at"] == "2021-01-01T12:00:00"


def test_trace_size_limit(make_run):
    program = Predict(pspec("question -> answer"))
    run = make_run(lm=DummyLM([{"answer": "Paris"}]), telemetry=TelemetryConfig(max_optimization_trace_entries=3))
    for _ in range(10):
        asyncio.run(program(question="What is the capital of France?", run=run))
    assert len(run.optimization_trace) == 3


def test_disable_trace(make_run):
    program = Predict(pspec("question -> answer"))
    run = make_run(lm=DummyLM([{"answer": "Paris"}]))
    for _ in range(10):
        asyncio.run(program(question="What is the capital of France?", run=run, options=PredictOptions(trace=False)))
    assert run.optimization_trace == []


def test_per_module_history_size_limit(make_run):
    program = Predict(pspec("question -> answer"))
    run = make_run(lm=DummyLM([{"answer": "Paris"}]), telemetry=TelemetryConfig(max_call_log_entries=5))
    for _ in range(10):
        asyncio.run(program(question="What is the capital of France?", run=run))
    assert len(program.call_log) == 5


def test_per_module_history_disabled(make_run):
    program = Predict(pspec("question -> answer"))
    run = make_run(lm=DummyLM([{"answer": "Paris"}]), telemetry=TelemetryConfig(call_log=CallLogMode.off))
    for _ in range(10):
        asyncio.run(program(question="What is the capital of France?", run=run))
    assert len(program.call_log) == 0


def test_input_field_default_value(make_run):
    SignatureWithDefault = make_task_spec(
        {
            "context": input_field("context", default="DEFAULT_CONTEXT", desc="The context."),
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Answer using context.",
        name="SignatureWithDefault",
    )
    lm = SpyLM("dummy_model", response_text="[[ ## answer ## ]]\ntest")
    run = make_run(lm=lm)
    predictor = Predict(SignatureWithDefault)
    asyncio.run(predictor(question="test", run=run))
    user_message = lm.calls[0]["messages"][-1]["content"]
    assert "DEFAULT_CONTEXT" in user_message


def log_test_helper():
    from dspy.adapters.chat_adapter import ChatAdapter
    from dspy.runtime import CallLogMode, RunContext, TelemetryConfig, TransparencyMode

    dspy_logger = logging.getLogger("dspy")
    dspy_logger.propagate = True
    return RunContext.create(
        lm=DummyLM([{"answer": "test output"}]),
        adapter=ChatAdapter(),
        telemetry=TelemetryConfig(transparency=TransparencyMode.off, call_log=CallLogMode.memory),
        init_run_log=False,
    )


def test_extra_fields_warning(make_run):
    run = log_test_helper()
    predict_instance = Predict(pspec("question -> answer"))
    with pytest.raises(ValueError, match="Unknown task input"):
        asyncio.run(predict_instance(question="test", extra_field="should warn", another="also warn", run=run))


def test_missing_optional_input_field_no_warning(caplog, make_run):
    run = log_test_helper()
    OptionalInputSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "context": input_field("context", type_=str | None, desc="The context."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Answer with optional context.",
        name="OptionalInputSignature",
    )
    predict_instance = Predict(OptionalInputSignature)
    with caplog.at_level(logging.WARNING, logger="dspy.predict.predict"):
        asyncio.run(predict_instance(question="test", run=run))
    assert "Not all input fields were provided" not in caplog.text


def test_missing_required_input_field_still_warns(make_run):
    run = log_test_helper()
    OptionalInputSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "context": input_field("context", type_=str | None, desc="The context."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Answer with optional context.",
        name="OptionalInputSignature",
    )
    predict_instance = Predict(OptionalInputSignature)
    with pytest.raises(ValueError, match="Missing required"):
        asyncio.run(predict_instance(run=run))


def test_warning_images(make_run):
    run = log_test_helper()
    predict_instance = Predict(pspec("question:Image -> answer"))
    asyncio.run(predict_instance(question=Image("https://example.com/image1.jpg"), run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(question="dog_image", run=run))


def test_type_mismatch_warning(make_run):
    TypedSignature = make_task_spec(
        {
            "count": input_field("count", type_=int, desc="The count."),
            "name": input_field("name", desc="The name."),
            "result": output_field("result", desc="The result."),
        },
        instructions="Typed inputs.",
        name="TypedSignature",
    )
    predict_instance = Predict(TypedSignature)
    lm = DummyLM([{"result": "test output"}])
    run = make_run(lm=lm)
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(count="not an int", name="test", run=run))


def test_correct_types_no_warning(make_run):
    TypedSignature = make_task_spec(
        {
            "count": input_field("count", type_=int, desc="The count."),
            "name": input_field("name", desc="The name."),
            "result": output_field("result", desc="The result."),
        },
        instructions="Typed inputs.",
        name="TypedSignature",
    )
    predict_instance = Predict(TypedSignature)
    lm = DummyLM([{"result": "test output"}])
    run = make_run(lm=lm)
    asyncio.run(predict_instance(count=42, name="test", run=run))


def test_list_type_validation(make_run):
    ComplexSignature = make_task_spec(
        {
            "items": input_field("items", type_=list[str], desc="The items."),
            "result": output_field("result", desc="The result."),
        },
        instructions="Process items.",
        name="ComplexSignature",
    )
    predict_instance = Predict(ComplexSignature)
    lm = DummyLM([{"result": "test output 1"}, {"result": "test output 2"}])
    run = make_run(lm=lm)
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(items="not a list", run=run))
    asyncio.run(predict_instance(items=["a", "b", "c"], run=run))


def test_literal_type_validation(make_run):
    from typing import Literal

    LiteralSignature = make_task_spec(
        {
            "status": input_field("status", type_=Literal["pending", "approved", "rejected"], desc="The status."),
            "priority": input_field("priority", type_=Literal[1, 2, 3], desc="The priority."),
            "result": output_field("result", desc="The result."),
        },
        instructions="Validate literals.",
        name="LiteralSignature",
    )
    predict_instance = Predict(LiteralSignature)
    lm = DummyLM([{"result": "test output"}])
    run = make_run(lm=lm)
    asyncio.run(predict_instance(status="approved", priority=2, run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(status="invalid", priority=2, run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(status="approved", priority=5, run=run))


def test_literal_union_type_validation(make_run):
    from typing import Literal

    UnionLiteralSignature = make_task_spec(
        {
            "mode": input_field("mode", type_=Literal["auto", "manual"] | None, desc="The mode."),
            "result": output_field("result", desc="The result."),
        },
        instructions="Validate union literals.",
        name="UnionLiteralSignature",
    )
    predict_instance = Predict(UnionLiteralSignature)
    lm = DummyLM([{"result": "test output"}, {"result": "test output"}])
    run = make_run(lm=lm)
    asyncio.run(predict_instance(mode="auto", run=run))
    asyncio.run(predict_instance(mode=None, run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(mode="invalid", run=run))


def test_list_string(make_run):
    TypedSignature = make_task_spec(
        {
            "nameList": input_field("nameList", type_=list[str], desc="The name list."),
            "result": output_field("result", desc="The result."),
        },
        instructions="Process name list.",
        name="TypedSignature",
    )
    predict_instance = Predict(TypedSignature)
    lm = DummyLM([{"result": "test output"}])
    run = make_run(lm=lm)
    asyncio.run(predict_instance(nameList=["Alice", "Bob", "Charlie"], run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(nameList=[1, 2, 3, None], run=run))


def test_nested_list_type_validation(make_run):
    NestedListSignature = make_task_spec(
        {
            "numbers": input_field("numbers", type_=list[int], desc="The numbers."),
            "names": input_field("names", type_=list[str], desc="The names."),
            "result": output_field("result", desc="The result."),
        },
        instructions="Validate nested lists.",
        name="NestedListSignature",
    )
    predict_instance = Predict(NestedListSignature)
    lm = DummyLM([{"result": "test output"}, {"result": "test output"}, {"result": "test output"}])
    run = make_run(lm=lm)
    asyncio.run(predict_instance(numbers=[1, 2, 3], names=["alice", "bob"], run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(numbers=["1", "2", "3"], names=["alice", "bob"], run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(numbers=[1, 2, 3], names=[1, 2, 3], run=run))
    asyncio.run(predict_instance(numbers=[], names=[], run=run))


def test_nested_dict_type_validation(make_run):
    DictSignature = make_task_spec(
        {
            "mapping": input_field("mapping", type_=dict[str, int], desc="The mapping."),
            "result": output_field("result", desc="The result."),
        },
        instructions="Validate dict input.",
        name="DictSignature",
    )
    predict_instance = Predict(DictSignature)
    lm = DummyLM([{"result": "test output"}])
    run = make_run(lm=lm)
    asyncio.run(predict_instance(mapping={"a": 1, "b": 2, "c": 3}, run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(mapping={"a": "1", "b": "2", "c": "3"}, run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(mapping={1: 1, 2: 2, 3: 3}, run=run))


def test_nested_tuple_type_validation(make_run):
    TupleSignature = make_task_spec(
        {
            "fixed_tuple": input_field("fixed_tuple", type_=tuple[str, int, bool], desc="The fixed tuple."),
            "var_tuple": input_field("var_tuple", type_=tuple[int, ...], desc="The var tuple."),
            "result": output_field("result", desc="The result."),
        },
        instructions="Validate tuple input.",
        name="TupleSignature",
    )
    predict_instance = Predict(TupleSignature)
    lm = DummyLM([{"result": "test output"}])
    run = make_run(lm=lm)
    asyncio.run(predict_instance(fixed_tuple=("hello", 42, True), var_tuple=(1, 2, 3, 4), run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(fixed_tuple=(123, 42, True), var_tuple=(1, 2, 3), run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(fixed_tuple=("hello", 42), var_tuple=(1, 2, 3), run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(fixed_tuple=("hello", 42, True), var_tuple=("a", "b", "c"), run=run))


def test_literal_type_validation_string_signature(make_run):
    predict_instance = Predict(
        pspec("status:Literal['pending','approved','rejected'], priority:Literal[1,2,3] -> result")
    )
    lm = DummyLM([{"result": "test output"}])
    run = make_run(lm=lm)
    asyncio.run(predict_instance(status="approved", priority=2, run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(status="invalid", priority=2, run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(status="approved", priority=5, run=run))


def test_list_type_validation_string_signature(make_run):
    predict_instance = Predict(pspec("numbers:list[int], names:list[str] -> result"))
    lm = DummyLM([{"result": "test output"}, {"result": "test output"}, {"result": "test output"}])
    run = make_run(lm=lm)
    asyncio.run(predict_instance(numbers=[1, 2, 3], names=["alice", "bob"], run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(numbers=["1", "2", "3"], names=["alice", "bob"], run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(numbers=[1, 2, 3], names=[1, 2, 3], run=run))
    asyncio.run(predict_instance(numbers=[], names=[], run=run))


def test_dict_type_validation_string_signature(make_run):
    predict_instance = Predict(pspec("mapping:dict[str,int] -> result"))
    lm = DummyLM([{"result": "test output"}])
    run = make_run(lm=lm)
    asyncio.run(predict_instance(mapping={"a": 1, "b": 2, "c": 3}, run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(mapping={"a": "1", "b": "2", "c": "3"}, run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(mapping={1: 1, 2: 2, 3: 3}, run=run))


def test_tuple_type_validation_string_signature(make_run):
    predict_instance = Predict(pspec("fixed_tuple:tuple[str,int,bool], var_tuple:tuple[int,...] -> result"))
    lm = DummyLM([{"result": "test output"}])
    run = make_run(lm=lm)
    asyncio.run(predict_instance(fixed_tuple=("hello", 42, True), var_tuple=(1, 2, 3, 4), run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(fixed_tuple=(123, 42, True), var_tuple=(1, 2, 3), run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(fixed_tuple=("hello", 42), var_tuple=(1, 2, 3), run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(fixed_tuple=("hello", 42, True), var_tuple=("a", "b", "c"), run=run))


def test_union_type_validation_string_signature(make_run):
    predict_instance = Predict(pspec("mode:Literal['auto','manual']|None -> result"))
    lm = DummyLM([{"result": "test output"}, {"result": "test output"}])
    run = make_run(lm=lm)
    asyncio.run(predict_instance(mode="auto", run=run))
    asyncio.run(predict_instance(mode=None, run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(mode="invalid", run=run))


@pytest.mark.parametrize("enable_type_warnings", [False, True])
def test_basic_types_string_signature(enable_type_warnings, make_run):
    predict_instance = Predict(pspec("count:int, name:str -> result"))
    lm = DummyLM([{"result": "test output"}])
    run = make_run(lm=lm, telemetry=TelemetryConfig(warn_on_type_mismatch=enable_type_warnings))
    asyncio.run(predict_instance(count=42, name="test", run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(count="not an int", name="test", run=run))


def test_untyped_string_signature(make_run):
    predict_instance = Predict(pspec("count, name -> result"))
    lm = DummyLM([{"result": "test output"}])
    run = make_run(lm=lm)
    asyncio.run(predict_instance(count="abc", name=123, run=run))


def test_untyped_class_signature(make_run):
    TestSignature = make_task_spec(
        {
            "count": input_field("count", is_type_undefined=True, desc="The count."),
            "name": input_field("name", is_type_undefined=True, desc="The name."),
            "result": output_field("result", desc="The result."),
        },
        instructions="Untyped class fields.",
        name="TestSignature",
    )
    predict_instance = Predict(TestSignature)
    lm = DummyLM([{"result": "test output"}])
    run = make_run(lm=lm)
    asyncio.run(predict_instance(count="abc", name=123, run=run))


def test_string_to_list_signature(make_run):
    TestSignature = make_task_spec(
        {
            "name": input_field("name", desc="The name."),
            "count": input_field("count", is_type_undefined=True, desc="The count."),
            "result": output_field("result", desc="The result."),
        },
        instructions="String to list validation.",
        name="TestSignature",
    )
    predict_instance = Predict(TestSignature)
    lm = DummyLM([{"result": "test output"}])
    run = make_run(lm=lm)
    asyncio.run(predict_instance(name=["abc", "def", "geh"], count=123, run=run))


@pytest.mark.parametrize("enable_type_warnings", [False, True])
def test_custom_signature_types(enable_type_warnings, make_run):
    class MyContainer:
        class Query(pydantic.BaseModel):
            text: str

    task_spec = make_task_spec(
        {
            "query": input_field("query", type_=MyContainer.Query, desc="The query."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Answer the query.",
    )
    predict_instance = Predict(task_spec)
    query_instance = MyContainer.Query(text="What is the capital of France?")
    lm = DummyLM([{"answer": "test output"}])
    run = make_run(lm=lm, telemetry=TelemetryConfig(warn_on_type_mismatch=enable_type_warnings))
    asyncio.run(predict_instance(query=query_instance, run=run))
    with pytest.raises(ValueError, match="Type mismatch"):
        asyncio.run(predict_instance(query="What is the capital of France?", run=run))
