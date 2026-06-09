from dspy.clients.openai_format.chat_request import request_messages_as_openai
from dspy.core.types import CallRecord, LMConfig, LMRequest, LMResponse, User


def test_config_extensions_surface_in_history_kwargs():
    config = LMConfig(temperature=0.2, extensions={"provider_flag": True})
    request = LMRequest(model="model", messages=[], config=config)
    entry = CallRecord(request=request, response=LMResponse.from_text("ok"), timestamp="timestamp", uuid="uuid")
    assert entry.kwargs == {"provider_flag": True, "temperature": 0.2}


def test_default_config_does_not_serialize_empty_stop_sequences():
    request = LMRequest.from_call(model="model", prompt="hi")
    entry = CallRecord(request=request, response=LMResponse.from_text("ok"), timestamp="timestamp", uuid="uuid")
    assert request.config.stop is None
    assert entry.kwargs == {}


def test_history_entry_exposes_typed_derived_properties():
    message = User("hi")
    request = LMRequest.from_call(model="model", messages=[message], config=LMConfig(temperature=0.2))
    response = LMResponse.from_text("ok", model="response-model", usage={"input_tokens": 1}, cost=0.5)
    entry = CallRecord(request=request, response=response, timestamp="timestamp", uuid="uuid")
    assert entry.model == "model"
    assert entry.prompt == "hi"
    assert entry.messages == [message]
    assert request_messages_as_openai(entry.request) == [{"role": "user", "content": "hi"}]
    assert entry.outputs == ["ok"]
    assert entry.usage["input_tokens"] == 1
    assert entry.cost == 0.5
    assert entry.kwargs == {"temperature": 0.2}
    assert entry.response_model == "response-model"
