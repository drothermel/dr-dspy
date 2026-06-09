import asyncio
import contextlib

import pytest
from typing_extensions import override

from dspy.clients.base_lm import BaseLM
from dspy.clients.openai_format import message_to_openai_chat, to_openai_chat_request
from dspy.core.types import LMRequest
from dspy.core.types.config import coerce_lm_config
from dspy.utils.dummies import DummyLM

try:
    from litellm.utils import Choices, Message, ModelResponse
except ImportError:
    Choices = Message = ModelResponse = None  # ty: ignore[invalid-assignment]


def default_model_response(content: str = "", *, model: str = "openai/gpt-4o-mini"):
    if ModelResponse is None:
        raise RuntimeError("litellm is required for adapter LM mock helpers")
    return ModelResponse(choices=[Choices(message=Message(content=content))], model=model)


def litellm_request_messages(call_args) -> list[dict]:
    return call_args.kwargs["request"]["messages"]


class StopAdapterCallCapture(BaseException):
    pass


def captured_lm_kwargs(request: LMRequest) -> dict:
    data = to_openai_chat_request(request)
    data.pop("model", None)
    data.pop("messages", None)
    return data


class CapturingLM(BaseLM):
    def __init__(self, source_lm=None):
        source_lm = source_lm or DummyLM([{}])
        super().__init__(model=source_lm.model, model_type=source_lm.model_type)
        self.source_lm = source_lm
        self.calls = []
        for key in ("reasoning", "reasoning_effort"):
            if key in source_lm.kwargs:
                self.kwargs[key] = source_lm.kwargs[key]

    @property
    @override
    def supports_function_calling(self):
        return self.source_lm.supports_function_calling

    @property
    @override
    def supports_reasoning(self):
        return self.source_lm.supports_reasoning

    @property
    @override
    def supports_response_schema(self):
        return self.source_lm.supports_response_schema

    @property
    @override
    def supported_params(self):
        return self.source_lm.supported_params

    @override
    async def __call__(self, request: LMRequest, *, run):
        self.calls.append({"request": request})
        raise StopAdapterCallCapture


def make_adapter_run(*, lm, adapter):
    from dspy.runtime import CallLogMode, RunContext, TelemetryConfig

    return RunContext.create(
        lm=lm,
        adapter=adapter,
        telemetry=TelemetryConfig(transparency="off", call_log=CallLogMode.memory),
        init_run_log=False,
    )


@pytest.fixture
def adapter_run(make_run):
    def _adapter_run(lm=None, adapter=None):
        from dspy.adapters.chat_adapter import ChatAdapter

        return make_run(lm=lm or DummyLM([{}]), adapter=adapter or ChatAdapter())

    return _adapter_run


async def _format_messages_and_lm_kwargs(
    *, adapter, task_spec, demos, inputs, config=None, lm=None, lm_kwargs=None, run=None
):
    if lm_kwargs is not None:
        if config is not None:
            raise TypeError("Pass either `config` or `lm_kwargs`, not both.")
        config = lm_kwargs
    capturing_lm = CapturingLM(lm)
    if run is None:
        run = make_adapter_run(lm=capturing_lm, adapter=adapter)
    with contextlib.suppress(StopAdapterCallCapture):
        await adapter.acall(
            lm=capturing_lm,
            config=coerce_lm_config(config),
            task_spec=task_spec,
            demos=demos,
            inputs=inputs,
            run=run,
        )
    assert len(capturing_lm.calls) == 1
    call = capturing_lm.calls[0]
    request = call["request"]
    return ([message_to_openai_chat(message) for message in request.messages], captured_lm_kwargs(request))


def format_messages_and_lm_kwargs(*, adapter, task_spec, demos, inputs, config=None, lm=None, lm_kwargs=None):
    return asyncio.run(
        _format_messages_and_lm_kwargs(
            adapter=adapter, task_spec=task_spec, demos=demos, inputs=inputs, config=config, lm=lm, lm_kwargs=lm_kwargs
        )
    )


def adapter_format_as_openai(*, adapter, task_spec, demos, inputs):
    return [
        message_to_openai_chat(message) for message in adapter.format(task_spec=task_spec, demos=demos, inputs=inputs)
    ]
