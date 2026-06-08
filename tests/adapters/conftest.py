import asyncio
import contextlib

from typing_extensions import override

from dspy.clients.base_lm import BaseLM
from dspy.clients.openai_format import message_to_openai_chat, to_openai_chat_request
from dspy.core.types import LMRequest, coerce_lm_config
from dspy.utils.dummies import DummyLM

try:
    from litellm.utils import Choices, Message, ModelResponse
except ImportError:
    Choices = Message = ModelResponse = None  # ty:ignore[invalid-assignment]


def default_model_response(content: str = "", *, model: str = "openai/gpt-4o-mini"):
    if ModelResponse is None:
        raise RuntimeError("litellm is required for adapter LM mock helpers")
    return ModelResponse(
        choices=[Choices(message=Message(content=content))],
        model=model,
    )


def litellm_request_messages(call_args) -> list[dict]:
    return call_args.kwargs["request"]["messages"]


class StopAdapterCallCapture(BaseException):
    """Stop adapter execution after capturing the LM call.

    The exact-format tests assert the adapter-to-LM boundary: the messages and
    keyword arguments passed to the LM. Raising here avoids needing to craft a
    parseable LM response for every signature under test.
    """


def captured_lm_kwargs(request: LMRequest) -> dict:
    """Return the OpenAI-shaped kwargs the typed LM boundary would send."""
    data = to_openai_chat_request(request)
    data.pop("model", None)
    data.pop("messages", None)
    if request.config.cache is not None:
        if request.config.cache.enabled is not None:
            data["cache"] = request.config.cache.enabled
        if request.config.cache.rollout_id is not None:
            data["rollout_id"] = request.config.cache.rollout_id
    return data


class CapturingLM(BaseLM):
    def __init__(self, source_lm=None):
        source_lm = source_lm or DummyLM([{}])
        super().__init__(
            model=source_lm.model,
            model_type=source_lm.model_type,
            cache=source_lm.cache,
        )
        self.source_lm = source_lm
        self.calls = []

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
    async def __call__(self, request: LMRequest):
        self.calls.append({"request": request})
        raise StopAdapterCallCapture


async def _format_messages_and_lm_kwargs(*, adapter, signature, demos, inputs, config=None, lm=None, lm_kwargs=None):
    if lm_kwargs is not None:
        if config is not None:
            raise TypeError("Pass either `config` or `lm_kwargs`, not both.")
        config = lm_kwargs
    capturing_lm = CapturingLM(lm)
    with contextlib.suppress(StopAdapterCallCapture):
        await adapter.acall(
            lm=capturing_lm,
            config=coerce_lm_config(config),
            signature=signature,
            demos=demos,
            inputs=inputs,
        )

    assert len(capturing_lm.calls) == 1
    call = capturing_lm.calls[0]
    request = call["request"]
    return (
        [message_to_openai_chat(message) for message in request.messages],
        captured_lm_kwargs(request),
    )


def format_messages_and_lm_kwargs(*, adapter, signature, demos, inputs, config=None, lm=None, lm_kwargs=None):
    return asyncio.run(
        _format_messages_and_lm_kwargs(
            adapter=adapter,
            signature=signature,
            demos=demos,
            inputs=inputs,
            config=config,
            lm=lm,
            lm_kwargs=lm_kwargs,
        )
    )


def adapter_format_as_openai(*, adapter, signature, demos, inputs):
    """Return OpenAI-chat-shaped messages from adapter.format()."""
    return [
        message_to_openai_chat(message) for message in adapter.format(signature=signature, demos=demos, inputs=inputs)
    ]
