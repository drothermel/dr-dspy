import contextlib

from dspy.clients.base_lm import BaseLM
from dspy.clients.openai_format import message_to_openai_chat, to_openai_chat_request
from dspy.core.types import LMRequest, LMResponse
from dspy.utils.dummies import DummyLM

try:
    from litellm.utils import Choices, Message, ModelResponse
except ImportError:
    Choices = Message = ModelResponse = None


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


def legacy_outputs_to_lm_response(outputs: list[dict]) -> LMResponse:
    from dspy.clients.openai_format import provider_tool_call_to_part
    from dspy.core.types import LMOutput, LMTextPart, LMThinkingPart

    lm_outputs = []
    for output in outputs:
        parts = []
        text = output.get("text")
        if isinstance(text, str):
            parts.append(LMTextPart(text=text))
        reasoning = output.get("reasoning_content")
        if isinstance(reasoning, str):
            parts.append(LMThinkingPart(text=reasoning))
        for tool_call in output.get("tool_calls") or []:
            parts.append(provider_tool_call_to_part(tool_call))
        lm_outputs.append(LMOutput(parts=parts, provider_output=output))
    return LMResponse(model="test", outputs=lm_outputs)


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
    def supports_function_calling(self):
        return self.source_lm.supports_function_calling

    @property
    def supports_reasoning(self):
        return self.source_lm.supports_reasoning

    @property
    def supports_response_schema(self):
        return self.source_lm.supports_response_schema

    @property
    def supported_params(self):
        return self.source_lm.supported_params

    def __call__(self, request: LMRequest):
        self.calls.append({"request": request})
        raise StopAdapterCallCapture


def format_messages_and_lm_kwargs(adapter, signature, demos, inputs, lm_kwargs=None, lm=None):
    capturing_lm = CapturingLM(lm)
    with contextlib.suppress(StopAdapterCallCapture):
        adapter(capturing_lm, dict(lm_kwargs or {}), signature, demos, inputs)

    assert len(capturing_lm.calls) == 1
    call = capturing_lm.calls[0]
    request = call["request"]
    return (
        [message_to_openai_chat(message) for message in request.messages],
        captured_lm_kwargs(request),
    )
