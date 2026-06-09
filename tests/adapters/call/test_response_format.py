import pytest
from pydantic import ValidationError

from dspy.adapters.call.policies.response_format import StructuredOutputPolicy
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types.tool import ToolCalls
from dspy.clients.lm import LM
from dspy.core.types import LMConfig
from dspy.task_spec import input_field, make_task_spec, output_field
from tests.adapters.conftest import CapturingLM, StopAdapterCallCapture, make_adapter_run
from tests.task_spec.helpers import ts


@pytest.mark.asyncio
async def test_json_adapter_passes_structured_output_when_supported():
    signature = ts("question -> answer", instructions="Answer.")
    adapter = JSONAdapter()
    lm = CapturingLM(LM("openai/gpt-4o-mini"))
    with pytest.raises(StopAdapterCallCapture):
        await adapter(
            lm=lm,
            config={},
            task_spec=signature,
            demos=[],
            inputs={"question": "hi"},
            run=make_adapter_run(lm=lm, adapter=adapter),
        )
    request = lm.calls[0]["request"]
    response_format = request.config.response_format
    assert response_format is not None
    assert response_format["type"] == "json_schema"
    assert "schema" in response_format["json_schema"]


@pytest.mark.asyncio
async def test_json_adapter_uses_json_object_mode_without_response_schema_support():
    signature = ts("question -> answer", instructions="Answer.")
    adapter = JSONAdapter()
    source_lm = LM("openai/gpt-4o-mini")

    class NoSchemaLM(CapturingLM):
        @property
        def supports_response_schema(self):
            return False

    lm = NoSchemaLM(source_lm)
    with pytest.raises(StopAdapterCallCapture):
        await adapter(
            lm=lm,
            config={},
            task_spec=signature,
            demos=[],
            inputs={"question": "hi"},
            run=make_adapter_run(lm=lm, adapter=adapter),
        )
    request = lm.calls[0]["request"]
    assert request.config.response_format == {"type": "json_object"}


@pytest.mark.asyncio
async def test_structured_output_policy_uses_json_object_for_open_ended_mapping():
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "metadata": output_field("metadata", type_=dict[str, str], desc="Open metadata."),
        },
        instructions="Answer.",
    )
    adapter = JSONAdapter()
    lm = CapturingLM(LM("openai/gpt-4o-mini"))
    with pytest.raises(StopAdapterCallCapture):
        await adapter(
            lm=lm,
            config={},
            task_spec=task_spec,
            demos=[],
            inputs={"question": "hi"},
            run=make_adapter_run(lm=lm, adapter=adapter),
        )
    assert lm.calls[0]["request"].config.response_format == {"type": "json_object"}


@pytest.mark.asyncio
async def test_structured_output_policy_passthrough_when_response_format_unsupported():
    signature = ts("question -> answer", instructions="Answer.")
    adapter = JSONAdapter()
    source_lm = LM("openai/gpt-4o-mini")

    class NoResponseFormatLM(CapturingLM):
        @property
        def supported_params(self):
            return [param for param in source_lm.supported_params if param != "response_format"]

    lm = NoResponseFormatLM(source_lm)
    with pytest.raises(StopAdapterCallCapture):
        await adapter(
            lm=lm,
            config={"temperature": 0.0},
            task_spec=signature,
            demos=[],
            inputs={"question": "hi"},
            run=make_adapter_run(lm=lm, adapter=adapter),
        )
    assert lm.calls[0]["request"].config.response_format is None


@pytest.mark.asyncio
async def test_structured_output_policy_falls_back_to_json_object_on_schema_error(monkeypatch):
    signature = ts("question -> answer", instructions="Answer.")
    adapter = JSONAdapter()
    lm = CapturingLM(LM("openai/gpt-4o-mini"))

    def raise_validation_error(**kwargs):
        raise ValidationError.from_exception_data("test", [])

    monkeypatch.setattr(
        "dspy.adapters.call.policies.response_format.get_structured_outputs_response_format",
        raise_validation_error,
    )
    with pytest.raises(StopAdapterCallCapture):
        await adapter(
            lm=lm,
            config={},
            task_spec=signature,
            demos=[],
            inputs={"question": "hi"},
            run=make_adapter_run(lm=lm, adapter=adapter),
        )
    assert lm.calls[0]["request"].config.response_format == {"type": "json_object"}


@pytest.mark.asyncio
async def test_structured_output_policy_uses_json_object_when_tool_calls_without_native_fc():
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Answer.",
    )
    adapter = JSONAdapter(use_native_function_calling=False)
    lm = CapturingLM(LM("openai/gpt-4o-mini"))
    policy = StructuredOutputPolicy()
    captured_configs: list[object] = []

    async def run_once(config):
        captured_configs.append(config.response_format if config else None)
        return [{"tool_calls": ToolCalls(tool_calls=[])}]

    await policy.execute(
        adapter=adapter,
        lm=lm,
        config=LMConfig(),
        task_spec=task_spec,
        demos=[],
        inputs={"question": "hi"},
        run_once=run_once,
    )
    assert captured_configs == [{"type": "json_object"}]
