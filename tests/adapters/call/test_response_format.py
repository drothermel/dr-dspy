import pytest

from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.lm import LM
from tests.adapters.conftest import CapturingLM, StopAdapterCallCapture, make_adapter_run
from tests.task_spec.helpers import ts


@pytest.mark.asyncio
async def test_json_adapter_passes_structured_output_when_supported():
    signature = ts("question -> answer", instructions="Answer.")
    adapter = JSONAdapter()
    lm = CapturingLM(LM("openai/gpt-4o-mini"))
    with pytest.raises(StopAdapterCallCapture):
        await adapter.acall(
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
    assert hasattr(response_format, "model_json_schema")


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
        await adapter.acall(
            lm=lm,
            config={},
            task_spec=signature,
            demos=[],
            inputs={"question": "hi"},
            run=make_adapter_run(lm=lm, adapter=adapter),
        )
    request = lm.calls[0]["request"]
    assert request.config.response_format == {"type": "json_object"}
