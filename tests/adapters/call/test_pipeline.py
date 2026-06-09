import pytest

from dspy.adapters.base import Adapter
from dspy.adapters.call.pipeline import AdapterCallPipeline
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.lm import LM
from tests.adapters.conftest import CapturingLM, StopAdapterCallCapture, make_adapter_run
from tests.task_spec.helpers import ts


@pytest.mark.asyncio
async def test_pipeline_preprocess_format_lm_boundary():
    signature = ts("question -> answer", instructions="Answer the question.")
    adapter = ChatAdapter()
    lm = CapturingLM(LM("openai/gpt-4o-mini"))
    run = make_adapter_run(lm=lm, adapter=adapter)
    with pytest.raises(StopAdapterCallCapture):
        await AdapterCallPipeline.execute(
            adapter,
            lm=lm,
            config={},
            task_spec=signature,
            demos=[],
            inputs={"question": "What is DSPy?"},
            run=run,
        )
    assert len(lm.calls) == 1
    request = lm.calls[0]["request"]
    assert request.messages[0].role == "system"
    assert request.messages[-1].role == "user"


def test_adapter_call_uses_pipeline():
    assert ChatAdapter.__call__ is Adapter.__call__
