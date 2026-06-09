from unittest.mock import AsyncMock, MagicMock

import pytest

from dspy.adapters.call.pipeline import AdapterCallPipeline
from dspy.adapters.call.wrappers import HintInjectingAdapter
from dspy.adapters.chat_adapter import ChatAdapter
from tests.adapters.conftest import CapturingLM, StopAdapterCallCapture, make_adapter_run
from tests.task_spec.helpers import ts


@pytest.mark.asyncio
async def test_hint_injecting_adapter_pipelines_inner_not_wrapper_format():
    inner = ChatAdapter()
    inner.format = MagicMock(wraps=inner.format)  # type: ignore[method-assign]
    task_spec = ts("question -> answer", instructions="Answer the question.")
    hinted_name = "predict"
    wrapper = HintInjectingAdapter(
        inner=inner,
        hint_map={hinted_name: "try again"},
        task_spec_to_name={task_spec: hinted_name},
    )

    lm = CapturingLM()
    run = make_adapter_run(lm=lm, adapter=wrapper)
    pipeline_spy = AsyncMock(wraps=AdapterCallPipeline.execute)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(AdapterCallPipeline, "execute", pipeline_spy)
        with pytest.raises(StopAdapterCallCapture):
            await wrapper(
                lm=lm,
                config={},
                task_spec=task_spec,
                demos=[],
                inputs={"question": "What is DSPy?"},
                run=run,
            )

    pipeline_spy.assert_awaited_once()
    pipeline_args = pipeline_spy.await_args
    assert pipeline_args is not None
    assert pipeline_args.args[0] is inner
    assert pipeline_args.kwargs["inputs"]["hint_"] == "try again"
    assert "hint_" in pipeline_args.kwargs["task_spec"].input_fields
    inner.format.assert_called_once()
    format_kwargs = inner.format.call_args.kwargs
    assert format_kwargs["inputs"]["hint_"] == "try again"


@pytest.mark.asyncio
async def test_hint_injecting_adapter_refreshes_policies_from_inner():
    inner = ChatAdapter()
    wrapper = HintInjectingAdapter(inner=inner, hint_map={}, task_spec_to_name={})
    from dspy.adapters.call.policies.response_format import NoOpResponseFormatPolicy

    new_policy = NoOpResponseFormatPolicy()
    inner.response_format_policy = new_policy
    wrapper._sync_policies_from_inner()
    assert wrapper.response_format_policy is new_policy
