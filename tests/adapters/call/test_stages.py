from unittest.mock import patch

import pytest

from dspy.adapters.call.stages import invoke_adapter_lm, prepare_adapter_call
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types.citation import Citations
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.core.types import NativeAdaptationMode
from dspy.task_spec import input_field, make_task_spec, output_field
from dspy.testing import DummyLM
from tests.adapters.call.test_pipeline import FunctionCallingLM, ReasoningFunctionCallingLM
from tests.adapters.conftest import CapturingLM, StopAdapterCallCapture, make_adapter_run
from tests.adapters.test_native_adaptation import StubLM
from tests.task_spec.helpers import ts


def test_prepare_adapter_call_records_native_function_calling_mutations():
    def search(query: str) -> str:
        return query

    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Call tools.",
    )
    adapter = ChatAdapter(use_native_function_calling=True)
    lm = FunctionCallingLM([{}])
    prepared = prepare_adapter_call(
        adapter,
        lm=lm,
        config={},
        task_spec=task_spec,
        demos=[],
        inputs={"question": "Q?", "tools": [Tool(search, description="Search for documents.")]},
    )
    assert "removed field tool_calls" in prepared.mutations
    assert "removed field tools" in prepared.mutations
    assert "tools" not in prepared.processed_task_spec.input_fields
    assert "tool_calls" not in prepared.processed_task_spec.output_fields


def test_prepare_adapter_call_records_native_reasoning_removal():
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "reasoning": output_field("reasoning", type_=Reasoning, desc="The reasoning."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Answer with reasoning.",
    )
    adapter = ChatAdapter()
    lm = ReasoningFunctionCallingLM([{}])
    prepared = prepare_adapter_call(
        adapter,
        lm=lm,
        config={},
        task_spec=task_spec,
        demos=[],
        inputs={"question": "Q?"},
    )
    assert "removed field reasoning" in prepared.mutations
    assert "reasoning" not in prepared.processed_task_spec.output_fields


def test_prepare_adapter_call_records_native_citations_removal():
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "citations": output_field("citations", type_=Citations, desc="The citations."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Answer with citations.",
    )
    adapter = ChatAdapter(native_response_types=[Citations])
    lm = StubLM(citations_adaptation_mode=NativeAdaptationMode.SKIP)
    prepared = prepare_adapter_call(
        adapter,
        lm=lm,
        config={},
        task_spec=task_spec,
        demos=[],
        inputs={"question": "Q?"},
    )
    assert "removed field citations" in prepared.mutations
    assert "citations" not in prepared.processed_task_spec.output_fields


@pytest.mark.asyncio
async def test_invoke_adapter_lm_passes_mutations_to_compiled_call():
    task_spec = ts("question -> answer", instructions="Answer.")
    adapter = ChatAdapter()
    lm = CapturingLM(DummyLM([{"answer": "ok"}]))
    run = make_adapter_run(lm=lm, adapter=adapter)
    prepared = prepare_adapter_call(
        adapter,
        lm=lm,
        config={},
        task_spec=task_spec,
        demos=[],
        inputs={"question": "hi"},
    )
    captured: list[list[str]] = []

    def capture_validate(compiled, transparency):
        captured.append(list(compiled.task_spec_mutations))

    with (
        patch("dspy.adapters.call.stages.enforce_compiled_call_transparency", side_effect=capture_validate),
        pytest.raises(StopAdapterCallCapture),
    ):
        await invoke_adapter_lm(adapter, prepared, lm=lm, run=run)
    assert captured == [prepared.mutations]
