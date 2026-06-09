import pytest
from typing_extensions import override

from dspy.adapters.base import Adapter
from dspy.adapters.call.pipeline import AdapterCallPipeline
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.two_step_adapter import TwoStepAdapter
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.clients.lm import LM
from dspy.clients.openai_format.chat_request import message_to_openai_chat
from dspy.task_spec import input_field, make_task_spec, output_field
from dspy.testing import DummyLM
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


class FunctionCallingLM(DummyLM):
    @property
    @override
    def supports_function_calling(self):
        return True


class ReasoningFunctionCallingLM(FunctionCallingLM):
    @property
    @override
    def supports_reasoning(self):
        return True

    def __init__(self, answers):
        super().__init__(answers)
        self.kwargs["reasoning"] = {"effort": "low"}


@pytest.mark.asyncio
async def test_two_step_main_call_applies_preprocess_native_function_calling():
    def search(query: str) -> str:
        return query

    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `tool_calls`.",
    )
    main_lm = CapturingLM(FunctionCallingLM([{}]))
    extraction_lm = DummyLM([{"tool_calls": ToolCalls(tool_calls=[])}])
    adapter = TwoStepAdapter(
        extraction_model=extraction_lm,
        extraction_adapter=ChatAdapter(),
        use_native_function_calling=True,
    )
    run = make_adapter_run(lm=main_lm, adapter=adapter)
    with pytest.raises(StopAdapterCallCapture):
        await AdapterCallPipeline.execute(
            adapter,
            lm=main_lm,
            config={},
            task_spec=task_spec,
            demos=[],
            inputs={"question": "Q?", "tools": [Tool(search, description="Search for documents.")]},
            run=run,
        )
    request = main_lm.calls[0]["request"]
    assert request.tools
    user_message = message_to_openai_chat(request.messages[-1])
    assert "question:" in user_message["content"].lower()
    assert "tools:" not in user_message["content"].lower()


@pytest.mark.asyncio
async def test_two_step_main_call_applies_preprocess_native_reasoning():
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "reasoning": output_field("reasoning", type_=Reasoning, desc="The reasoning."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `question`, produce the fields `reasoning`, `answer`.",
    )
    main_lm = CapturingLM(ReasoningFunctionCallingLM([{}]))
    extraction_lm = DummyLM([{"reasoning": Reasoning(content="think"), "answer": "ok"}])
    adapter = TwoStepAdapter(
        extraction_model=extraction_lm,
        extraction_adapter=ChatAdapter(),
        use_native_function_calling=True,
    )
    run = make_adapter_run(lm=main_lm, adapter=adapter)
    with pytest.raises(StopAdapterCallCapture):
        await AdapterCallPipeline.execute(
            adapter,
            lm=main_lm,
            config={},
            task_spec=task_spec,
            demos=[],
            inputs={"question": "Q?"},
            run=run,
        )
    request = main_lm.calls[0]["request"]
    assert request.config.reasoning is not None
    assert request.config.reasoning.effort == "low"
