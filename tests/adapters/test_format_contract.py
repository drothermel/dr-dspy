import pytest

from dspy.adapters.baml_adapter import BAMLAdapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.format.message_assembler import MESSAGE_BUILD_ORDER, MessageAssembler
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.xml_adapter import XMLAdapter
from dspy.history import TurnLog
from dspy.task_spec import input_field, make_task_spec, output_field
from tests.adapters.assertions import assert_content_contains, assert_message_roles, assert_multimodal_blocks
from tests.adapters.conftest import format_messages_and_lm_kwargs
from tests.adapters.scenarios.kitchen_sink import kitchen_sink_case
from tests.adapters.scenarios.qa import SIMPLE_QA_CONTRACT_INPUTS, SIMPLE_QA_CONTRACT_SIGNATURE


@pytest.mark.parametrize(
    "adapter_factory",
    [ChatAdapter, JSONAdapter, XMLAdapter, BAMLAdapter],
)
def test_simple_qa_message_contract(adapter_factory):
    adapter = adapter_factory()
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=adapter,
        task_spec=SIMPLE_QA_CONTRACT_SIGNATURE,
        demos=[],
        inputs=SIMPLE_QA_CONTRACT_INPUTS,
    )
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user"
    assert "question" in messages[-1]["content"]
    assert "response_format" not in lm_kwargs or adapter_factory is ChatAdapter or adapter_factory is XMLAdapter


def test_message_build_order_matches_contract():
    task_spec = make_task_spec(
        {
            "turn_log": input_field("turn_log", type_=TurnLog, desc="The history."),
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Answer the question.",
    )
    history = TurnLog.model_validate(
        {
            "turns": [
                {
                    "agent": "react",
                    "thought": "prior",
                    "tool_name": "search",
                    "tool_args": {},
                    "observation": "done",
                }
            ]
        }
    )
    demos = [{"turn_log": history, "question": "demo q", "answer": "demo a"}]

    class OrderCapturingChatAdapter(ChatAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.segment_order: list[str] = []
            self.message_assembler = _OrderCapturingMessageAssembler(self, self.segment_order)

    class _OrderCapturingMessageAssembler(MessageAssembler):
        def __init__(self, host, segment_order: list[str]) -> None:
            super().__init__(host)
            self._segment_order = segment_order

        def _append_system_message(self, *, messages, task_spec):
            self._segment_order.append("system")
            super()._append_system_message(messages=messages, task_spec=task_spec)

        def _append_demos(self, *, messages, task_spec, demos):
            self._segment_order.append("demos")
            super()._append_demos(messages=messages, task_spec=task_spec, demos=demos)

        def _append_conversation_history(self, *, messages, conversation_history):
            self._segment_order.append("conversation_history")
            super()._append_conversation_history(messages=messages, conversation_history=conversation_history)

        def _append_current_user_message(self, *, messages, task_spec, inputs):
            self._segment_order.append("current_user")
            super()._append_current_user_message(messages=messages, task_spec=task_spec, inputs=inputs)

    adapter = OrderCapturingChatAdapter()
    adapter.format(
        task_spec=task_spec,
        demos=demos,
        inputs={"turn_log": history, "question": "live q"},
    )
    assert adapter.segment_order == list(MESSAGE_BUILD_ORDER)


INCOMPLETE_DEMO_PREAMBLE = "This is an example of the task, though some input or output fields are not supplied."

KITCHEN_SINK_SYSTEM_FIELDS = (
    "turn_log",
    "image",
    "audio",
    "file",
    "document",
    "event",
    "tools",
    "profile",
    "context",
    "question",
    "answer",
    "verdict",
    "confidence",
)


@pytest.mark.parametrize("adapter_factory", [ChatAdapter, JSONAdapter, XMLAdapter])
def test_kitchen_sink_message_contract(adapter_factory):
    scenario = kitchen_sink_case()
    messages, _lm_kwargs = format_messages_and_lm_kwargs(
        adapter=adapter_factory(),
        task_spec=scenario.task_spec,
        demos=list(scenario.demos),
        inputs=scenario.inputs,
    )
    system_content = messages[0]["content"]
    assert isinstance(system_content, str)
    for field_name in KITCHEN_SINK_SYSTEM_FIELDS:
        assert field_name in system_content

    assert_message_roles(
        messages=messages,
        roles=["system", "user", "assistant", "user", "assistant", "user", "assistant", "user"],
    )
    assert INCOMPLETE_DEMO_PREAMBLE in _message_text(messages[1])
    assert INCOMPLETE_DEMO_PREAMBLE in _message_text(messages[3])

    assert_content_contains(content=_message_text(messages[5]), fragments=["Who is Ada?", "old note", "older note"])
    assert_content_contains(
        content=_message_text(messages[-1]),
        fragments=["What should the answer include?", "current context one", "Grace"],
    )

    if adapter_factory is ChatAdapter:
        assert_multimodal_blocks(
            message=messages[1],
            expected_blocks=[
                {"type": "image_url", "image_url": {"url": "https://example.com/demo.png"}},
                {"type": "input_audio", "input_audio": {"data": "REVNTw==", "format": "wav"}},
                {"type": "event", "event": {"label": "demo-event"}},
            ],
        )
        assert_multimodal_blocks(
            message=messages[-1],
            expected_blocks=[
                {"type": "image_url", "image_url": {"url": "https://example.com/current.png"}},
                {"type": "input_audio", "input_audio": {"data": "Q1VSUkVOVA==", "format": "wav"}},
                {"type": "event", "event": {"label": "current-event"}},
            ],
        )


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return ""
