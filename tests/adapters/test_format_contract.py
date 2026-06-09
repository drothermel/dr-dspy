import pytest

from dspy.adapters.baml_adapter import BAMLAdapter
from dspy.adapters.base.format import MESSAGE_BUILD_ORDER
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.xml_adapter import XMLAdapter
from dspy.history import TurnLog
from dspy.task_spec import input_field, make_task_spec, output_field
from tests.adapters.conftest import format_messages_and_lm_kwargs
from tests.adapters.scenarios.simple_qa import SIMPLE_QA_INPUTS, SIMPLE_QA_SIGNATURE


@pytest.mark.parametrize(
    "adapter_factory",
    [ChatAdapter, JSONAdapter, XMLAdapter, BAMLAdapter],
)
def test_simple_qa_message_contract(adapter_factory):
    adapter = adapter_factory()
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=adapter,
        task_spec=SIMPLE_QA_SIGNATURE,
        demos=[],
        inputs=SIMPLE_QA_INPUTS,
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
    history = TurnLog.model_validate({"turns": [{"thought": "prior", "observation": "done"}]})
    demos = [{"turn_log": history, "question": "demo q", "answer": "demo a"}]

    class OrderCapturingChatAdapter(ChatAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.segment_order: list[str] = []

        def _append_system_message(self, *, messages, task_spec):
            self.segment_order.append("system")
            super()._append_system_message(messages=messages, task_spec=task_spec)

        def _append_demos(self, *, messages, task_spec, demos):
            self.segment_order.append("demos")
            super()._append_demos(messages=messages, task_spec=task_spec, demos=demos)

        def _append_conversation_history(self, *, messages, conversation_history):
            self.segment_order.append("conversation_history")
            super()._append_conversation_history(messages=messages, conversation_history=conversation_history)

        def _append_current_user_message(self, *, messages, task_spec, inputs):
            self.segment_order.append("current_user")
            super()._append_current_user_message(messages=messages, task_spec=task_spec, inputs=inputs)

    adapter = OrderCapturingChatAdapter()
    adapter.format(
        task_spec=task_spec,
        demos=demos,
        inputs={"turn_log": history, "question": "live q"},
    )
    assert adapter.segment_order == list(MESSAGE_BUILD_ORDER)
