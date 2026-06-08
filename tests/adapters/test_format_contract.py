import pytest

from dspy.adapters.baml_adapter import BAMLAdapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.xml_adapter import XMLAdapter
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
