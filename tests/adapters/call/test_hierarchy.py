from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.xml_adapter import XMLAdapter


def test_json_adapter_is_not_chat_adapter_subclass():
    assert isinstance(JSONAdapter(), Adapter)
    assert not isinstance(JSONAdapter(), ChatAdapter)


def test_xml_adapter_inherits_chat_parse_fallback():
    adapter = XMLAdapter()
    assert adapter.parse_fallback_policy is not None
