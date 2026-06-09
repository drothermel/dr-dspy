from dspy.adapters.base import Adapter
from dspy.adapters.call.policies.json_parse_fallback import JSONParseFallbackPolicy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.xml_adapter import XMLAdapter


def test_json_adapter_is_not_chat_adapter_subclass():
    assert isinstance(JSONAdapter(), Adapter)
    assert not isinstance(JSONAdapter(), ChatAdapter)


def test_xml_adapter_is_not_chat_adapter_subclass():
    assert isinstance(XMLAdapter(), Adapter)
    assert not isinstance(XMLAdapter(), ChatAdapter)


def test_xml_adapter_has_explicit_parse_fallback_policy():
    adapter = XMLAdapter()
    assert isinstance(adapter.parse_fallback_policy, JSONParseFallbackPolicy)
