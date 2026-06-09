import pytest

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.xml_adapter import XMLAdapter
from dspy.clients.finetune import TrainDataFormat, infer_data_format


def test_chat_adapter_supports_finetune():
    assert ChatAdapter().capabilities.supports_finetune is True
    assert infer_data_format(ChatAdapter()) == TrainDataFormat.CHAT


def test_xml_adapter_supports_finetune_via_chat_mixin():
    assert XMLAdapter().capabilities.supports_finetune is True
    assert infer_data_format(XMLAdapter()) == TrainDataFormat.CHAT


def test_json_adapter_does_not_support_finetune():
    assert JSONAdapter().capabilities.supports_finetune is False
    with pytest.raises(ValueError, match="Could not infer the data format"):
        infer_data_format(JSONAdapter())


def test_field_value_role_capabilities():
    assert ChatAdapter().capabilities.field_value_role == "none"
    assert JSONAdapter().capabilities.field_value_role == "assistant"


def test_adapter_native_fc_and_structured_output_capabilities():
    assert ChatAdapter().capabilities.default_native_fc is False
    assert ChatAdapter().capabilities.supports_structured_output is False
    assert JSONAdapter().capabilities.default_native_fc is True
    assert JSONAdapter().capabilities.supports_structured_output is True
    assert XMLAdapter().capabilities.default_native_fc is False
