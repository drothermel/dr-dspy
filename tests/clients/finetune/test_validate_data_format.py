import pytest

from dspy.clients.finetune import TrainDataFormat, validate_data_format

VALID_CHAT = [
    {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
    }
]

VALID_COMPLETION = [{"prompt": "Question?", "completion": "Answer."}]

VALID_COMPLETION_RESPONSE = [{"prompt": "Question?", "response": "Answer."}]


def test_grpo_chat_raises_value_error_not_assertion():
    with pytest.raises(ValueError, match="Data format grpo_chat is not supported"):
        validate_data_format(VALID_CHAT, TrainDataFormat.GRPO_CHAT)


def test_valid_chat_passes():
    validate_data_format(VALID_CHAT, TrainDataFormat.CHAT)


def test_valid_completion_passes():
    validate_data_format(VALID_COMPLETION, TrainDataFormat.COMPLETION)


def test_completion_with_response_alias_passes():
    validate_data_format(VALID_COMPLETION_RESPONSE, TrainDataFormat.COMPLETION)


def test_chat_missing_messages_raises():
    with pytest.raises(ValueError, match="Data format errors found"):
        validate_data_format([{"not_messages": []}], TrainDataFormat.CHAT)


def test_completion_missing_prompt_raises():
    with pytest.raises(ValueError, match="Data format errors found"):
        validate_data_format([{"completion": "only completion"}], TrainDataFormat.COMPLETION)
