import pytest

from dspy.core.types import Assistant, LMMessage, LMMessageRole, System, User


@pytest.mark.parametrize(
    "role_str",
    ["system", "developer", "user", "assistant", "tool"],
)
def test_lm_message_coerces_valid_role_strings(role_str: str):
    message = LMMessage.model_validate({"role": role_str, "parts": [{"type": "text", "text": "hi"}]})
    assert message.role == LMMessageRole(role_str)


def test_lm_message_rejects_invalid_role():
    with pytest.raises(ValueError, match="Invalid LMMessage role"):
        LMMessage.model_validate({"role": "invalid", "parts": [{"type": "text", "text": "hi"}]})


def test_message_builders_use_enum_roles():
    assert System("hi").role == LMMessageRole.SYSTEM
    assert User("hi").role == LMMessageRole.USER
    assert Assistant("hi").role == LMMessageRole.ASSISTANT
