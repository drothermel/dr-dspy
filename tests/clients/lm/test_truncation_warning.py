from unittest.mock import patch

from dspy.clients.lm import LM


def test_check_truncation_with_default_lm_kwargs():
    lm = LM("openai/gpt-4.1-mini")
    results = {"choices": [type("Choice", (), {"finish_reason": "length"})()]}
    with patch("dspy.clients.lm.client.logger") as logger:
        lm._check_truncation(results)
    assert logger.warning.called
    message = logger.warning.call_args.args[0]
    assert "max_tokens=unset" in message
    assert "currently unset" in message
