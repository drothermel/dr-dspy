from dspy.clients.lm.transport import DEFAULT_LITELLM_TIMEOUT_SECONDS, _with_default_timeout
from dspy.primitives import Completions


def test_completions_dict_path_copies_caller_data():
    source = {"answer": ["a", "b"]}
    completions = Completions(source)
    source["answer"].append("c")
    assert completions["answer"] == ["a", "b"]


def test_transport_applies_default_timeout():
    request = _with_default_timeout({"model": "openai/gpt-4.1-mini"})
    assert request["timeout"] == DEFAULT_LITELLM_TIMEOUT_SECONDS


def test_transport_preserves_explicit_timeout():
    request = _with_default_timeout({"model": "openai/gpt-4.1-mini", "timeout": 30})
    assert request["timeout"] == 30
