import pydantic
import pytest

from dspy.core.types import (
    LMConfig,
    LMPromptCacheConfig,
    LMProviderOptions,
    LMReasoningConfig,
    LMToolChoice,
    ReasoningEffort,
    coerce_lm_config,
    merge_lm_config,
    merge_lm_request_config,
    merge_provider_options,
)
from dspy.core.types.embedding_options import EmbedderOptions, merge_embedder_options


def test_merge_lm_config_merges_extensions():
    left = LMConfig(extensions={"a": 1})
    right = LMConfig(extensions={"b": 2})
    merged = merge_lm_config(left, right)
    assert merged is not None
    assert merged.extensions == {"a": 1, "b": 2}


def test_merge_lm_config_clears_extensions_when_none():
    config = LMConfig(extensions={"a": 1})
    merged = merge_lm_config(config, LMConfig.model_construct(extensions=None))
    assert merged is not None
    assert merged.extensions == {}


def test_merge_lm_config_empty_right_extensions_preserves_left_keys():
    left = LMConfig(extensions={"a": 1})
    right = LMConfig(extensions={})
    merged = merge_lm_config(left, right)
    assert merged is not None
    assert merged.extensions == {"a": 1}


def test_merge_lm_config_extension_key_override():
    left = LMConfig(extensions={"a": 1})
    right = LMConfig(extensions={"a": 2})
    merged = merge_lm_config(left, right)
    assert merged is not None
    assert merged.extensions == {"a": 2}


def test_merge_lm_config_nested_shallow_merge():
    left = LMConfig(reasoning=LMReasoningConfig(effort=ReasoningEffort.LOW))
    right = LMConfig(reasoning=LMReasoningConfig(summary="auto"))
    merged = merge_lm_config(left, right)
    assert merged is not None
    assert merged.reasoning is not None
    assert merged.reasoning.effort == ReasoningEffort.LOW
    assert merged.reasoning.summary == "auto"


def test_merge_lm_config_nested_none_clears():
    left = LMConfig(reasoning=LMReasoningConfig(effort=ReasoningEffort.LOW))
    right = LMConfig(reasoning=None)
    merged = merge_lm_config(left, right)
    assert merged is not None
    assert merged.reasoning is None


def test_merge_lm_config_scalar_none_clears():
    left = LMConfig(temperature=0.2)
    right = LMConfig(temperature=None)
    merged = merge_lm_config(left, right)
    assert merged is not None
    assert merged.temperature is None


def test_merge_provider_options_extensions_union():
    left = LMProviderOptions(extensions={"a": 1})
    right = LMProviderOptions(extensions={"b": 2})
    merged = merge_provider_options(left, right)
    assert merged is not None
    assert merged.extensions == {"a": 1, "b": 2}


def test_merge_embedder_options_scalar_override():
    left = EmbedderOptions(dimensions=1536, timeout=10.0)
    right = EmbedderOptions(timeout=30.0)
    merged = merge_embedder_options(left, right)
    assert merged.dimensions == 1536
    assert merged.timeout == 30.0


def test_merge_embedder_options_explicit_none_clears():
    left = EmbedderOptions(dimensions=1536, timeout=10.0)
    right = EmbedderOptions.model_construct(timeout=None)
    merged = merge_embedder_options(left, right)
    assert merged.dimensions == 1536
    assert merged.timeout is None


def test_merge_provider_options_scalar_override():
    left = LMProviderOptions(api_key="left-key", timeout=10.0)
    right = LMProviderOptions(api_key="right-key")
    merged = merge_provider_options(left, right)
    assert merged is not None
    assert merged.api_key == "right-key"
    assert merged.timeout == 10.0


def test_lm_config_rejects_unknown_top_level_keys():
    with pytest.raises(pydantic.ValidationError):
        LMConfig.from_kwargs(temperature=0.2, provider_flag=True)


def test_lm_config_accepts_canonical_nested_fields():
    config = LMConfig(
        reasoning=LMReasoningConfig(effort=ReasoningEffort.HIGH, summary="auto"),
        tool_choice=LMToolChoice(mode="auto", parallel=False),
        prompt_cache=LMPromptCacheConfig(enabled=True, key="prompt-cache"),
        extensions={"provider_flag": True},
    )
    assert config.reasoning is not None
    assert config.tool_choice is not None
    assert config.prompt_cache is not None
    assert config.reasoning.effort == "high"
    assert config.reasoning.summary == "auto"
    assert config.tool_choice.mode == "auto"
    assert config.tool_choice.parallel is False
    assert config.prompt_cache.enabled is True
    assert config.prompt_cache.key == "prompt-cache"
    assert config.extensions == {"provider_flag": True}


def test_coerce_lm_config_rejects_reasoning_effort():
    with pytest.raises(ValueError, match="reasoning_effort"):
        coerce_lm_config({"reasoning_effort": "low"})


def test_coerce_lm_config_rejects_max_completion_tokens():
    with pytest.raises(ValueError, match="max_completion_tokens"):
        coerce_lm_config({"max_completion_tokens": 100})


def test_coerce_lm_config_rejects_bool_prompt_cache():
    with pytest.raises(TypeError, match="bool prompt_cache"):
        coerce_lm_config({"prompt_cache": True})


def test_merge_lm_request_config_per_call_response_format_wins():
    lm = type("_LMDefaults", (), {"kwargs": {"response_format": {"type": "json_object"}}})()
    merged = merge_lm_request_config(lm, LMConfig(response_format={"type": "json_schema"}))
    assert merged.response_format == {"type": "json_schema"}
