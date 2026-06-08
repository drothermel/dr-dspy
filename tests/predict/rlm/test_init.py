"""
Tests for the RLM (Recursive Language Model) module.
"""

import asyncio

import pytest

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types.tool import Tool
from dspy.clients.lm import LM
from dspy.dsp.utils.settings import settings
from dspy.predict.rlm import RLM
from dspy.utils.dummies import DummyLM
from tests.mock_interpreter import MockInterpreter
from tests.predict.rlm.conftest import (
    FailingSubLM,
)
from tests.task_spec.helpers import ts


class TestRLMInitialization:
    """Tests for RLM module initialization."""

    def test_basic_initialization(self):
        """Test RLM module initializes correctly with signature."""
        rlm = RLM(ts("context, query -> answer"), max_iterations=5)
        assert rlm.max_iterations == 5
        assert rlm.generate_action is not None
        assert rlm.extract is not None
        assert rlm.tools == {}  # No user tools provided
        assert "context" in rlm.task_spec.input_fields
        assert "query" in rlm.task_spec.input_fields
        assert "answer" in rlm.task_spec.output_fields

    def test_custom_signature(self):
        """Test RLM with custom signature."""
        rlm = RLM(ts("document, question -> summary, key_facts"), max_iterations=5)
        assert "document" in rlm.task_spec.input_fields
        assert "question" in rlm.task_spec.input_fields
        assert "summary" in rlm.task_spec.output_fields
        assert "key_facts" in rlm.task_spec.output_fields

    def test_custom_tools(self):
        """Test RLM with custom tools."""

        def custom_tool(x: str = "") -> str:
            return x.upper()

        rlm = RLM(ts("context -> answer"), max_iterations=5, tools=[custom_tool])
        assert "custom_tool" in rlm.tools
        assert len(rlm.tools) == 1  # Only user tools, not internal llm_query/llm_query_batched

    @pytest.mark.parametrize("tool_name", ["invalid-name", "123start"])
    def test_tool_validation_invalid_identifier(self, tool_name):
        """Test RLM rejects tool names that aren't valid Python identifiers."""

        def my_tool() -> str:
            return "result"

        tool = Tool(my_tool, description="My tool.", name=tool_name)
        with pytest.raises(ValueError, match="must be a valid Python identifier"):
            RLM(ts("context -> answer"), tools=[tool])

    @pytest.mark.parametrize("tool_name", ["llm_query", "SUBMIT", "print"])
    def test_tool_validation_reserved_names(self, tool_name):
        """Test RLM rejects tool names that conflict with built-in functions."""

        def my_tool() -> str:
            return "result"

        tool = Tool(my_tool, description="My tool.", name=tool_name)
        with pytest.raises(ValueError, match="conflicts with built-in"):
            RLM(ts("context -> answer"), tools=[tool])

    @pytest.mark.parametrize("invalid_value", ["not a function", 123])
    def test_tool_validation_not_callable(self, invalid_value):
        """Test RLM rejects tools that aren't callable."""
        with pytest.raises(TypeError, match="must be callable"):
            RLM(ts("context -> answer"), tools=[invalid_value])

    def test_tools_dict_rejected(self):
        """Test RLM rejects dict format for tools with helpful error."""

        def my_tool() -> str:
            return "result"

        with pytest.raises(TypeError, match="tools must be a list, not a dict"):
            RLM(ts("context -> answer"), tools={"my_tool": my_tool})  # ty:ignore[invalid-argument-type]

    def test_optional_parameters(self):
        """Test RLM optional parameters and their defaults."""

        # Test defaults
        rlm = RLM(ts("context -> answer"))
        assert rlm.max_llm_calls == 50
        assert rlm.sub_lm is None
        assert rlm._interpreter is None

        # Test custom values
        mock = MockInterpreter()
        mock_lm = LM("openai/gpt-4o-mini")
        rlm = RLM(ts("context -> answer"), max_llm_calls=100, sub_lm=mock_lm, interpreter=mock)
        assert rlm.max_llm_calls == 100
        assert rlm.sub_lm is mock_lm
        assert rlm._interpreter is mock

    def test_validates_required_inputs(self):
        """Test that aforward() raises ValueError for missing required inputs."""
        mock = MockInterpreter(responses=["result"])

        # Single missing input
        rlm = RLM(ts("context, query -> answer"), max_iterations=3, interpreter=mock)
        with pytest.raises(ValueError, match="Missing required input"):
            asyncio.run(rlm(context="some context"))  # Missing 'query'

        # Multiple missing inputs - all should be reported
        rlm = RLM(ts("a, b, c -> answer"), max_iterations=3, interpreter=mock)
        with pytest.raises(ValueError) as exc_info:  # noqa: PT011
            asyncio.run(rlm(a="only a"))  # Missing 'b' and 'c'
        assert "b" in str(exc_info.value)
        assert "c" in str(exc_info.value)

    def test_batched_query_errors_have_clear_markers(self):
        """Test that errors in llm_query_batched are prefixed with [ERROR]."""
        settings.configure(adapter=ChatAdapter())
        rlm = RLM(ts("context -> answer"), max_llm_calls=10, sub_lm=FailingSubLM())
        tools = rlm._make_llm_tools()

        results = tools["llm_query_batched"](prompts=["test prompt"])
        assert len(results) == 1
        assert results[0].startswith("[ERROR]")
        assert "LM failed" in results[0]

    def test_tools_call_counter_is_thread_safe(self):
        """Test that the LLM call counter is thread-safe for concurrent llm_query_batched calls.

        The call counter must be protected by a lock since llm_query_batched uses
        ThreadPoolExecutor for concurrent execution.
        """
        from concurrent.futures import ThreadPoolExecutor

        sub_lm = DummyLM([{"response": "response"} for _ in range(11)], adapter=ChatAdapter())
        settings.configure(lm=sub_lm, adapter=ChatAdapter())
        rlm = RLM(ts("context -> answer"), max_llm_calls=10, sub_lm=sub_lm)
        tools = rlm._make_llm_tools()

        call_count = [0]
        errors = []

        def make_call():
            try:
                tools["llm_query"](prompt="test")
                call_count[0] += 1
            except RuntimeError as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_call) for _ in range(10)]
            for f in futures:
                f.result()

        assert call_count[0] == 10, f"Expected 10 successful calls, got {call_count[0]}"
        assert len(errors) == 0, f"Unexpected errors: {errors}"

        with pytest.raises(RuntimeError, match="LLM call limit exceeded"):
            tools["llm_query"](prompt="one more")


class TestRLMDynamicSignature:
    """Tests for the dynamically built RLM signatures."""

    def test_action_signature_structure(self):
        """Test action signature has required fields and instructions."""
        rlm = RLM(ts("document, question -> summary, answer"))
        action_sig = rlm.generate_action.task_spec

        # Required input/output fields
        assert "variables_info" in action_sig.input_fields
        assert "repl_history" in action_sig.input_fields
        assert "reasoning" in action_sig.output_fields
        assert "code" in action_sig.output_fields

        # Instructions mention key tools and variables
        instructions = action_sig.instructions
        assert "llm_query" in instructions
        assert "llm_query_batched" in instructions
        assert "SUBMIT" in instructions
        assert "`document`" in instructions
        assert "`question`" in instructions
        assert "`summary`" in instructions
        assert "`answer`" in instructions

    def test_extract_signature_structure(self):
        """Test extract signature has required fields for all outputs."""
        rlm = RLM(ts("document, question -> summary, key_facts, confidence"))
        extract_sig = rlm.extract.task_spec
        assert "variables_info" in extract_sig.input_fields
        assert "repl_history" in extract_sig.input_fields
        assert "summary" in extract_sig.output_fields
        assert "key_facts" in extract_sig.output_fields
        assert "confidence" in extract_sig.output_fields
