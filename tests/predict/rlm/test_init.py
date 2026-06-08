import asyncio

import pytest

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types.tool import Tool
from dspy.clients.lm import LM
from dspy.predict.rlm import RLM
from dspy.utils.dummies import DummyLM
from tests.mock_interpreter import MockInterpreter
from tests.predict.rlm.conftest import FailingSubLM
from tests.task_spec.helpers import ts


class TestRLMInitialization:
    def test_basic_initialization(self, make_run):
        rlm = RLM(ts("context, query -> answer"), max_iterations=5)
        assert rlm.max_iterations == 5
        assert rlm.generate_action is not None
        assert rlm.extract is not None
        assert rlm.tools == {}
        assert "context" in rlm.task_spec.input_fields
        assert "query" in rlm.task_spec.input_fields
        assert "answer" in rlm.task_spec.output_fields

    def test_custom_signature(self, make_run):
        rlm = RLM(ts("document, question -> summary, key_facts"), max_iterations=5)
        assert "document" in rlm.task_spec.input_fields
        assert "question" in rlm.task_spec.input_fields
        assert "summary" in rlm.task_spec.output_fields
        assert "key_facts" in rlm.task_spec.output_fields

    def test_custom_tools(self, make_run):

        def custom_tool(x: str = "") -> str:
            return x.upper()

        rlm = RLM(
            ts("context -> answer"),
            max_iterations=5,
            tools=[Tool(custom_tool, description="Uppercase the input string.")],
        )
        assert "custom_tool" in rlm.tools
        assert len(rlm.tools) == 1

    @pytest.mark.parametrize("tool_name", ["invalid-name", "123start"])
    def test_tool_validation_invalid_identifier(self, tool_name, make_run):

        def my_tool() -> str:
            return "result"

        tool = Tool(my_tool, description="My tool.", name=tool_name)
        with pytest.raises(ValueError, match="must be a valid Python identifier"):
            RLM(ts("context -> answer"), tools=[tool])

    @pytest.mark.parametrize("tool_name", ["llm_query", "SUBMIT", "print"])
    def test_tool_validation_reserved_names(self, tool_name, make_run):

        def my_tool() -> str:
            return "result"

        tool = Tool(my_tool, description="My tool.", name=tool_name)
        with pytest.raises(ValueError, match="conflicts with built-in"):
            RLM(ts("context -> answer"), tools=[tool])

    @pytest.mark.parametrize("invalid_value", ["not a function", 123])
    def test_tool_validation_not_callable(self, invalid_value, make_run):
        with pytest.raises(TypeError, match="tools must be Tool instances"):
            RLM(ts("context -> answer"), tools=[invalid_value])

    def test_tools_dict_rejected(self, make_run):

        def my_tool() -> str:
            return "result"

        with pytest.raises(TypeError, match="tools must be a list, not a dict"):
            RLM(ts("context -> answer"), tools={"my_tool": my_tool})

    def test_optional_parameters(self, make_run):
        rlm = RLM(ts("context -> answer"))
        assert rlm.max_llm_calls == 50
        assert rlm.sub_lm is None
        assert rlm._interpreter is None
        mock = MockInterpreter()
        mock_lm = LM("openai/gpt-4o-mini")
        rlm = RLM(ts("context -> answer"), max_llm_calls=100, sub_lm=mock_lm, interpreter=mock)
        assert rlm.max_llm_calls == 100
        assert rlm.sub_lm is mock_lm
        assert rlm._interpreter is mock

    def test_validates_required_inputs(self, make_run):
        run = make_run(lm=DummyLM([{}]))
        mock = MockInterpreter(responses=["result"])
        rlm = RLM(ts("context, query -> answer"), max_iterations=3, interpreter=mock)
        with pytest.raises(ValueError, match="Missing required input"):
            asyncio.run(rlm(context="some context", run=run))
        rlm = RLM(ts("a, b, c -> answer"), max_iterations=3, interpreter=mock)
        with pytest.raises(ValueError) as exc_info:
            asyncio.run(rlm(a="only a", run=run))
        assert "b" in str(exc_info.value)
        assert "c" in str(exc_info.value)

    def test_batched_query_errors_have_clear_markers(self, make_run):
        sub_lm = FailingSubLM()
        run = make_run(lm=sub_lm)
        rlm = RLM(ts("context -> answer"), max_llm_calls=10, sub_lm=sub_lm)
        tools = rlm._make_llm_tools(run=run)
        results = tools["llm_query_batched"](prompts=["test prompt"])
        assert len(results) == 1
        assert results[0].startswith("[ERROR]")
        assert "LM failed" in results[0]

    def test_tools_call_counter_is_thread_safe(self, make_run):
        from concurrent.futures import ThreadPoolExecutor

        sub_lm = DummyLM([{"response": "response"} for _ in range(11)], adapter=ChatAdapter())
        run = make_run(lm=sub_lm, adapter=ChatAdapter())
        rlm = RLM(ts("context -> answer"), max_llm_calls=10, sub_lm=sub_lm)
        tools = rlm._make_llm_tools(run=run)
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
    def test_action_signature_structure(self):
        rlm = RLM(ts("document, question -> summary, answer"))
        action_sig = rlm.generate_action.task_spec
        assert "variables_info" in action_sig.input_fields
        assert "repl_history" in action_sig.input_fields
        assert "reasoning" in action_sig.output_fields
        assert "code" in action_sig.output_fields
        instructions = action_sig.instructions
        assert "llm_query" in instructions
        assert "llm_query_batched" in instructions
        assert "SUBMIT" in instructions
        assert "`document`" in instructions
        assert "`question`" in instructions
        assert "`summary`" in instructions
        assert "`answer`" in instructions

    def test_extract_signature_structure(self):
        rlm = RLM(ts("document, question -> summary, key_facts, confidence"))
        extract_sig = rlm.extract.task_spec
        assert "variables_info" in extract_sig.input_fields
        assert "repl_history" in extract_sig.input_fields
        assert "summary" in extract_sig.output_fields
        assert "key_facts" in extract_sig.output_fields
        assert "confidence" in extract_sig.output_fields
