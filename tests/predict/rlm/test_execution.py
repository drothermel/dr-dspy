"""
Tests for the RLM (Recursive Language Model) module.
"""

import asyncio

import pytest

from dspy.predict.rlm import RLM, _strip_code_fences
from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from dspy.primitives.repl_types import REPLEntry, REPLHistory, REPLVariable
from dspy.task_spec import FieldSpec, make_task_spec
from dspy.task_spec.pydantic_bridge import task_spec_input_field_infos
from tests.mock_interpreter import MockInterpreter
from tests.predict.rlm.conftest import (
    make_mock_predictor,
)
from tests.task_spec.helpers import ts


class TestRLMCodeFenceParsing:
    """Tests for robust fenced-code extraction."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # Standard python fence
            ("```python\nprint(1)\n```", "print(1)"),
            ("```py\nx = 1\nprint(x)\n```", "x = 1\nprint(x)"),
            # Bare fence (no language tag)
            ("```\nprint('no lang')\n```", "print('no lang')"),
            # No fences at all
            ("not fenced code", "not fenced code"),
            # Text before fence (preamble is skipped)
            ("I'll inspect first.\n```python\nprint('hello')\n```\nThen I will submit.", "print('hello')"),
            # Text after closing fence (ignored)
            ("```python\nprint(1)\n```\nsome trailing text", "print(1)"),
            # Unclosed fence (just return the body)
            ("```python\nprint('oops')", "print('oops')"),
            # Double fences (outer decorative ```)
            ("```\n```python\nprint(1)\n```\n```", "print(1)"),
            ("```\n```\nprint(2)\n```\n```", "print(2)"),
        ],
    )
    def test_strip_code_fences(self, raw, expected):
        assert _strip_code_fences(raw) == expected

    def test_strip_code_fences_rejects_non_python_lang(self):
        with pytest.raises(SyntaxError, match="json"):
            _strip_code_fences('```json\n{"a": 1}\n```')


class TestRLMFormatting:
    """Tests for RLM formatting helpers."""

    def test_format_history(self):
        """Test history formatting using REPLHistory."""
        history = REPLHistory()
        history = history.append(reasoning="Need to check the data", code="print(1)", output="1")
        history = history.append(reasoning="Now calculate", code="x = 2", output="")
        formatted = history.format()
        assert "Step 1" in formatted
        assert "Step 2" in formatted
        assert "print(1)" in formatted
        assert "Need to check" in formatted

    def test_format_history_empty(self):
        """Test history formatting with empty history."""
        history = REPLHistory()
        formatted = history.format()
        assert "have not interacted with the REPL" in formatted

    def test_action_signature_has_iteration_field(self):
        """Test action signature includes iteration input field."""
        rlm = RLM(ts("context -> answer"))
        action_sig = rlm.generate_action.task_spec
        assert "iteration" in action_sig.input_fields

    def test_format_output(self):
        """Test output formatting."""
        rlm = RLM(ts("context -> answer"))
        formatted = rlm._format_output("output text")
        assert "output text" in formatted

    def test_format_output_empty(self):
        """Test output formatting with empty output."""
        rlm = RLM(ts("context -> answer"))
        formatted = rlm._format_output("")
        assert "no output" in formatted.lower()

    def test_format_output_passthrough(self):
        """Test that _format_output passes through non-empty output without truncation."""
        rlm = RLM(ts("context -> answer"), max_output_chars=100)
        long_output = "a" * 200
        formatted = rlm._format_output(long_output)
        assert formatted == long_output

    def test_format_variable_info_string(self):
        """Test variable info formatting for string value using REPLVariable."""
        var = REPLVariable.from_value("context", "Hello world", preview_chars=5)
        formatted = var.format()
        assert "Variable: `context`" in formatted
        assert "Type: str" in formatted
        assert "11" in formatted  # length
        assert "He" in formatted  # head
        assert "ld" in formatted  # tail
        assert "..." in formatted  # truncation indicator

    def test_format_variable_info_dict(self):
        """Test variable info formatting for dict value using REPLVariable."""
        var = REPLVariable.from_value("data", {"key": "value"})
        formatted = var.format()
        assert "Variable: `data`" in formatted
        assert "Type: dict" in formatted
        assert "key" in formatted

    def test_build_variables_multiple(self):
        """Test building multiple variables."""
        rlm = RLM(ts("context, query -> answer"))
        variables = rlm._build_variables(context="Hello world", query="What is this?")
        assert len(variables) == 2
        formatted = "\n\n".join(v.format() for v in variables)
        assert "Variable: `context`" in formatted
        assert "Variable: `query`" in formatted
        assert "Hello world" in formatted
        assert "What is this?" in formatted


class TestREPLTypes:
    """Tests for the REPL type classes."""

    def test_repl_history_immutability(self):
        """Test that REPLHistory.append() returns new instance."""
        h1 = REPLHistory()
        h2 = h1.append(code="print(1)", output="1")
        assert len(h1) == 0  # Original unchanged
        assert len(h2) == 1  # New has entry

    def test_repl_history_len_iter_bool(self):
        """Test REPLHistory list-like interface."""
        h = REPLHistory()
        assert len(h) == 0
        assert not bool(h)

        h = h.append(code="x = 1", output="")
        h = h.append(code="x = 2", output="")
        assert len(h) == 2
        assert bool(h)

        codes = [e.code for e in h]
        assert codes == ["x = 1", "x = 2"]

    def test_repl_entry_format(self):
        """Test REPLEntry formatting."""
        entry = REPLEntry(reasoning="test reason", code="print(1)", output="1")
        formatted = entry.format(index=0)
        assert "Step 1" in formatted
        assert "test reason" in formatted
        assert "print(1)" in formatted
        assert "1" in formatted

    def test_repl_entry_format_truncation(self):
        """Test REPLEntry.format() truncates with head+tail and shows true length."""
        output = "a" * 100 + "b" * 100
        entry = REPLEntry(code="print('a' + 'b')", output=output)
        formatted = entry.format(index=0, max_output_chars=100)
        # Head and tail preserved
        assert "a" * 50 in formatted
        assert "b" * 50 in formatted
        assert "100 characters omitted" in formatted
        # True original length shown in header
        assert "200 chars" in formatted

    def test_repl_entry_format_no_truncation(self):
        """Test REPLEntry.format() passes short output through without truncation."""
        output = "a" * 50
        entry = REPLEntry(code="print('a')", output=output)
        formatted = entry.format(index=0, max_output_chars=100)
        assert output in formatted
        assert "omitted" not in formatted

    def test_repl_history_threads_max_output_chars(self):
        """Test REPLHistory carries max_output_chars through append()."""
        h = REPLHistory(max_output_chars=50)
        h2 = h.append(code="print(1)", output="a" * 100)
        assert h2.max_output_chars == 50
        # Formatting should truncate at 50 chars
        formatted = h2.format()
        assert "50 characters omitted" in formatted

    def test_repl_variable_from_value(self):
        """Test REPLVariable.from_value() factory."""
        var = REPLVariable.from_value("test", "hello world")
        assert var.name == "test"
        assert var.type_name == "str"
        assert var.total_length == 11
        assert "hello world" in var.preview

    def test_repl_variable_truncation(self):
        """Test REPLVariable preview shows head and tail."""
        var = REPLVariable.from_value("big", "a" * 500 + "b" * 500, preview_chars=50)
        assert var.preview.startswith("a" * 25)
        assert var.preview.endswith("b" * 25)
        assert "..." in var.preview

    def test_repl_variable_with_field_info(self):
        """Test REPLVariable includes desc and constraints from field_info."""

        spec = make_task_spec(
            {
                "query": FieldSpec.input(
                    "query",
                    desc="The user's question",
                    constraints="greater than or equal to: 0.0, less than or equal to: 100.0",
                ),
            },
            instructions="Query field.",
        )
        field = task_spec_input_field_infos(spec)["query"]

        var = REPLVariable.from_value("query", "What is 2+2?", field_info=field)
        assert var.desc == "The user's question"
        assert "greater than or equal to" in var.constraints

        # Verify format includes the metadata
        formatted = var.format()
        assert "Description: The user's question" in formatted
        assert "Constraints:" in formatted

    def test_repl_variable_without_field_info(self):
        """Test REPLVariable works without field_info."""
        var = REPLVariable.from_value("data", [1, 2, 3])
        assert var.desc == ""
        assert var.constraints == ""

        # Format should not include empty desc/constraints lines
        formatted = var.format()
        assert "Description:" not in formatted
        assert "Constraints:" not in formatted

    def test_build_variables_includes_field_metadata(self):
        """Test _build_variables passes field_info to REPLVariable."""

        QASig = make_task_spec(
            {
                "context": FieldSpec.input("context", desc="Background information"),
                "question": FieldSpec.input("question", desc="The question to answer"),
                "answer": FieldSpec.output("answer"),
            },
            instructions="Answer questions.",
            name="QASig",
        )

        rlm = RLM(QASig, max_iterations=3)
        variables = rlm._build_variables(context="Some text", question="What?")

        # Find the context variable
        context_var = next(v for v in variables if v.name == "context")
        assert context_var.desc == "Background information"

        question_var = next(v for v in variables if v.name == "question")
        assert question_var.desc == "The question to answer"


class TestRLMCallMethod:
    """Tests for RLM __call__ method."""

    def test_call_executes_rlm(self):
        """Test that __call__ executes RLM and returns a Prediction."""
        mock = MockInterpreter(responses=[FinalOutput({"answer": "42"})])
        rlm = RLM(ts("query -> answer"), max_iterations=3, interpreter=mock)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "Return answer", "code": 'SUBMIT("42")'},
            ]
        )

        result = asyncio.run(rlm(query="What is the answer?"))
        assert result.answer == "42"


class TestRLMMaxIterationsFallback:
    """Tests for max_iterations reached and extract fallback."""

    def test_max_iterations_triggers_extract(self):
        """Test that reaching max_iterations uses extract fallback."""
        mock = MockInterpreter(
            responses=[
                "exploring...",
                "still exploring...",
                "more exploring...",
            ]
        )
        rlm = RLM(ts("query -> answer"), max_iterations=3, interpreter=mock)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "Explore 1", "code": "print('exploring')"},
                {"reasoning": "Explore 2", "code": "print('exploring')"},
                {"reasoning": "Explore 3", "code": "print('exploring')"},
            ]
        )
        # Mock the extract predictor to return a value
        rlm.extract = make_mock_predictor(
            [
                {"answer": "extracted_answer"},
            ]
        )

        result = asyncio.run(rlm(query="test"))
        assert result.answer == "extracted_answer"
        assert result.final_reasoning == "Extract forced final output"


class TestRLMToolExceptions:
    """Tests for tool exception handling."""

    def test_tool_exception_returns_error_in_output(self):
        """Test that tool exceptions are caught and returned as errors."""

        def failing_tool() -> str:
            raise RuntimeError("Tool failed!")

        mock = MockInterpreter(
            responses=[
                CodeInterpreterError("RuntimeError: Tool failed!"),
                FinalOutput({"answer": "recovered"}),
            ]
        )
        rlm = RLM(ts("query -> answer"), max_iterations=5, interpreter=mock, tools=[failing_tool])
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "Call tool", "code": "failing_tool()"},
                {"reasoning": "Recover", "code": 'SUBMIT("recovered")'},
            ]
        )

        result = asyncio.run(rlm(query="test"))
        assert result.answer == "recovered"

    def test_runtime_error_history_uses_stripped_code(self):
        """Runtime execution failures should preserve stripped code in history."""
        mock = MockInterpreter(
            responses=[
                CodeInterpreterError("NameError: name 'x' is not defined"),
                FinalOutput({"answer": "recovered"}),
            ]
        )
        rlm = RLM(ts("query -> answer"), max_iterations=5, interpreter=mock)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "Will fail", "code": "```python\nprint(x)\n```"},
                {"reasoning": "Recover", "code": 'SUBMIT("recovered")'},
            ]
        )

        result = asyncio.run(rlm(query="test"))
        assert result.answer == "recovered"
        first_step = result.trajectory[0]
        assert first_step["code"] == "print(x)"

    def test_syntax_error_from_execute_is_recoverable(self):
        """SyntaxError from interpreter.execute should be surfaced as an iteration error."""
        mock = MockInterpreter(
            responses=[
                SyntaxError("invalid syntax"),
                FinalOutput({"answer": "recovered"}),
            ]
        )
        rlm = RLM(ts("query -> answer"), max_iterations=5, interpreter=mock)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "Bad code", "code": "```python\ndef incomplete(\n```"},
                {"reasoning": "Recover", "code": 'SUBMIT("recovered")'},
            ]
        )

        result = asyncio.run(rlm(query="test"))
        assert result.answer == "recovered"
        assert result.trajectory[0]["output"].startswith("[Error] invalid syntax")

    def test_syntax_error_from_strip_code_fences_is_recoverable(self):
        """SyntaxError raised by _strip_code_fences (e.g. non-Python fence tag) should be recoverable."""
        mock = MockInterpreter(
            responses=[
                FinalOutput({"answer": "recovered"}),
            ]
        )
        rlm = RLM(ts("query -> answer"), max_iterations=5, interpreter=mock)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "Wrong language", "code": "```bash\nls -la\n```"},
                {"reasoning": "Recover", "code": 'SUBMIT("recovered")'},
            ]
        )

        result = asyncio.run(rlm(query="test"))
        assert result.answer == "recovered"
        assert result.trajectory[0]["output"].startswith("[Error]")


class TestRLMAsyncMock:
    """Unit tests for RLM aforward() using MockInterpreter (no Deno required)."""

    @pytest.mark.asyncio
    async def test_aforward_basic(self):
        """Test aforward() returns Prediction with expected output (MockInterpreter)."""
        mock = MockInterpreter(responses=[FinalOutput({"answer": "42"})])
        rlm = RLM(ts("query -> answer"), max_iterations=3, interpreter=mock)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "Return answer", "code": 'SUBMIT("42")'},
            ]
        )

        result = await rlm.aforward(query="What is the answer?")
        assert result.answer == "42"

    @pytest.mark.asyncio
    async def test_aforward_int_output_mock(self):
        """Test aforward() returns int when signature expects int (MockInterpreter)."""
        mock = MockInterpreter(responses=[FinalOutput({"count": 42})])
        rlm = RLM(ts("query -> count: int"), max_iterations=3, interpreter=mock)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "Return count", "code": "SUBMIT(42)"},
            ]
        )

        result = await rlm.aforward(query="count items")
        assert result.count == 42
        assert isinstance(result.count, int)

    @pytest.mark.asyncio
    async def test_aforward_multi_iteration_mock(self):
        """Test aforward() handles multiple iterations before SUBMIT (MockInterpreter)."""
        mock = MockInterpreter(
            responses=[
                "explored data",
                FinalOutput({"answer": "done"}),
            ]
        )
        rlm = RLM(ts("query -> answer"), max_iterations=5, interpreter=mock)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "Explore first", "code": "print('exploring')"},
                {"reasoning": "Now finish", "code": 'SUBMIT("done")'},
            ]
        )

        result = await rlm.aforward(query="test")
        assert result.answer == "done"


class TestRLMTypeCoercionMock:
    """Unit tests for RLM type coercion using MockInterpreter (no Deno required)."""

    @pytest.mark.parametrize(
        ("output_field", "output_type", "final_value", "code", "expected"),
        [
            ("count", "int", 42, "SUBMIT(42)", 42),
            ("score", "float", 3.14, "SUBMIT(3.14)", 3.14),
            ("valid", "bool", True, "SUBMIT(True)", True),
            ("numbers", "list[int]", [1, 2, 3], "SUBMIT([1, 2, 3])", [1, 2, 3]),
            ("answer", "Literal['yes', 'no']", "yes", 'SUBMIT("yes")', "yes"),
        ],
    )
    def test_type_coercion(self, output_field, output_type, final_value, code, expected):
        """Test RLM type coercion for various types (MockInterpreter)."""
        mock = MockInterpreter(responses=[FinalOutput({output_field: final_value})])
        rlm = RLM(ts(f"query -> {output_field}: {output_type}"), max_iterations=3, interpreter=mock)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "Return value", "code": code},
            ]
        )

        result = asyncio.run(rlm(query="test"))
        assert getattr(result, output_field) == expected

    def test_type_error_retries(self):
        """Test RLM retries when type validation fails (MockInterpreter)."""
        mock = MockInterpreter(
            responses=[
                FinalOutput({"answer": "maybe"}),  # Invalid for Literal
                FinalOutput({"answer": "yes"}),  # Valid
            ]
        )
        rlm = RLM(ts("query -> answer: Literal['yes', 'no']"), max_iterations=5, interpreter=mock)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "Try maybe", "code": 'SUBMIT("maybe")'},
                {"reasoning": "Try yes", "code": 'SUBMIT("yes")'},
            ]
        )

        result = asyncio.run(rlm(query="is it yes?"))
        assert result.answer == "yes"
