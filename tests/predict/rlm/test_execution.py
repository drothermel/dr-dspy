import asyncio

import pytest

from dspy.adapters.types.tool import Tool
from dspy.history import REPLEntry, REPLHistory, REPLVariable, TurnEvent
from dspy.predict.agent_termination import AgentTerminationReason
from dspy.predict.code_execution import strip_python_fences
from dspy.predict.rlm import RLM
from dspy.predict.rlm import execution as rlm_execution
from dspy.primitives import CodeInterpreterError, FinalOutput
from dspy.task_spec import input_field, make_task_spec, output_field
from dspy.testing import DummyLM
from tests.mock_interpreter import MockInterpreter
from tests.predict.rlm.conftest import make_mock_predictor
from tests.task_spec.helpers import ts


class TestRLMCodeFenceParsing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("```python\nprint(1)\n```", "print(1)"),
            ("```py\nx = 1\nprint(x)\n```", "x = 1\nprint(x)"),
            ("```\nprint('no lang')\n```", "print('no lang')"),
            ("not fenced code", "not fenced code"),
            ("I'll inspect first.\n```python\nprint('hello')\n```\nThen I will submit.", "print('hello')"),
            ("```python\nprint(1)\n```\nsome trailing text", "print(1)"),
            ("```python\nprint('oops')", "print('oops')"),
            ("```\n```python\nprint(1)\n```\n```", "print(1)"),
            ("```\n```\nprint(2)\n```\n```", "print(2)"),
        ],
    )
    def teststrip_python_fences(self, raw, expected):
        assert strip_python_fences(raw) == expected

    def teststrip_python_fences_rejects_non_python_lang(self):
        with pytest.raises(SyntaxError, match="json"):
            strip_python_fences('```json\n{"a": 1}\n```')


class TestRLMFormatting:
    def test_format_history(self):
        history = REPLHistory()
        history = history.append_turn(TurnEvent(reasoning="Need to check the data", code="print(1)", output="1"))
        history = history.append_turn(TurnEvent(reasoning="Now calculate", code="x = 2", output=""))
        formatted = history.format()
        assert "Step 1" in formatted
        assert "Step 2" in formatted
        assert "print(1)" in formatted
        assert "Need to check" in formatted

    def test_format_history_empty(self):
        history = REPLHistory()
        formatted = history.format()
        assert "have not interacted with the REPL" in formatted

    def test_action_signature_has_iteration_field(self):
        rlm = RLM(ts("context -> answer"))
        action_sig = rlm.generate_action.task_spec
        assert "iteration" in action_sig.input_fields

    def test_format_output(self):
        formatted = rlm_execution.format_output("output text")
        assert "output text" in formatted

    def test_format_output_empty(self):
        formatted = rlm_execution.format_output("")
        assert "no output" in formatted.lower()

    def test_format_output_passthrough(self):
        formatted = rlm_execution.format_output("a" * 200)
        assert formatted == "a" * 200

    def test_format_variable_info_string(self):
        var = REPLVariable.from_value("context", "Hello world", preview_chars=5)
        formatted = var.format()
        assert "Variable: `context`" in formatted
        assert "Type: str" in formatted
        assert "11" in formatted
        assert "He" in formatted
        assert "ld" in formatted
        assert "..." in formatted

    def test_format_variable_info_dict(self):
        var = REPLVariable.from_value("data", {"key": "value"})
        formatted = var.format()
        assert "Variable: `data`" in formatted
        assert "Type: dict" in formatted
        assert "key" in formatted

    def test_build_variables_multiple(self):
        rlm = RLM(ts("context, query -> answer"))
        variables = rlm_execution.build_variables(rlm, context="Hello world", query="What is this?")
        assert len(variables) == 2
        formatted = "\n\n".join(v.format() for v in variables)
        assert "Variable: `context`" in formatted
        assert "Variable: `query`" in formatted
        assert "Hello world" in formatted
        assert "What is this?" in formatted


class TestREPLTypes:
    def test_repl_history_immutability(self):
        h1 = REPLHistory()
        h2 = h1.append_turn(TurnEvent(code="print(1)", output="1"))
        assert len(h1) == 0
        assert len(h2) == 1

    def test_repl_history_len_iter_bool(self):
        h = REPLHistory()
        assert len(h) == 0
        assert not bool(h)
        h = h.append_turn(TurnEvent(code="x = 1", output=""))
        h = h.append_turn(TurnEvent(code="x = 2", output=""))
        assert len(h) == 2
        assert bool(h)
        codes = [e.code for e in h]
        assert codes == ["x = 1", "x = 2"]

    def test_repl_entry_format(self):
        entry = REPLEntry(reasoning="test reason", code="print(1)", output="1")
        formatted = entry.format(index=0)
        assert "Step 1" in formatted
        assert "test reason" in formatted
        assert "print(1)" in formatted
        assert "1" in formatted

    def test_repl_entry_format_truncation(self):
        output = "a" * 100 + "b" * 100
        entry = REPLEntry(code="print('a' + 'b')", output=output)
        formatted = entry.format(index=0, max_output_chars=100)
        assert "a" * 50 in formatted
        assert "b" * 50 in formatted
        assert "100 characters omitted" in formatted
        assert "200 chars" in formatted

    def test_repl_entry_format_no_truncation(self, make_run):
        output = "a" * 50
        entry = REPLEntry(code="print('a')", output=output)
        formatted = entry.format(index=0, max_output_chars=100)
        assert output in formatted
        assert "omitted" not in formatted

    def test_repl_history_threads_max_output_chars(self, make_run):
        h = REPLHistory(max_output_chars=50)
        h2 = h.append_turn(TurnEvent(code="print(1)", output="a" * 100))
        assert h2.max_output_chars == 50
        formatted = h2.format()
        assert "50 characters omitted" in formatted

    def test_repl_variable_from_value(self, make_run):
        var = REPLVariable.from_value("test", "hello world")
        assert var.name == "test"
        assert var.type_name == "str"
        assert var.total_length == 11
        assert "hello world" in var.preview

    def test_repl_variable_truncation(self, make_run):
        var = REPLVariable.from_value("big", "a" * 500 + "b" * 500, preview_chars=50)
        assert var.preview.startswith("a" * 25)
        assert var.preview.endswith("b" * 25)
        assert "..." in var.preview

    def test_repl_variable_with_field_info(self, make_run):
        spec = make_task_spec(
            {
                "query": input_field(
                    "query",
                    desc="The user's question",
                    constraints="greater than or equal to: 0.0, less than or equal to: 100.0",
                )
            },
            instructions="Query field.",
        )
        field = spec.input_fields["query"]
        var = REPLVariable.from_value("query", "What is 2+2?", field=field)
        assert var.desc == "The user's question"
        assert "greater than or equal to" in var.constraints
        formatted = var.format()
        assert "Description: The user's question" in formatted
        assert "Constraints:" in formatted

    def test_repl_variable_without_field_info(self, make_run):
        var = REPLVariable.from_value("data", [1, 2, 3])
        assert var.desc == ""
        assert var.constraints == ""
        formatted = var.format()
        assert "Description:" not in formatted
        assert "Constraints:" not in formatted

    def test_build_variables_includes_field_metadata(self, make_run):
        QASig = make_task_spec(
            {
                "context": input_field("context", desc="Background information"),
                "question": input_field("question", desc="The question to answer"),
                "answer": output_field("answer", desc="The answer."),
            },
            instructions="Answer questions.",
            name="QASig",
        )
        rlm = RLM(QASig, max_iterations=3)
        variables = rlm_execution.build_variables(rlm, context="Some text", question="What?")
        context_var = next(v for v in variables if v.name == "context")
        assert context_var.desc == "Background information"
        question_var = next(v for v in variables if v.name == "question")
        assert question_var.desc == "The question to answer"


class TestRLMCallMethod:
    def test_call_executes_rlm(self, make_run):
        mock = MockInterpreter(responses=[FinalOutput({"answer": "42"})])
        rlm = RLM(ts("query -> answer"), max_iterations=3, interpreter=mock)
        rlm.generate_action = make_mock_predictor([{"reasoning": "Return answer", "code": 'SUBMIT("42")'}])
        run = make_run(lm=DummyLM([{}]))
        result = asyncio.run(rlm(query="What is the answer?", run=run))
        assert result.answer == "42"
        assert result.termination_reason == AgentTerminationReason.SUBMIT


class TestRLMMaxIterationsFallback:
    def test_max_iterations_triggers_extract(self, make_run):
        mock = MockInterpreter(responses=["exploring...", "still exploring...", "more exploring..."])
        rlm = RLM(ts("query -> answer"), max_iterations=3, interpreter=mock)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "Explore 1", "code": "print('exploring')"},
                {"reasoning": "Explore 2", "code": "print('exploring')"},
                {"reasoning": "Explore 3", "code": "print('exploring')"},
            ]
        )
        rlm.extract = make_mock_predictor([{"answer": "extracted_answer"}])
        run = make_run(lm=DummyLM([{}]))
        result = asyncio.run(rlm(query="test", run=run))
        assert result.answer == "extracted_answer"
        assert result.final_reasoning == "Extract forced final output"
        assert result.termination_reason == AgentTerminationReason.MAX_ITERS


class TestRLMToolExceptions:
    def test_tool_exception_returns_error_in_output(self, make_run):

        def failing_tool() -> str:
            raise RuntimeError("Tool failed!")

        mock = MockInterpreter(
            responses=[CodeInterpreterError("RuntimeError: Tool failed!"), FinalOutput({"answer": "recovered"})]
        )
        rlm = RLM(
            ts("query -> answer"),
            max_iterations=5,
            interpreter=mock,
            tools=[Tool(failing_tool, description="A tool that always fails.")],
        )
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "Call tool", "code": "failing_tool()"},
                {"reasoning": "Recover", "code": 'SUBMIT("recovered")'},
            ]
        )
        run = make_run(lm=DummyLM([{}]))
        result = asyncio.run(rlm(query="test", run=run))
        assert result.answer == "recovered"

    def test_runtime_error_history_uses_stripped_code(self, make_run):
        mock = MockInterpreter(
            responses=[CodeInterpreterError("NameError: name 'x' is not defined"), FinalOutput({"answer": "recovered"})]
        )
        rlm = RLM(ts("query -> answer"), max_iterations=5, interpreter=mock)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "Will fail", "code": "```python\nprint(x)\n```"},
                {"reasoning": "Recover", "code": 'SUBMIT("recovered")'},
            ]
        )
        run = make_run(lm=DummyLM([{}]))
        result = asyncio.run(rlm(query="test", run=run))
        assert result.answer == "recovered"
        first_step = result.turn_log.entries[0].model_dump()
        assert first_step["code"] == "print(x)"

    def test_syntax_error_from_execute_is_recoverable(self, make_run):
        mock = MockInterpreter(responses=[SyntaxError("invalid syntax"), FinalOutput({"answer": "recovered"})])
        rlm = RLM(ts("query -> answer"), max_iterations=5, interpreter=mock)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "Bad code", "code": "```python\ndef incomplete(\n```"},
                {"reasoning": "Recover", "code": 'SUBMIT("recovered")'},
            ]
        )
        run = make_run(lm=DummyLM([{}]))
        result = asyncio.run(rlm(query="test", run=run))
        assert result.answer == "recovered"
        assert result.turn_log.entries[0].output.startswith("[Error] invalid syntax")

    def test_syntax_error_fromstrip_python_fences_is_recoverable(self, make_run):
        mock = MockInterpreter(responses=[FinalOutput({"answer": "recovered"})])
        rlm = RLM(ts("query -> answer"), max_iterations=5, interpreter=mock)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "Wrong language", "code": "```bash\nls -la\n```"},
                {"reasoning": "Recover", "code": 'SUBMIT("recovered")'},
            ]
        )
        run = make_run(lm=DummyLM([{}]))
        result = asyncio.run(rlm(query="test", run=run))
        assert result.answer == "recovered"
        assert result.turn_log.entries[0].output.startswith("[Error]")


class TestRLMAsyncMock:
    @pytest.mark.asyncio
    async def test_aforward_basic(self, make_run):
        mock = MockInterpreter(responses=[FinalOutput({"answer": "42"})])
        rlm = RLM(ts("query -> answer"), max_iterations=3, interpreter=mock)
        rlm.generate_action = make_mock_predictor([{"reasoning": "Return answer", "code": 'SUBMIT("42")'}])
        run = make_run(lm=DummyLM([{}]))
        result = await rlm.aforward(query="What is the answer?", run=run)
        assert result.answer == "42"

    @pytest.mark.asyncio
    async def test_aforward_int_output_mock(self, make_run):
        mock = MockInterpreter(responses=[FinalOutput({"count": 42})])
        rlm = RLM(ts("query -> count: int"), max_iterations=3, interpreter=mock)
        rlm.generate_action = make_mock_predictor([{"reasoning": "Return count", "code": "SUBMIT(42)"}])
        run = make_run(lm=DummyLM([{}]))
        result = await rlm.aforward(query="count items", run=run)
        assert result.count == 42
        assert isinstance(result.count, int)

    @pytest.mark.asyncio
    async def test_aforward_multi_iteration_mock(self, make_run):
        mock = MockInterpreter(responses=["explored data", FinalOutput({"answer": "done"})])
        rlm = RLM(ts("query -> answer"), max_iterations=5, interpreter=mock)
        rlm.generate_action = make_mock_predictor(
            [
                {"reasoning": "Explore first", "code": "print('exploring')"},
                {"reasoning": "Now finish", "code": 'SUBMIT("done")'},
            ]
        )
        run = make_run(lm=DummyLM([{}]))
        result = await rlm.aforward(query="test", run=run)
        assert result.answer == "done"


class TestRLMTypeCoercionMock:
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
    def test_type_coercion(self, output_field, output_type, final_value, code, expected, make_run):
        mock = MockInterpreter(responses=[FinalOutput({output_field: final_value})])
        rlm = RLM(ts(f"query -> {output_field}: {output_type}"), max_iterations=3, interpreter=mock)
        rlm.generate_action = make_mock_predictor([{"reasoning": "Return value", "code": code}])
        run = make_run(lm=DummyLM([{}]))
        result = asyncio.run(rlm(query="test", run=run))
        assert getattr(result, output_field) == expected

    def test_type_error_retries(self, make_run):
        mock = MockInterpreter(responses=[FinalOutput({"answer": "maybe"}), FinalOutput({"answer": "yes"})])
        rlm = RLM(ts("query -> answer: Literal['yes', 'no']"), max_iterations=5, interpreter=mock)
        rlm.generate_action = make_mock_predictor(
            [{"reasoning": "Try maybe", "code": 'SUBMIT("maybe")'}, {"reasoning": "Try yes", "code": 'SUBMIT("yes")'}]
        )
        run = make_run(lm=DummyLM([{}]))
        result = asyncio.run(rlm(query="is it yes?", run=run))
        assert result.answer == "yes"
