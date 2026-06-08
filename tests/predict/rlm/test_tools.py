"""
Tests for the RLM (Recursive Language Model) module.
"""

import pytest

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from dspy.primitives.python_interpreter import PythonInterpreter
from tests.mock_interpreter import MockInterpreter


class TestMockInterpreter:
    """Unit tests for MockInterpreter."""

    def test_scripted_responses(self):
        """Test that MockInterpreter returns scripted responses in order."""
        mock = MockInterpreter(responses=["first", "second", "third"])
        assert mock.execute("code1") == "first"
        assert mock.execute("code2") == "second"
        assert mock.execute("code3") == "third"

    def test_returns_final_output_result(self):
        """Test that MockInterpreter can return FinalOutput."""
        mock = MockInterpreter(responses=["exploring", FinalOutput("42")])
        assert mock.execute("print(len(data))") == "exploring"
        result = mock.execute("SUBMIT('42')")
        assert isinstance(result, FinalOutput)
        assert result.output == "42"

    def test_raises_exception_from_responses(self):
        """Test that MockInterpreter raises exceptions from responses."""
        mock = MockInterpreter(responses=["ok", CodeInterpreterError("undefined variable")])
        assert mock.execute("code1") == "ok"
        with pytest.raises(CodeInterpreterError, match="undefined variable"):
            mock.execute("code2")

    def test_records_call_history(self):
        """Test that MockInterpreter records call history for test assertions."""
        mock = MockInterpreter(responses=["resp"])
        mock.execute("print(1)", variables={"x": 10})
        assert mock.call_history == [("print(1)", {"x": 10})]


# ============================================================================
# Unit Tests: RLM Module (no interpreter needed)
# ============================================================================


@pytest.mark.deno
class TestPythonInterpreter:
    """Integration tests for the secure sandbox with tool support."""

    def test_start_prewarms_sandbox(self):
        """Test that start() pre-warms the sandbox."""
        interp = PythonInterpreter()
        try:
            # Before start, deno_process should be None
            assert interp.deno_process is None
            # After start, it should be running
            interp.start()
            assert interp.deno_process is not None
            assert interp.deno_process.poll() is None  # Still running
            # Execute should work
            result = interp.execute("print(42)")
            assert "42" in result
        finally:
            interp.shutdown()

    def test_start_is_idempotent(self):
        """Test that start() can be called multiple times safely."""
        interp = PythonInterpreter()
        try:
            interp.start()
            first_process = interp.deno_process
            interp.start()  # Second call - should be idempotent
            assert interp.deno_process is first_process  # Same process
        finally:
            interp.shutdown()

    def test_basic_execution(self):
        """Test basic code execution."""
        with PythonInterpreter() as interp:
            result = interp.execute("print(1 + 1)")
            assert "2" in result

    def test_variable_injection(self):
        """Test variable injection."""
        with PythonInterpreter(tools={}) as interp:
            result = interp.execute("print(x + y)", variables={"x": 10, "y": 5})
            assert "15" in result

    def test_variable_injection_with_none_values(self):
        """Test variable injection with None values in dicts/lists (JSON null -> Python None)."""
        with PythonInterpreter(tools={}) as interp:
            # Test None in dict
            result = interp.execute("print(data['key'] is None)", variables={"data": {"key": None, "other": "value"}})
            assert "True" in result

            # Test None in list
            result = interp.execute("print(items[1] is None)", variables={"items": [1, None, 3]})
            assert "True" in result

            # Test nested None
            result = interp.execute(
                "print(nested['inner']['value'] is None)", variables={"nested": {"inner": {"value": None}}}
            )
            assert "True" in result

    def test_tool_call_kwargs(self):
        """Test tool call with keyword arguments."""

        def echo(message: str = "") -> str:
            return f"Echo: {message}"

        with PythonInterpreter(tools={"echo": echo}) as interp:
            result = interp.execute('print(echo(message="hello"))')
            assert "Echo: hello" in result

    def test_tool_call_positional(self):
        """Test tool call with positional arguments."""

        def greet(name: str) -> str:
            return f"Hello: {name}"

        with PythonInterpreter(tools={"greet": greet}) as interp:
            result = interp.execute('print(greet("world"))')
            assert "Hello: world" in result

    def test_multiple_tools(self):
        """Test multiple tools."""

        def add(a: int = 0, b: int = 0) -> str:
            return str(a + b)

        def multiply(a: int = 0, b: int = 0) -> str:
            return str(a * b)

        with PythonInterpreter(tools={"add": add, "multiply": multiply}) as interp:
            result = interp.execute("""
sum_result = add(a=3, b=4)
prod_result = multiply(a=3, b=4)
print(f"Sum: {sum_result}, Product: {prod_result}")
""")
            assert "Sum: 7" in result
            assert "Product: 12" in result

    def test_tool_returns_list(self):
        """Test tool that returns a list (like llm_query_batched)."""

        def batch_process(items: list | None = None) -> list:
            items = items or []
            return [f"processed_{item}" for item in items]

        with PythonInterpreter(tools={"batch_process": batch_process}) as interp:  # ty:ignore[invalid-argument-type]
            result = interp.execute("""
results = batch_process(items=["a", "b", "c"])
print(f"Type: {type(results).__name__}")
print(f"Length: {len(results)}")
print(f"First: {results[0]}")
print(f"All: {results}")
""")
            assert "Type: list" in result
            assert "Length: 3" in result
            assert "First: processed_a" in result

    def test_tool_returns_dict(self):
        """Test tool that returns a dict."""

        def get_info() -> dict:
            return {"name": "test", "count": 42}

        with PythonInterpreter(tools={"get_info": get_info}) as interp:  # ty:ignore[invalid-argument-type]
            result = interp.execute("""
info = get_info()
print(f"Type: {type(info).__name__}")
print(f"Name: {info['name']}")
print(f"Count: {info['count']}")
""")
            assert "Type: dict" in result
            assert "Name: test" in result
            assert "Count: 42" in result

    def test_state_persists(self):
        """Test that state persists across executions."""
        with PythonInterpreter(tools={}) as interp:
            interp.execute("x = 10")
            result = interp.execute("print(x + 5)")
            assert "15" in result

    def test_syntax_error(self):
        """Test syntax error handling."""
        with PythonInterpreter(tools={}) as interp, pytest.raises(SyntaxError):
            interp.execute("def incomplete(")

    def test_runtime_error(self):
        """Test runtime error handling."""
        with PythonInterpreter(tools={}) as interp, pytest.raises(CodeInterpreterError):
            interp.execute("undefined_variable")


@pytest.mark.deno
class TestSandboxSecurity:
    """Integration tests for sandbox security restrictions."""

    def test_no_network_access(self):
        """Test that network access is blocked."""
        with PythonInterpreter(tools={}) as interp:
            with pytest.raises(CodeInterpreterError) as exc_info:
                interp.execute("""
from pyodide.http import pyfetch
import asyncio
asyncio.get_event_loop().run_until_complete(pyfetch("https://example.com"))
""")
            assert "net access" in str(exc_info.value).lower() or "allow-net" in str(exc_info.value).lower()

    def test_imports_work(self):
        """Test that standard library imports work."""
        with PythonInterpreter(tools={}) as interp:
            result = interp.execute("""
import json
import re
from collections import Counter
data = {"key": "value"}
print(json.dumps(data))
""")
            assert "key" in result


# ============================================================================
# Unit Tests: RLM with MockInterpreter (no Deno required)
# ============================================================================
