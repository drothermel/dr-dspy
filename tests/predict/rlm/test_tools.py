from typing import Any, cast

import pytest

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from dspy.primitives.python_interpreter import PythonInterpreter
from tests.mock_interpreter import MockInterpreter


class TestMockInterpreter:
    def test_scripted_responses(self):
        mock = MockInterpreter(responses=["first", "second", "third"])
        assert mock.execute("code1") == "first"
        assert mock.execute("code2") == "second"
        assert mock.execute("code3") == "third"

    def test_returns_final_output_result(self):
        mock = MockInterpreter(responses=["exploring", FinalOutput("42")])
        assert mock.execute("print(len(data))") == "exploring"
        result = mock.execute("SUBMIT('42')")
        assert isinstance(result, FinalOutput)
        assert result.output == "42"

    def test_raises_exception_from_responses(self):
        mock = MockInterpreter(responses=["ok", CodeInterpreterError("undefined variable")])
        assert mock.execute("code1") == "ok"
        with pytest.raises(CodeInterpreterError, match="undefined variable"):
            mock.execute("code2")

    def test_records_call_history(self):
        mock = MockInterpreter(responses=["resp"])
        mock.execute("print(1)", variables={"x": 10})
        assert mock.call_history == [("print(1)", {"x": 10})]


@pytest.mark.deno
class TestPythonInterpreter:
    def test_start_prewarms_sandbox(self):
        interp = PythonInterpreter()
        try:
            assert interp.deno_process is None
            interp.start()
            assert interp.deno_process is not None
            assert interp.deno_process.poll() is None
            result = interp.execute("print(42)")
            assert "42" in result
        finally:
            interp.shutdown()

    def test_start_is_idempotent(self):
        interp = PythonInterpreter()
        try:
            interp.start()
            first_process = interp.deno_process
            interp.start()
            assert interp.deno_process is first_process
        finally:
            interp.shutdown()

    def test_basic_execution(self):
        with PythonInterpreter() as interp:
            result = interp.execute("print(1 + 1)")
            assert "2" in result

    def test_variable_injection(self):
        with PythonInterpreter(tools={}) as interp:
            result = interp.execute("print(x + y)", variables={"x": 10, "y": 5})
            assert "15" in result

    def test_variable_injection_with_none_values(self):
        with PythonInterpreter(tools={}) as interp:
            result = interp.execute("print(data['key'] is None)", variables={"data": {"key": None, "other": "value"}})
            assert "True" in result
            result = interp.execute("print(items[1] is None)", variables={"items": [1, None, 3]})
            assert "True" in result
            result = interp.execute(
                "print(nested['inner']['value'] is None)", variables={"nested": {"inner": {"value": None}}}
            )
            assert "True" in result

    def test_tool_call_kwargs(self):

        def echo(message: str = "") -> str:
            return f"Echo: {message}"

        with PythonInterpreter(tools={"echo": echo}) as interp:
            result = interp.execute('print(echo(message="hello"))')
            assert "Echo: hello" in result

    def test_tool_call_positional(self):

        def greet(name: str) -> str:
            return f"Hello: {name}"

        with PythonInterpreter(tools={"greet": greet}) as interp:
            result = interp.execute('print(greet("world"))')
            assert "Hello: world" in result

    def test_multiple_tools(self):

        def add(a: int = 0, b: int = 0) -> str:
            return str(a + b)

        def multiply(a: int = 0, b: int = 0) -> str:
            return str(a * b)

        with PythonInterpreter(tools={"add": add, "multiply": multiply}) as interp:
            result = interp.execute(
                '\nsum_result = add(a=3, b=4)\nprod_result = multiply(a=3, b=4)\nprint(f"Sum: {sum_result}, Product: {prod_result}")\n'
            )
            assert "Sum: 7" in result
            assert "Product: 12" in result

    def test_tool_returns_list(self):

        def batch_process(items: list | None = None) -> list:
            items = items or []
            return [f"processed_{item}" for item in items]

        with PythonInterpreter(tools=cast("Any", {"batch_process": batch_process})) as interp:
            result = interp.execute(
                '\nresults = batch_process(items=["a", "b", "c"])\nprint(f"Type: {type(results).__name__}")\nprint(f"Length: {len(results)}")\nprint(f"First: {results[0]}")\nprint(f"All: {results}")\n'
            )
            assert "Type: list" in result
            assert "Length: 3" in result
            assert "First: processed_a" in result

    def test_tool_returns_dict(self):

        def get_info() -> dict:
            return {"name": "test", "count": 42}

        with PythonInterpreter(tools=cast("Any", {"get_info": get_info})) as interp:
            result = interp.execute(
                '\ninfo = get_info()\nprint(f"Type: {type(info).__name__}")\nprint(f"Name: {info[\'name\']}")\nprint(f"Count: {info[\'count\']}")\n'
            )
            assert "Type: dict" in result
            assert "Name: test" in result
            assert "Count: 42" in result

    def test_state_persists(self):
        with PythonInterpreter(tools={}) as interp:
            interp.execute("x = 10")
            result = interp.execute("print(x + 5)")
            assert "15" in result

    def test_syntax_error(self):
        with PythonInterpreter(tools={}) as interp, pytest.raises(SyntaxError):
            interp.execute("def incomplete(")

    def test_runtime_error(self):
        with PythonInterpreter(tools={}) as interp, pytest.raises(CodeInterpreterError):
            interp.execute("undefined_variable")


@pytest.mark.deno
class TestSandboxSecurity:
    def test_no_network_access(self):
        with PythonInterpreter(tools={}) as interp:
            with pytest.raises(CodeInterpreterError) as exc_info:
                interp.execute(
                    '\nfrom pyodide.http import pyfetch\nimport asyncio\nasyncio.get_event_loop().run_until_complete(pyfetch("https://example.com"))\n'
                )
            assert "net access" in str(exc_info.value).lower() or "allow-net" in str(exc_info.value).lower()

    def test_imports_work(self):
        with PythonInterpreter(tools={}) as interp:
            result = interp.execute(
                '\nimport json\nimport re\nfrom collections import Counter\ndata = {"key": "value"}\nprint(json.dumps(data))\n'
            )
            assert "key" in result
