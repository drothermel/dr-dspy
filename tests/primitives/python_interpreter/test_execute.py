import random

import pytest

from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput
from dspy.primitives.python_interpreter import PythonInterpreter

pytestmark = pytest.mark.deno


def test_execute_simple_code():
    with PythonInterpreter() as interpreter:
        code = "print('Hello, World!')"
        result = interpreter.execute(code)
        assert result == "Hello, World!\n", "Simple print statement should return 'Hello World!\n'"


def test_import():
    with PythonInterpreter() as interpreter:
        code = "import math\nresult = math.sqrt(4)\nresult"
        result = interpreter.execute(code)
        assert result == 2, "Should be able to import and use math.sqrt"


def test_user_variable_definitions():
    with PythonInterpreter() as interpreter:
        code = "result = number + 1\nresult"
        result = interpreter.execute(code, variables={"number": 4})
        assert result == 5, "User variable assignment should work"


def test_rejects_python_keywords_as_variable_names():
    with PythonInterpreter() as interpreter:
        keywords_to_test = ["for", "class", "import", "def", "return", "if", "while"]
        for keyword in keywords_to_test:
            with pytest.raises(CodeInterpreterError, match="Invalid variable name"):
                interpreter.execute("print(x)", variables={keyword: 42})


def test_failure_syntax_error():
    with PythonInterpreter() as interpreter:
        code = "+++"
        with pytest.raises(SyntaxError, match="Invalid Python syntax"):
            interpreter.execute(code)


def test_failure_zero_division():
    with PythonInterpreter() as interpreter:
        code = "1+0/0"
        with pytest.raises(CodeInterpreterError, match="ZeroDivisionError"):
            interpreter.execute(code)


def test_exception_args():
    with PythonInterpreter() as interpreter:
        token = random.randint(1, 10**9)
        code = f"raise ValueError({token})"
        with pytest.raises(CodeInterpreterError, match=f"ValueError: \\[{token}\\]"):
            interpreter.execute(code)


def test_submit_with_list():
    with PythonInterpreter() as interpreter:
        token = random.randint(1, 10**9)
        code = f"SUBMIT(['The result is', {token}])"
        result = interpreter(code)
        assert isinstance(result, FinalOutput)
        assert result.output == {"output": ["The result is", token]}


def test_submit_with_typed_signature():
    output_fields = [{"name": "answer", "type": "str"}, {"name": "confidence", "type": "float"}]
    with PythonInterpreter(output_fields=output_fields) as sandbox:
        result = sandbox.execute('SUBMIT(answer="the answer", confidence=0.95)')
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "the answer", "confidence": 0.95}


def test_submit_positional_args():
    output_fields = [{"name": "answer", "type": "str"}, {"name": "confidence", "type": "float"}]
    with PythonInterpreter(output_fields=output_fields) as sandbox:
        result = sandbox.execute('SUBMIT("the answer", 0.95)')
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "the answer", "confidence": 0.95}


def test_submit_multi_output():
    output_fields = [{"name": "answer", "type": "str"}, {"name": "score", "type": "int"}]
    with PythonInterpreter(output_fields=output_fields) as sandbox:
        code = '\na = "my answer"\ns = 42\nSUBMIT(a, s)\n'
        result = sandbox.execute(code)
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "my answer", "score": 42}


def test_submit_wrong_arg_count():
    output_fields = [{"name": "answer", "type": "str"}, {"name": "score", "type": "int"}]
    with PythonInterpreter(output_fields=output_fields) as sandbox:
        with pytest.raises(CodeInterpreterError) as exc_info:
            sandbox.execute("x = 1; SUBMIT(x)")
        assert "missing 1 required positional argument" in str(exc_info.value)


def test_extract_parameters():

    def example_fn(required: str, optional: int = 5, untyped=None) -> str:
        pass

    sandbox = PythonInterpreter()
    params = sandbox._extract_parameters(example_fn)
    assert len(params) == 3
    assert params[0] == {"name": "required", "type": "str"}
    assert params[1] == {"name": "optional", "type": "int", "default": 5}
    assert params[2] == {"name": "untyped", "default": None}


def test_extract_parameters_complex_types():

    def complex_fn(items: list | None = None, data: dict[str, int] | None = None) -> list:
        pass

    sandbox = PythonInterpreter()
    params = sandbox._extract_parameters(complex_fn)
    assert len(params) == 2
    assert params[0] == {"name": "items", "default": None}
    assert params[1] == {"name": "data", "default": None}
