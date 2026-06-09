import json
from typing import cast

import pytest

from dspy.predict.code_execution import execute_generated_code, parse_generated_code, strip_python_fences
from dspy.primitives import FinalOutput, Prediction
from dspy.primitives.python_interpreter import PythonInterpreter
from tests.mock_interpreter import MockInterpreter


def test_parse_generated_code_happy_path_fenced_block():
    code_data = Prediction(generated_code="```python\nx = 1 + 1\nprint(x)\n```")
    code, error = parse_generated_code(code_data)
    assert error is None
    assert "x = 1 + 1" in code
    assert "print(x)" in code


def test_parse_generated_code_empty_code():
    _code, error = parse_generated_code({"generated_code": ""})
    assert error == "Error: Empty code after parsing."


def test_parse_generated_code_malformed_single_line_multi_assign():
    _code, error = parse_generated_code({"generated_code": "a=1 b=2"})
    assert error == "Error: Code format is not correct."


def test_parse_generated_code_appends_last_line_variable_echo():
    code, error = parse_generated_code({"generated_code": "```python\nx = 1\ny = x + 1\n```"})
    assert error is None
    assert code.endswith("\ny")


def test_execute_generated_code_final_output_path():
    interpreter = cast("PythonInterpreter", MockInterpreter(responses=[FinalOutput({"answer": 42})]))
    output, error = execute_generated_code(code="SUBMIT({'answer': 42})", interpreter=interpreter)
    assert error is None
    assert output == json.dumps({"answer": 42})
    interpreter.shutdown()


def test_execute_generated_code_empty_code():
    interpreter = PythonInterpreter()
    output, error = execute_generated_code(code="", interpreter=interpreter)
    assert output is None
    assert error == "Error: Empty code before execution."
    interpreter.shutdown()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("print(1)", "print(1)"),
        ("```python\nprint(1)\n```", "print(1)"),
        ("```\nprint(1)\n```", "print(1)"),
    ],
)
def test_strip_python_fences(raw, expected):
    assert strip_python_fences(raw) == expected


def test_strip_python_fences_rejects_non_python_lang():
    with pytest.raises(SyntaxError, match="Expected Python code"):
        strip_python_fences('```json\n{"a": 1}\n```')
