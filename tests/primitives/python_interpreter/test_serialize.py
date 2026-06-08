import pytest

from dspy.primitives.python_interpreter import PythonInterpreter

pytestmark = pytest.mark.deno


def test_serialize_tuple():
    with PythonInterpreter() as interpreter:
        result = interpreter.execute("x", variables={"x": (1, 2, 3)})
        assert result == [1, 2, 3]


def test_serialize_set():
    with PythonInterpreter() as interpreter:
        result = interpreter.execute("sorted(x)", variables={"x": {3, 1, 2}})
        assert result == [1, 2, 3]


def test_serialize_set_mixed_types():
    with PythonInterpreter() as interpreter:
        result = interpreter.execute("x", variables={"x": {1, "a"}})
        assert isinstance(result, list)
        assert set(result) == {1, "a"}


def test_large_variable_injection():
    from dspy.primitives.python_interpreter import LARGE_VAR_THRESHOLD

    large_data = "x" * (LARGE_VAR_THRESHOLD + 1024)
    with PythonInterpreter() as interpreter:
        result = interpreter.execute("len(data)", variables={"data": large_data})
        assert result == len(large_data), "Large variable should be correctly injected and accessible"


def test_large_variable_content_integrity():
    from dspy.primitives.python_interpreter import LARGE_VAR_THRESHOLD

    pattern = "ABCDEFGHIJ" * 100
    large_data = pattern * (LARGE_VAR_THRESHOLD // len(pattern) + 1)
    with PythonInterpreter() as interpreter:
        code = "\nfirst_100 = data[:100]\nlast_100 = data[-100:]\n(first_100, last_100)\n"
        result = interpreter.execute(code, variables={"data": large_data})
        assert result[0] == large_data[:100], "First 100 chars should match"
        assert result[1] == large_data[-100:], "Last 100 chars should match"


def test_mixed_small_and_large_variables():
    from dspy.primitives.python_interpreter import LARGE_VAR_THRESHOLD

    small_var = "hello"
    large_var = "x" * (LARGE_VAR_THRESHOLD + 1024)
    with PythonInterpreter() as interpreter:
        code = "f'{small} has {len(small)} chars, large has {len(large)} chars'"
        result = interpreter.execute(code, variables={"small": small_var, "large": large_var})
        expected = f"{small_var} has {len(small_var)} chars, large has {len(large_var)} chars"
        assert result == expected, "Both small and large variables should work together"


def test_multiple_large_variables():
    from dspy.primitives.python_interpreter import LARGE_VAR_THRESHOLD

    large_a = "a" * (LARGE_VAR_THRESHOLD + 100)
    large_b = "b" * (LARGE_VAR_THRESHOLD + 200)
    with PythonInterpreter() as interpreter:
        code = "(len(var_a), len(var_b), var_a[0], var_b[0])"
        result = interpreter.execute(code, variables={"var_a": large_a, "var_b": large_b})
        assert result == [len(large_a), len(large_b), "a", "b"], "Multiple large variables should work"


def test_large_list_variable():
    from dspy.primitives.python_interpreter import LARGE_VAR_THRESHOLD

    num_elements = LARGE_VAR_THRESHOLD // 3
    large_list = ["x"] * num_elements
    with PythonInterpreter() as interpreter:
        code = "(len(data), data[0], data[-1], type(data).__name__)"
        result = interpreter.execute(code, variables={"data": large_list})
        assert result == [num_elements, "x", "x", "list"]


def test_nested_sets_and_tuples():
    complex_data = {"tags": {1, 2, 3}, "coords": (10, 20), "nested": [{"inner_set": {"a", "b"}}]}
    with PythonInterpreter() as interpreter:
        result = interpreter.execute("data", variables={"data": complex_data})
        assert result["tags"] == [1, 2, 3]
        assert result["coords"] == [10, 20]
        assert result["nested"][0]["inner_set"] == ["a", "b"]


def test_small_variable_not_using_filesystem():
    small_var = "small string"
    interpreter = PythonInterpreter()
    interpreter._pending_large_vars = {}
    interpreter._inject_variables("print(x)", {"x": small_var})
    assert interpreter._pending_large_vars == {}, "Small variables should not be in _pending_large_vars"


def test_large_variable_threshold_boundary():
    from dspy.primitives.python_interpreter import LARGE_VAR_THRESHOLD

    at_threshold = "x" * (LARGE_VAR_THRESHOLD - 2)
    interpreter = PythonInterpreter()
    interpreter._pending_large_vars = {}
    interpreter._inject_variables("print(x)", {"x": at_threshold})
    assert interpreter._pending_large_vars == {}, "Serialized size at threshold should be embedded"
    over_threshold = "x" * (LARGE_VAR_THRESHOLD - 1)
    interpreter._pending_large_vars = {}
    interpreter._inject_variables("print(x)", {"x": over_threshold})
    assert "x" in interpreter._pending_large_vars, "Serialized size over threshold should use filesystem"
