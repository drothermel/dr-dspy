import inspect

import pydantic
import pytest

from dspy.adapters.types.code import Code


def test_code_validate_input():
    # Create a `Code` instance with valid code.
    code = Code["python"](code="print('Hello, world!')")  # ty:ignore[invalid-argument-type, unknown-argument]
    assert code.code == "print('Hello, world!')"  # ty:ignore[unresolved-attribute]

    with pytest.raises(ValueError):  # noqa: PT011
        # Try to create a `Code` instance with invalid type.
        Code["python"](code=123)  # ty: ignore[invalid-argument-type, unknown-argument]

    def foo(x):
        return x + 1

    code_source = inspect.getsource(foo)
    code = Code["python"](code=code_source)  # ty:ignore[invalid-argument-type, unknown-argument]

    assert code.code == code_source  # ty:ignore[unresolved-attribute]


def test_code_in_nested_type():
    class Wrapper(pydantic.BaseModel):
        code: Code

    code = Code(code="print('Hello, world!')")
    wrapper = Wrapper(code=code)
    assert wrapper.code.code == "print('Hello, world!')"


def test_code_with_language():
    java_code = Code["java"](code="System.out.println('Hello, world!');")  # ty:ignore[invalid-argument-type, unknown-argument]
    assert java_code.code == "System.out.println('Hello, world!');"  # ty:ignore[unresolved-attribute]
    assert java_code.language == "java"  # ty:ignore[unresolved-attribute]
    assert "Programming language: java" in java_code.description()  # ty:ignore[unresolved-attribute]

    cpp_code = Code["cpp"](code="std::cout << 'Hello, world!' << std::endl;")  # ty:ignore[invalid-argument-type, unknown-argument]
    assert cpp_code.code == "std::cout << 'Hello, world!' << std::endl;"  # ty:ignore[unresolved-attribute]
    assert cpp_code.language == "cpp"  # ty:ignore[unresolved-attribute]
    assert "Programming language: cpp" in cpp_code.description()  # ty:ignore[unresolved-attribute]


def test_code_parses_from_dirty_code():
    dirty_code = "```python\nprint('Hello, world!')```"
    code = Code(code=dirty_code)
    assert code.code == "print('Hello, world!')"

    dirty_code_with_reasoning = """
The generated code is:
```python
print('Hello, world!')
```

The reasoning is:
The code is a simple print statement.
"""
    code = Code(code=dirty_code_with_reasoning)
    assert code.code == "print('Hello, world!')"
