import pydantic
import pytest

from dspy.adapters.types.code import Code
from dspy.utils.source_format import get_formatted_source


def test_code_validate_input():
    code = Code["python"](code="print('Hello, world!')")
    assert code.code == "print('Hello, world!')"
    with pytest.raises(ValueError):
        Code["python"](code=123)

    def foo(x):
        return x + 1

    code_source = get_formatted_source(foo)
    code = Code["python"](code=code_source)
    assert code.code == code_source


def test_code_in_nested_type():

    class Wrapper(pydantic.BaseModel):
        code: Code

    code = Code(code="print('Hello, world!')")
    wrapper = Wrapper(code=code)
    assert wrapper.code.code == "print('Hello, world!')"


def test_code_with_language():
    java_code = Code["java"](code="System.out.println('Hello, world!');")
    assert java_code.code == "System.out.println('Hello, world!');"
    assert java_code.language == "java"
    assert "Programming language: java" in java_code.description()
    cpp_code = Code["cpp"](code="std::cout << 'Hello, world!' << std::endl;")
    assert cpp_code.code == "std::cout << 'Hello, world!' << std::endl;"
    assert cpp_code.language == "cpp"
    assert "Programming language: cpp" in cpp_code.description()


def test_code_parses_from_dirty_code():
    dirty_code = "```python\nprint('Hello, world!')```"
    code = Code(code=dirty_code)
    assert code.code == "print('Hello, world!')"
    dirty_code_with_reasoning = "\nThe generated code is:\n```python\nprint('Hello, world!')\n```\n\nThe reasoning is:\nThe code is a simple print statement.\n"
    code = Code(code=dirty_code_with_reasoning)
    assert code.code == "print('Hello, world!')"
