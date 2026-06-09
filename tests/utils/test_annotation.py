from typing import Any, cast

from dspy._meta.experimental import experimental, is_experimental


def test_experimental_decorator_on_function():

    @experimental
    def test_function():
        return "test"

    assert is_experimental(test_function)
    assert cast("Any", test_function).__dspy_experimental__ is True
    assert cast("Any", test_function).__dspy_experimental_version__ is None
    assert test_function() == "test"


def test_experimental_decorator_on_function_with_version():
    experimental_any = cast("Any", experimental)

    @experimental_any(version="3.1.0")
    def test_function():
        return "versioned"

    assert test_function.__dspy_experimental_version__ == "3.1.0"
    assert is_experimental(test_function)
    assert test_function() == "versioned"


def test_experimental_decorator_on_class():

    @experimental
    class TestClass:
        def method(self):
            return "method"

    assert cast("Any", TestClass).__dspy_experimental__ is True
    assert cast("Any", TestClass).__dspy_experimental_version__ is None
    instance = TestClass()
    assert instance.method() == "method"


def test_experimental_decorator_on_class_with_version():
    experimental_any = cast("Any", experimental)

    @experimental_any(version="2.5.0")
    class TestClass:
        pass

    assert TestClass.__dspy_experimental__ is True
    assert TestClass.__dspy_experimental_version__ == "2.5.0"


def test_experimental_decorator_without_docstring():

    @experimental
    def test_function():
        return "no_doc"

    assert cast("Any", test_function).__dspy_experimental__ is True
    assert cast("Any", test_function).__dspy_experimental_version__ is None
    assert test_function() == "no_doc"


def test_experimental_decorator_without_docstring_with_version():
    experimental_any = cast("Any", experimental)

    @experimental_any(version="1.0.0")
    def test_function():
        return "no_doc_version"

    assert test_function.__dspy_experimental__ is True
    assert test_function.__dspy_experimental_version__ == "1.0.0"
    assert test_function() == "no_doc_version"


def test_experimental_decorator_with_callable_syntax():

    def test_function():
        return "callable"

    decorated = cast("Any", experimental)(test_function)
    assert decorated.__dspy_experimental__ is True
    assert decorated() == "callable"


def test_experimental_decorator_with_version_callable_syntax():

    def test_function():
        return "callable_version"

    decorated = cast("Any", experimental)(test_function, version="4.0.0")
    assert decorated.__dspy_experimental_version__ == "4.0.0"
    assert is_experimental(decorated)
    assert decorated() == "callable_version"
