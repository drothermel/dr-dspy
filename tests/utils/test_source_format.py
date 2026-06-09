from dspy.propose.source_format import get_formatted_source


def _sample_with_docstring_and_comment(x: int) -> int:
    return x + 1


class _SampleClass:
    def method(self) -> str:
        return "ok"


def test_get_formatted_source_strips_docstrings_and_comments():
    result = get_formatted_source(_sample_with_docstring_and_comment)
    assert '"""' not in result
    assert "# inline comment" not in result
    assert "def _sample_with_docstring_and_comment(x: int) -> int:" in result
    assert "return x + 1" in result


def test_get_formatted_source_strips_class_docstring_and_comments():
    result = get_formatted_source(_SampleClass)
    assert '"""' not in result
    assert "# method comment" not in result
    assert "class _SampleClass:" in result
    assert "def method(self) -> str:" in result


def test_get_formatted_source_include_docstring_after_physical_strip():
    result = get_formatted_source(_sample_with_docstring_and_comment, include_docstring=True, include_comments=True)
    assert "def _sample_with_docstring_and_comment(x: int) -> int:" in result
    assert "return x + 1" in result


def test_get_formatted_source_explicit_docstring_injection():
    result = get_formatted_source(_sample_with_docstring_and_comment, docstring="Explicit description.")
    assert "Explicit description." in result
    assert "Docstring to strip." not in result


def test_get_formatted_source_matches_inspect_for_comment_free_function():

    def plain(a: int) -> int:
        return a

    assert get_formatted_source(plain) == "def plain(a: int) -> int:\n    return a\n"


def test_get_formatted_source_is_idempotent():
    once = get_formatted_source(_sample_with_docstring_and_comment)
    twice = get_formatted_source(_sample_with_docstring_and_comment)
    assert once == twice
