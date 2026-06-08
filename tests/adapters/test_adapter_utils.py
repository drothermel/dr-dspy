from typing import Literal

import pytest
from pydantic import BaseModel

from dspy.adapters.utils import parse_value


class Profile(BaseModel):
    name: str
    age: int


def test_parse_value_str_annotation():
    # Test basic string conversion
    assert parse_value(value=123, annotation=str) == "123"
    assert parse_value(value=True, annotation=str) == "True"
    assert parse_value(value="hello", annotation=str) == "hello"
    assert parse_value(value=None, annotation=str) == "None"
    assert parse_value(value=[1, 2, 3], annotation=str) == "[1, 2, 3]"


def test_parse_value_pydantic_types():
    # Test with pydantic BaseModel - JSON string input
    json_str = '{"name": "John", "age": 30}'
    result = parse_value(value=json_str, annotation=Profile)
    assert isinstance(result, Profile)
    assert result.name == "John"
    assert result.age == 30

    # Test with pydantic BaseModel - dict input
    dict_input = {"name": "Jane", "age": 25}
    result = parse_value(value=dict_input, annotation=Profile)
    assert isinstance(result, Profile)
    assert result.name == "Jane"
    assert result.age == 25

    # Test with invalid pydantic data
    with pytest.raises(Exception):  # noqa: B017, PT011
        parse_value(value='{"name": "John"}', annotation=Profile)  # missing required age field


def test_parse_value_basic_types():
    # Test int
    assert parse_value(value="42", annotation=int) == 42
    assert parse_value(value=42, annotation=int) == 42

    # Test float
    assert parse_value(value="3.14", annotation=float) == 3.14
    assert parse_value(value=3.14, annotation=float) == 3.14

    # Test bool
    assert parse_value(value="true", annotation=bool) is True
    assert parse_value(value=True, annotation=bool) is True
    assert parse_value(value="false", annotation=bool) is False

    # Test list
    assert parse_value(value="[1, 2, 3]", annotation=list[int]) == [1, 2, 3]
    assert parse_value(value=[1, 2, 3], annotation=list[int]) == [1, 2, 3]


def test_parse_value_literal():
    # Test Literal type
    assert parse_value(value="option1", annotation=Literal["option1", "option2"]) == "option1"
    assert parse_value(value="option2", annotation=Literal["option1", "option2"]) == "option2"

    # Test Literal with quotes and prefixes
    assert parse_value(value="'option1'", annotation=Literal["option1", "option2"]) == "option1"
    assert parse_value(value='"option1"', annotation=Literal["option1", "option2"]) == "option1"
    assert parse_value(value="Literal[option1]", annotation=Literal["option1", "option2"]) == "option1"
    assert parse_value(value="str[option1]", annotation=Literal["option1", "option2"]) == "option1"

    # Test invalid literal
    with pytest.raises(ValueError):  # noqa: PT011
        parse_value(value="invalid", annotation=Literal["option1", "option2"])


def test_parse_value_union():
    # Test union with None
    assert parse_value(value="test", annotation=str | None) == "test"
    assert parse_value(value="5", annotation=int | None) == 5
    assert parse_value(value=None, annotation=str | None) is None
    assert parse_value(value="text with [placeholder]", annotation=str | None) == "text with [placeholder]"

    # Test union fallback to str
    assert parse_value(value="fallback", annotation=int | str | None) == "fallback"
    assert parse_value(value=5, annotation=int | str | None) == 5
    assert parse_value(value="text with [placeholder]", annotation=int | str | None) == "text with [placeholder]"


def test_parse_value_json_repair():
    # Test cases where json_repair is needed
    assert parse_value(value='{"key": "value"}', annotation=dict) == {"key": "value"}

    # Test ast.literal_eval fallback
    assert parse_value(value="{'key': 'value'}", annotation=dict) == {"key": "value"}

    # Test fallback to original value when parsing fails
    malformed = "not json or literal"
    with pytest.raises(Exception):  # noqa: B017, PT011
        parse_value(value=malformed, annotation=dict)
