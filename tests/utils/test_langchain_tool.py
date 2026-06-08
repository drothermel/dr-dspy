import importlib
import importlib.util
from typing import Any, cast

import pytest

if importlib.util.find_spec("langchain_core") is None:
    pytest.skip("langchain_core is not installed", allow_module_level=True)
from pydantic import BaseModel

from dspy.utils.langchain_tool import convert_langchain_tool


@pytest.mark.asyncio
@pytest.mark.extra
async def test_convert_custom_simple_tool():
    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        return a + b

    converted_tool = convert_langchain_tool(add)
    assert converted_tool.name == "add"
    assert converted_tool.desc == "Add two numbers."
    assert converted_tool.args == {"a": {"title": "A", "type": "integer"}, "b": {"title": "B", "type": "integer"}}
    assert converted_tool.arg_types == {"a": int, "b": int}
    assert converted_tool.arg_desc == {
        "a": "No description provided. (Required)",
        "b": "No description provided. (Required)",
    }
    assert await cast("Any", converted_tool).acall(a=1, b=2) == 3


@pytest.mark.asyncio
@pytest.mark.extra
async def test_convert_custom_tool_with_custom_class():
    from langchain_core.tools import tool

    class Profile(BaseModel):
        name: str
        age: int

    @tool
    def get_age(profile: Profile) -> int:
        return profile.age

    converted_tool = convert_langchain_tool(get_age)
    assert converted_tool.name == "get_age"
    assert converted_tool.desc == "Get the age of the profile."
    assert converted_tool.args == {
        "profile": {
            "title": "Profile",
            "type": "object",
            "properties": {"name": {"title": "Name", "type": "string"}, "age": {"title": "Age", "type": "integer"}},
            "required": ["name", "age"],
        }
    }
    assert converted_tool.arg_types == {"profile": Profile}
    assert converted_tool.arg_desc == {"profile": "No description provided. (Required)"}
    assert await cast("Any", converted_tool).acall(profile=Profile(name="John", age=20)) == 20
