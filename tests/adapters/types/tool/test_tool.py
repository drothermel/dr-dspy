import asyncio
from typing import Any

import pytest

from dspy.adapters.types.tool import Tool
from tests.adapters.types.tool.conftest import (
    Address,
    ContactInfo,
    DummyModel,
    Note,
    UserProfile,
    async_complex_dummy_function,
    async_dummy_function,
    async_dummy_with_pydantic,
    complex_dummy_function,
    dummy_function,
    dummy_with_pydantic,
    requires_jsonschema,
)


def _tool_args(tool: Tool) -> dict[str, Any]:
    assert tool.args is not None
    return tool.args


def test_basic_initialization():
    tool = Tool(lambda x: x, description="A test tool", name="test_tool", args={"param1": {"type": "string"}})
    assert tool.name == "test_tool"
    assert tool.desc == "A test tool"
    assert _tool_args(tool) == {"param1": {"type": "string"}}
    assert callable(tool.func)


def test_to_lm_tool_spec_excludes_defaulted_params_from_required():
    tool = Tool(dummy_function, description="A dummy function for testing.")
    spec = tool.to_lm_tool_spec()
    assert spec.parameters["required"] == ["x"]


@requires_jsonschema
def test_validate_and_parse_args_raises_on_missing_required():
    def required_only(x: int, y: int) -> int:
        return x + y

    tool = Tool(required_only, description="Add two required integers.")
    with pytest.raises(ValueError, match="Missing required arg\\(s\\): y"):
        tool(x=1)


@requires_jsonschema
def test_validate_and_parse_args_raises_when_sole_required_missing():
    tool = Tool(dummy_function, description="A dummy function for testing.")
    with pytest.raises(ValueError, match="Missing required arg\\(s\\): x"):
        tool(y="hello")


def test_tool_from_function():
    tool = Tool(dummy_function, description="A dummy function for testing.")
    assert tool.name == "dummy_function"
    assert tool.desc is not None
    assert "A dummy function for testing" in tool.desc
    assert "x" in _tool_args(tool)
    assert "y" in _tool_args(tool)
    assert _tool_args(tool)["x"]["type"] == "integer"
    assert _tool_args(tool)["y"]["type"] == "string"
    assert _tool_args(tool)["y"]["default"] == "hello"


def test_tool_from_class():

    class Foo:
        def __init__(self, user_id: str):
            self.user_id = user_id

        def __call__(self, a: int, b: int) -> int:
            return a + b

    tool = Tool(Foo("123"), description="Add two numbers.")
    assert tool.name == "Foo"
    assert tool.desc == "Add two numbers."
    assert _tool_args(tool) == {"a": {"type": "integer"}, "b": {"type": "integer"}}


def test_tool_from_function_with_pydantic():
    tool = Tool(dummy_with_pydantic, description="A dummy function that accepts a Pydantic model.")
    assert tool.name == "dummy_with_pydantic"
    assert "model" in _tool_args(tool)
    assert _tool_args(tool)["model"]["type"] == "object"
    assert "field1" in _tool_args(tool)["model"]["properties"]
    assert "field2" in _tool_args(tool)["model"]["properties"]
    assert _tool_args(tool)["model"]["properties"]["field1"]["default"] == "hello"


def test_tool_from_function_with_pydantic_nesting():
    tool = Tool(complex_dummy_function, description="Process user profile with complex nested structure.")
    assert tool.name == "complex_dummy_function"
    assert "profile" in _tool_args(tool)
    assert "priority" in _tool_args(tool)
    assert "notes" in _tool_args(tool)
    assert _tool_args(tool)["profile"]["type"] == "object"
    assert _tool_args(tool)["profile"]["properties"]["user_id"]["type"] == "integer"
    assert _tool_args(tool)["profile"]["properties"]["name"]["type"] == "string"
    assert _tool_args(tool)["profile"]["properties"]["age"]["anyOf"] == [{"type": "integer"}, {"type": "null"}]
    assert _tool_args(tool)["profile"]["properties"]["contact"]["type"] == "object"
    assert _tool_args(tool)["profile"]["properties"]["contact"]["properties"]["email"]["type"] == "string"
    assert "$defs" not in str(_tool_args(tool)["notes"])
    assert _tool_args(tool)["notes"]["anyOf"][0]["type"] == "array"
    assert _tool_args(tool)["notes"]["anyOf"][0]["items"]["type"] == "object"
    assert _tool_args(tool)["notes"]["anyOf"][0]["items"]["properties"]["content"]["type"] == "string"
    assert _tool_args(tool)["notes"]["anyOf"][0]["items"]["properties"]["author"]["type"] == "string"


@requires_jsonschema
def test_tool_callable():
    tool = Tool(dummy_function, description="A dummy function for testing.")
    result = tool(x=42, y="hello")
    assert result == "hello 42"


@requires_jsonschema
def test_tool_with_pydantic_callable():
    tool = Tool(dummy_with_pydantic, description="A dummy function that accepts a Pydantic model.")
    model = DummyModel(field1="test", field2=123)
    result = tool(model=model)
    assert result == "test 123"


@requires_jsonschema
def test_invalid_function_call():
    tool = Tool(dummy_function, description="A dummy function for testing.")
    with pytest.raises(ValueError, match=r"Arg x is invalid"):
        tool(x="not an integer", y="hello")


def test_parameter_desc():
    tool = Tool(dummy_function, description="A dummy function for testing.", arg_desc={"x": "The x parameter"})
    assert _tool_args(tool)["x"]["description"] == "The x parameter"


def test_tool_with_default_args_without_type_hints():

    def foo(x=100):
        return x

    tool = Tool(foo, description="Return x.")
    assert _tool_args(tool)["x"]["default"] == 100
    assert "type" not in _tool_args(tool)["x"]


@requires_jsonschema
def test_tool_call_parses_args():
    tool = Tool(dummy_with_pydantic, description="A dummy function that accepts a Pydantic model.")
    args = {"model": {"field1": "hello", "field2": 123}}
    result = tool(**args)
    assert result == "hello 123"


@requires_jsonschema
def test_tool_call_parses_nested_list_of_pydantic_model():

    def dummy_function(x: list[list[DummyModel]]):
        return x

    tool = Tool(dummy_function, description="A dummy function for testing.")
    args = {"x": [[{"field1": "hello", "field2": 123}]]}
    result = tool(**args)
    assert result == [[DummyModel(field1="hello", field2=123)]]


@requires_jsonschema
def test_tool_call_kwarg():

    def fn(x: int, **kwargs: object):
        return kwargs

    tool = Tool(fn, description="Accept kwargs.")
    assert tool(x=1, y=2, z=3) == {"y": 2, "z": 3}


def test_tool_str():

    def add(x: int, y: int = 0) -> int:
        return x + y

    tool = Tool(add, description="Add two integers.")
    assert (
        str(tool)
        == "add, whose description is <desc>Add two integers.</desc>. It takes arguments {'x': {'type': 'integer'}, 'y': {'type': 'integer', 'default': 0}}."
    )


@requires_jsonschema
@pytest.mark.asyncio
async def test_async_tool_from_function():
    tool = Tool(async_dummy_function, description="An async dummy function for testing.")
    assert tool.name == "async_dummy_function"
    assert tool.desc is not None
    assert "An async dummy function for testing" in tool.desc
    assert "x" in _tool_args(tool)
    assert "y" in _tool_args(tool)
    assert _tool_args(tool)["x"]["type"] == "integer"
    assert _tool_args(tool)["y"]["type"] == "string"
    assert _tool_args(tool)["y"]["default"] == "hello"
    result = await tool.acall(x=42, y="hello")
    assert result == "hello 42"


@requires_jsonschema
@pytest.mark.asyncio
async def test_async_tool_with_pydantic():
    tool = Tool(async_dummy_with_pydantic, description="An async dummy function that accepts a Pydantic model.")
    assert tool.name == "async_dummy_with_pydantic"
    assert "model" in _tool_args(tool)
    assert _tool_args(tool)["model"]["type"] == "object"
    assert "field1" in _tool_args(tool)["model"]["properties"]
    assert "field2" in _tool_args(tool)["model"]["properties"]
    model = DummyModel(field1="test", field2=123)
    result = await tool.acall(model=model)
    assert result == "test 123"
    result = await tool.acall(model={"field1": "test", "field2": 123})
    assert result == "test 123"


@requires_jsonschema
@pytest.mark.asyncio
async def test_async_tool_with_complex_pydantic():
    tool = Tool(
        async_complex_dummy_function, description="Process user profile with complex nested structure asynchronously."
    )
    profile = UserProfile(
        user_id=1,
        name="Test User",
        contact=ContactInfo(
            email="test@example.com",
            addresses=[
                Address(street="123 Main St", city="Test City", zip_code="12345", is_primary=True),
                Address(street="456 Side St", city="Test City", zip_code="12345"),
            ],
        ),
    )
    result = await tool.acall(profile=profile, priority=1, notes=[Note(content="Test note", author="Test author")])
    assert result["user_id"] == 1
    assert result["name"] == "Test User"
    assert result["priority"] == 1
    assert result["notes"] == [Note(content="Test note", author="Test author")]
    assert result["primary_address"]["street"] == "123 Main St"


@requires_jsonschema
@pytest.mark.asyncio
async def test_async_tool_invalid_call():
    tool = Tool(async_dummy_function, description="An async dummy function for testing.")
    with pytest.raises(ValueError, match=r"Arg x is invalid"):
        await tool.acall(x="not an integer", y="hello")


@requires_jsonschema
@pytest.mark.asyncio
async def test_async_tool_with_kwargs():

    async def fn(x: int, **kwargs: object):
        return kwargs

    tool = Tool(fn, description="Accept kwargs.")
    result = await tool.acall(x=1, y=2, z=3)
    assert result == {"y": 2, "z": 3}


@requires_jsonschema
@pytest.mark.asyncio
async def test_async_concurrent_calls():
    tool = Tool(async_dummy_function, description="An async dummy function for testing.")
    tasks = [tool.acall(x=i, y=f"hello{i}") for i in range(5)]
    start_time = asyncio.get_event_loop().time()
    results = await asyncio.gather(*tasks)
    end_time = asyncio.get_event_loop().time()
    assert results == [f"hello{i} {i}" for i in range(5)]
    assert end_time - start_time < 0.3


def test_async_tool_call_in_sync_mode_rejected():
    tool = Tool(async_dummy_function, description="An async dummy function for testing.")
    with pytest.raises(ValueError, match=r".*acall.*"):
        tool(x=1, y="hello")
