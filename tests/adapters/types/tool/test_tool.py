import asyncio

import pytest

from dspy.adapters.types.tool import Tool
from dspy.dsp.utils.settings import settings
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


def test_basic_initialization():
    tool = Tool(lambda x: x, description="A test tool", name="test_tool", args={"param1": {"type": "string"}})
    assert tool.name == "test_tool"
    assert tool.desc == "A test tool"
    assert tool.args == {"param1": {"type": "string"}}
    assert callable(tool.func)


def test_tool_from_function():
    tool = Tool(dummy_function, description="A dummy function for testing.")
    assert tool.name == "dummy_function"
    assert "A dummy function for testing" in tool.desc
    assert "x" in tool.args
    assert "y" in tool.args
    assert tool.args["x"]["type"] == "integer"
    assert tool.args["y"]["type"] == "string"
    assert tool.args["y"]["default"] == "hello"


def test_tool_from_class():

    class Foo:
        def __init__(self, user_id: str):
            self.user_id = user_id

        def __call__(self, a: int, b: int) -> int:
            return a + b

    tool = Tool(Foo("123"), description="Add two numbers.")
    assert tool.name == "Foo"
    assert tool.desc == "Add two numbers."
    assert tool.args == {"a": {"type": "integer"}, "b": {"type": "integer"}}


def test_tool_from_function_with_pydantic():
    tool = Tool(dummy_with_pydantic, description="A dummy function that accepts a Pydantic model.")
    assert tool.name == "dummy_with_pydantic"
    assert "model" in tool.args
    assert tool.args["model"]["type"] == "object"
    assert "field1" in tool.args["model"]["properties"]
    assert "field2" in tool.args["model"]["properties"]
    assert tool.args["model"]["properties"]["field1"]["default"] == "hello"


def test_tool_from_function_with_pydantic_nesting():
    tool = Tool(complex_dummy_function, description="Process user profile with complex nested structure.")
    assert tool.name == "complex_dummy_function"
    assert "profile" in tool.args
    assert "priority" in tool.args
    assert "notes" in tool.args
    assert tool.args["profile"]["type"] == "object"
    assert tool.args["profile"]["properties"]["user_id"]["type"] == "integer"
    assert tool.args["profile"]["properties"]["name"]["type"] == "string"
    assert tool.args["profile"]["properties"]["age"]["anyOf"] == [{"type": "integer"}, {"type": "null"}]
    assert tool.args["profile"]["properties"]["contact"]["type"] == "object"
    assert tool.args["profile"]["properties"]["contact"]["properties"]["email"]["type"] == "string"
    assert "$defs" not in str(tool.args["notes"])
    assert tool.args["notes"]["anyOf"][0]["type"] == "array"
    assert tool.args["notes"]["anyOf"][0]["items"]["type"] == "object"
    assert tool.args["notes"]["anyOf"][0]["items"]["properties"]["content"]["type"] == "string"
    assert tool.args["notes"]["anyOf"][0]["items"]["properties"]["author"]["type"] == "string"


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
    with pytest.raises(ValueError):
        tool(x="not an integer", y="hello")


def test_parameter_desc():
    tool = Tool(dummy_function, description="A dummy function for testing.", arg_desc={"x": "The x parameter"})
    assert tool.args["x"]["description"] == "The x parameter"


def test_tool_with_default_args_without_type_hints():

    def foo(x=100):
        return x

    tool = Tool(foo, description="Return x.")
    assert tool.args["x"]["default"] == 100
    assert not hasattr(tool.args["x"], "type")


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
    assert "An async dummy function for testing" in tool.desc
    assert "x" in tool.args
    assert "y" in tool.args
    assert tool.args["x"]["type"] == "integer"
    assert tool.args["y"]["type"] == "string"
    assert tool.args["y"]["default"] == "hello"
    result = await tool.acall(x=42, y="hello")
    assert result == "hello 42"


@requires_jsonschema
@pytest.mark.asyncio
async def test_async_tool_with_pydantic():
    tool = Tool(async_dummy_with_pydantic, description="An async dummy function that accepts a Pydantic model.")
    assert tool.name == "async_dummy_with_pydantic"
    assert "model" in tool.args
    assert tool.args["model"]["type"] == "object"
    assert "field1" in tool.args["model"]["properties"]
    assert "field2" in tool.args["model"]["properties"]
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
    with pytest.raises(ValueError):
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


@requires_jsonschema
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_async_tool_call_in_sync_mode():
    tool = Tool(async_dummy_function, description="An async dummy function for testing.")
    with settings.context(allow_tool_async_sync_conversion=False):
        with pytest.raises(ValueError, match=".*acall.*allow_tool_async_sync_conversion.*"):
            result = tool(x=1, y="hello")
    with settings.context(allow_tool_async_sync_conversion=True):
        result = tool(x=1, y="hello")
        assert result == "hello 1"
