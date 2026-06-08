import asyncio
import inspect
from collections.abc import Coroutine
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast, get_origin, get_type_hints

import json_repair
import pydantic
from pydantic import BaseModel, TypeAdapter, create_model
from typing_extensions import override

from dspy.adapters.types.base_type import Type
from dspy.dsp.utils.settings import settings
from dspy.utils.lazy_import import require

if TYPE_CHECKING:
    import mcp
    from langchain.tools import BaseTool  # ty: ignore[unresolved-import]

_TYPE_MAPPING = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}
jsonschema = require("jsonschema", extra="tools", feature="dspy.adapters.types.tool.Tool argument validation")


class PydanticJsonSchemaHandler(Protocol):
    def resolve_ref_schema(self, schema: dict[str, Any]) -> dict[str, Any]: ...


def _with_callbacks(fn: Callable[..., object]) -> Callable[..., object]:
    if inspect.iscoroutinefunction(fn):

        @wraps(fn)
        async def async_wrapper(*args: object, **kwargs: object) -> object:
            from dspy.utils.callback import with_callbacks

            return await with_callbacks(fn)(*args, **kwargs)

        return async_wrapper

    @wraps(fn)
    def sync_wrapper(*args: object, **kwargs: object) -> object:
        from dspy.utils.callback import with_callbacks

        return with_callbacks(fn)(*args, **kwargs)

    return sync_wrapper


def _validate_json_schema(instance: object, schema: dict[str, Any], arg_name: str) -> None:
    validation_error_cls = jsonschema.ValidationError
    validate = jsonschema.validate
    try:
        validate(instance=instance, schema=schema)
    except validation_error_cls as e:
        raise ValueError(f"Arg {arg_name} is invalid: {e.message}") from e


class Tool(Type):
    """Tool class.

    This class is used to simplify the creation of tools for tool calling (function calling) in LLMs. Only supports
    functions for now.
    """

    func: Callable[..., object]
    name: str | None = None
    desc: str | None = None
    args: dict[str, Any] | None = None
    arg_types: dict[str, Any] | None = None
    arg_desc: dict[str, str] | None = None
    has_kwargs: bool = False

    def __init__(
        self,
        func: Callable[..., object],
        name: str | None = None,
        desc: str | None = None,
        args: dict[str, Any] | None = None,
        arg_types: dict[str, Any] | None = None,
        arg_desc: dict[str, str] | None = None,
    ) -> None:
        """Initialize the Tool class.

        Users can choose to specify the `name`, `desc`, `args`, and `arg_types`, or let the `Tool`
        automatically infer the values from the function. For values that are specified by the user, automatic inference
        will not be performed on them.

        Args:
            func (Callable): The actual function that is being wrapped by the tool.
            name (str | None, optional): The name of the tool. Defaults to None.
            desc (str | None, optional): The description of the tool. Defaults to None.
            args (dict[str, Any] | None, optional): The args and their schema of the tool, represented as a
                dictionary from arg name to arg's json schema. Defaults to None.
            arg_types (dict[str, Any] | None, optional): The argument types of the tool, represented as a dictionary
                from arg name to the type of the argument. Defaults to None.
            arg_desc (dict[str, str] | None, optional): Descriptions for each arg, represented as a
                dictionary from arg name to description string. Defaults to None.

        Examples:

        ```python
        def foo(x: int, y: str = "hello"):
            return str(x) + y

        tool = Tool(foo)
        print(tool.args)
        # Expected output: {'x': {'type': 'integer'}, 'y': {'type': 'string', 'default': 'hello'}}
        ```
        """
        super().__init__(func=func, name=name, desc=desc, args=args, arg_types=arg_types, arg_desc=arg_desc)  # ty: ignore[unknown-argument]
        self._parse_function(func, arg_desc)

    def _parse_function(self, func: Callable, arg_desc: dict[str, str] | None = None) -> None:
        """Helper method that parses a function to extract the name, description, and args.

        This is a helper function that automatically infers the name, description, and args of the tool from the
        provided function. In order to make the inference work, the function must have valid type hints.
        """
        annotations_func = func if inspect.isfunction(func) or inspect.ismethod(func) else func.__call__
        name = getattr(func, "__name__", type(func).__name__)
        desc = getattr(func, "__doc__", None) or getattr(annotations_func, "__doc__", "")
        args = {}
        arg_types = {}

        sig = inspect.signature(annotations_func)
        available_hints = get_type_hints(annotations_func)
        hints = {param_name: available_hints.get(param_name, Any) for param_name in sig.parameters}
        default_values = {param_name: sig.parameters[param_name].default for param_name in sig.parameters}

        for k, v in hints.items():
            arg_types[k] = v
            if k == "return":
                continue
            origin = get_origin(v) or v
            if isinstance(origin, type) and issubclass(origin, BaseModel):
                v_json_schema = _resolve_json_schema_reference(v.model_json_schema())
                args[k] = v_json_schema
            else:
                args[k] = _resolve_json_schema_reference(TypeAdapter(v).json_schema())
            if default_values[k] is not inspect.Parameter.empty:
                args[k]["default"] = default_values[k]
            if arg_desc and k in arg_desc:
                args[k]["description"] = arg_desc[k]

        self.name = self.name or name
        self.desc = self.desc or desc
        self.args = self.args if self.args is not None else args
        self.arg_types = self.arg_types if self.arg_types is not None else arg_types
        self.has_kwargs = any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values())

    def _validate_and_parse_args(self, **kwargs: object) -> dict[str, object]:
        args_schema = self.args or {}
        arg_types = self.arg_types or {}

        for k, v in kwargs.items():
            if k not in args_schema:
                if self.has_kwargs:
                    continue
                raise ValueError(f"Arg {k} is not in the tool's args.")
            instance = v.model_dump() if isinstance(v, BaseModel) else v
            type_str = args_schema[k].get("type")
            if type_str is not None and type_str != "Any":
                _validate_json_schema(instance=instance, schema=args_schema[k], arg_name=k)

        parsed_kwargs: dict[str, object] = {}
        for k, v in kwargs.items():
            if k in arg_types and arg_types[k] != Any:
                # Create a pydantic model wrapper with a dummy field `value` to parse the arg to the correct type.
                # This is specifically useful for handling nested Pydantic models like `list[list[MyPydanticModel]]`
                pydantic_wrapper = create_model("Wrapper", value=(arg_types[k], ...))
                parsed = pydantic_wrapper.model_validate({"value": v})
                parsed_kwargs[k] = parsed.value  # ty: ignore[unresolved-attribute]
            else:
                parsed_kwargs[k] = v
        return parsed_kwargs

    @override
    def format(self) -> str:
        return str(self)

    def format_as_litellm_function_call(self) -> dict[str, object]:
        args_schema = self.args or {}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.desc,
                "parameters": {
                    "type": "object",
                    "properties": args_schema,
                    "required": list(args_schema.keys()),
                },
            },
        }

    def _run_async_in_sync(self, coroutine: Coroutine[object, Any, object]) -> object:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Run the coroutine outside of "except" block to avoid propagation
            loop = None

        if loop is None:
            return asyncio.run(coroutine)
        return loop.run_until_complete(coroutine)

    @_with_callbacks
    def __call__(self, **kwargs: object) -> object:
        parsed_kwargs = self._validate_and_parse_args(**kwargs)
        result = self.func(**parsed_kwargs)
        if asyncio.iscoroutine(result):
            if settings.allow_tool_async_sync_conversion:
                return self._run_async_in_sync(result)
            raise ValueError(
                "You are calling `__call__` on an async tool, please use `acall` instead or enable "
                "async-to-sync conversion with `settings.configure(allow_tool_async_sync_conversion=True)` "
                "or `with settings.context(allow_tool_async_sync_conversion=True):` from "
                "`dspy.dsp.utils.settings`."
            )
        return result

    @_with_callbacks
    async def acall(self, **kwargs: object) -> object:
        parsed_kwargs = self._validate_and_parse_args(**kwargs)
        result = self.func(**parsed_kwargs)
        if asyncio.iscoroutine(result):
            return await result
        # We should allow calling a sync tool in the async path.
        return result

    @classmethod
    def from_mcp_tool(cls, session: "mcp.ClientSession", tool: "mcp.types.Tool") -> "Tool":
        """
        Build a DSPy tool from an MCP tool and a ClientSession.

        Args:
            session: The MCP session to use.
            tool: The MCP tool to convert.

        Returns:
            A Tool object.
        """
        from dspy.utils.mcp import convert_mcp_tool

        return convert_mcp_tool(session, tool)

    @classmethod
    def from_langchain(cls, tool: "BaseTool") -> "Tool":
        """
        Build a DSPy tool from a LangChain tool.

        Args:
            tool: The LangChain tool to convert.

        Returns:
            A Tool object.

        Examples:

        ```python
        import asyncio
        from dspy.adapters.types.tool import Tool
        from langchain.tools import tool as lc_tool

        @lc_tool
        def add(x: int, y: int):
            "Add two numbers together."
            return x + y

        dspy_tool = Tool.from_langchain(add)

        async def run_tool():
            return await dspy_tool.acall(x=1, y=2)

        print(asyncio.run(run_tool()))
        # 3
        ```
        """
        from dspy.utils.langchain_tool import convert_langchain_tool

        return convert_langchain_tool(tool)

    @override
    def __repr__(self) -> str:
        return f"Tool(name={self.name}, desc={self.desc}, args={self.args})"

    @override
    def __str__(self) -> str:
        desc = f", whose description is <desc>{self.desc}</desc>.".replace("\n", "  ") if self.desc else "."
        arg_desc = f"It takes arguments {self.args}."
        return f"{self.name}{desc} {arg_desc}"


class ToolCalls(Type):
    class ToolCall(Type):
        id: str | None = None
        name: str
        args: dict[str, Any]

        @classmethod
        @override
        def __get_pydantic_json_schema__(  # ty: ignore[invalid-method-override]
            cls,
            core_schema: object,
            handler: PydanticJsonSchemaHandler,
        ) -> dict[str, Any]:
            schema = super().__get_pydantic_json_schema__(core_schema, handler)  # ty: ignore[invalid-argument-type]
            schema = handler.resolve_ref_schema(schema)
            properties = schema.get("properties")
            if isinstance(properties, dict):
                properties.pop("id", None)
            required = schema.get("required")
            if isinstance(required, list):
                schema["required"] = [field for field in required if field != "id"]
            return schema

        @override
        def format(self) -> dict[str, Any]:
            return {"name": self.name, "args": self.args}

        def execute(self, functions: dict[str, Callable[..., object]] | list[Tool] | None = None) -> object:
            """Execute this individual tool call and return its result.

            Args:
                functions: Functions to search for the tool. Can be:
                          - Dict mapping tool names to functions: {"tool_name": function}
                          - List of Tool objects: [Tool(function), ...]
                          - None: Will search in caller's locals and globals (automatic lookup)

            Returns:
                The result from executing this tool call.

            Raises:
                ValueError: If the tool function cannot be found.
                Exception: Any exception raised by the tool function.
            """
            func = None

            if functions is None:
                # Automatic lookup in caller's globals and locals
                current_frame = inspect.currentframe()
                frame = current_frame.f_back if current_frame is not None else None
                try:
                    if frame is not None:
                        caller_globals = frame.f_globals
                        caller_locals = frame.f_locals
                        func = caller_locals.get(self.name) or caller_globals.get(self.name)
                finally:
                    del frame

            elif isinstance(functions, dict):
                func = functions.get(self.name)
            elif isinstance(functions, list):
                for tool in functions:
                    if tool.name == self.name:
                        func = tool.func
                        break

            if func is None:
                raise ValueError(
                    f"Tool function '{self.name}' not found. Please pass the tool functions to the `execute` method."
                )

            try:
                args = self.args or {}
                return func(**args)
            except Exception as e:
                raise RuntimeError(f"Error executing tool '{self.name}': {e}") from e

    tool_calls: list[ToolCall]
    tool_call_results: Any | None = None

    @classmethod
    @override
    def __get_pydantic_json_schema__(  # ty: ignore[invalid-method-override]
        cls,
        core_schema: object,
        handler: PydanticJsonSchemaHandler,
    ) -> dict[str, Any]:
        schema = super().__get_pydantic_json_schema__(core_schema, handler)  # ty: ignore[invalid-argument-type]
        schema = handler.resolve_ref_schema(schema)
        properties = schema.get("properties")
        if isinstance(properties, dict):
            properties.pop("tool_call_results", None)
        required = schema.get("required")
        if isinstance(required, list):
            schema["required"] = [field for field in required if field != "tool_call_results"]
        return schema

    @classmethod
    def from_dict_list(cls, tool_calls_dicts: list[dict[str, Any]]) -> "ToolCalls":
        """Convert a list of dictionaries to a ToolCalls instance.

        Args:
            dict_list: A list of dictionaries, where each dictionary should have 'name' and 'args' keys.

        Returns:
            A ToolCalls instance.

        Examples:

            ```python
            tool_calls_dict = [
                {"name": "search", "args": {"query": "hello"}},
                {"name": "translate", "args": {"text": "world"}}
            ]
            tool_calls = ToolCalls.from_dict_list(tool_calls_dict)
            ```
        """
        tool_calls = [cls.ToolCall(**_normalize_tool_call_dict(item)) for item in tool_calls_dicts]
        return cls(tool_calls=tool_calls)

    @classmethod
    @override
    def description(cls) -> str:
        return (
            "Tool calls must be a JSON object with `tool_calls`, a list of calls. "
            "Each call must include `name` and `args`. "
            'Example: {"tool_calls": [{"name": "search", "args": {"query": "cats"}}]}'
        )

    @override
    def format(self) -> dict[str, Any]:
        return {
            "tool_calls": [tool_call.format() for tool_call in self.tool_calls],
        }

    @pydantic.model_serializer()
    @override
    def serialize_model(self) -> dict[str, Any]:
        data = self.format()
        if self.tool_call_results is not None:
            data["tool_call_results"] = TypeAdapter(type(self.tool_call_results)).dump_python(
                self.tool_call_results,
                mode="json",
            )
        return data

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: object) -> object:
        if isinstance(data, cls):
            return data

        if isinstance(data, list) and all(
            isinstance(item, dict) and _is_tool_call_dict(cast("dict[str, Any]", item)) for item in data
        ):
            return {
                "tool_calls": [cls.ToolCall(**_normalize_tool_call_dict(cast("dict[str, Any]", item))) for item in data]
            }
        if isinstance(data, dict):
            data = cast("dict[str, Any]", data)
            if "tool_calls" in data:
                tool_calls_data = data["tool_calls"]
                if isinstance(tool_calls_data, list):
                    normalized = {
                        "tool_calls": [
                            cls.ToolCall(**_normalize_tool_call_dict(cast("dict[str, Any]", item)))
                            if isinstance(item, dict)
                            else item
                            for item in tool_calls_data
                        ]
                    }
                    if "tool_call_results" in data:
                        normalized["tool_call_results"] = data["tool_call_results"]
                    return normalized
            elif _is_tool_call_dict(data):
                return {"tool_calls": [cls.ToolCall(**_normalize_tool_call_dict(data))]}

        raise ValueError(f"Received invalid value for `dspy.adapters.types.tool.ToolCalls`: {data}")


class ToolCallResults(pydantic.BaseModel):
    class ToolCallResult(pydantic.BaseModel):
        call_id: str | None = None
        name: str
        value: Any
        is_error: bool = False

    tool_call_results: list[ToolCallResult]

    @classmethod
    def from_tool_calls_and_values(
        cls,
        tool_calls: list[ToolCalls.ToolCall] | ToolCalls,
        values: list[Any],
        is_errors: list[bool] | None = None,
    ) -> "ToolCallResults":
        if isinstance(tool_calls, ToolCalls):
            tool_calls = tool_calls.tool_calls

        if len(tool_calls) != len(values):
            raise ValueError("`tool_calls` and `values` must have the same length.")

        if is_errors is None:
            is_errors = [False] * len(tool_calls)
        elif len(is_errors) != len(tool_calls):
            raise ValueError("`is_errors` must have the same length as `tool_calls` when provided.")

        return cls(
            tool_call_results=[
                cls.ToolCallResult(call_id=tool_call.id, name=tool_call.name, value=value, is_error=is_error)
                for tool_call, value, is_error in zip(tool_calls, values, is_errors, strict=True)
            ]
        )

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: object) -> object:
        if isinstance(data, cls):
            return data

        if isinstance(data, list):
            return {"tool_call_results": data}

        if isinstance(data, dict):
            if "tool_call_results" in data:
                return data
            if {"name", "value"}.issubset(data):
                return {"tool_call_results": [data]}

        raise ValueError(f"Received invalid value for `dspy.adapters.types.tool.ToolCallResults`: {data}")


def _is_tool_call_dict(data: dict[str, Any]) -> bool:
    return ("name" in data and ("args" in data or "arguments" in data)) or "function" in data


def _normalize_tool_call_dict(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"Received invalid tool call value for `dspy.adapters.types.tool.ToolCalls`: {data}")

    if "function" in data:
        function = data.get("function") or {}
        if not isinstance(function, dict):
            raise ValueError(f"Received invalid function value for `dspy.adapters.types.tool.ToolCalls`: {function}")

        arguments = function.get("arguments", {})
        name = function.get("name") or data.get("name")
    else:
        arguments = data.get("args", data.get("arguments", {}))
        name = data.get("name")

    if isinstance(arguments, str):
        arguments = json_repair.loads(arguments)
    elif not isinstance(arguments, dict):
        arguments = {}

    return {
        "id": data.get("id") or data.get("call_id"),
        "name": name,
        "args": arguments,
    }


def _resolve_json_schema_reference(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve json model schema, expanding all references."""

    if "$defs" not in schema and "definitions" not in schema:
        return schema

    def resolve_refs(obj: object) -> object:
        if not isinstance(obj, dict | list):
            return obj
        if isinstance(obj, dict):
            obj = cast("dict[str, Any]", obj)
            if "$ref" in obj:
                ref_value = obj["$ref"]
                ref_path = ref_value.split("/")[-1] if isinstance(ref_value, str) else str(ref_value).split("/")[-1]
                return resolve_refs(schema["$defs"][ref_path])
            return {k: resolve_refs(v) for k, v in obj.items()}

        return [resolve_refs(item) for item in obj]

    resolved_schema = cast("dict[str, Any]", resolve_refs(schema))
    resolved_schema.pop("$defs", None)
    return resolved_schema


def convert_input_schema_to_tool_args(
    schema: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Type], dict[str, str]]:
    """Convert an input json schema to tool arguments compatible with DSPy Tool.

    Args:
        schema: An input json schema describing the tool's input parameters

    Returns:
        A tuple of (args, arg_types, arg_desc) for DSPy Tool definition.
    """
    args, arg_types, arg_desc = {}, {}, {}
    properties = schema.get("properties")
    if properties is None:
        return args, arg_types, arg_desc

    required = schema.get("required", [])

    defs = schema.get("$defs", {})

    for name, prop in properties.items():
        prop = cast("dict[str, Any]", prop)
        if len(defs) > 0:
            prop = _resolve_json_schema_reference({"$defs": defs, **prop})
        args[name] = prop
        prop_type = prop.get("type")
        arg_types[name] = _TYPE_MAPPING.get(prop_type, Any) if isinstance(prop_type, str) else Any
        arg_desc[name] = prop.get("description", "No description provided.")
        if name in required:
            arg_desc[name] += " (Required)"

    return args, arg_types, arg_desc
