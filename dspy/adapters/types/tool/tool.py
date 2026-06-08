import asyncio
import inspect
from collections.abc import Coroutine
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, get_origin, get_type_hints

from pydantic import BaseModel, TypeAdapter, create_model
from typing_extensions import override

from dspy.adapters.types.base_type import Type
from dspy.core.types import LMToolSpec
from dspy.dsp.utils.settings import settings

from .schema import _resolve_json_schema_reference, jsonschema

if TYPE_CHECKING:
    import mcp
    from langchain.tools import BaseTool  # ty: ignore[unresolved-import]


def tool_from_callable(func: Callable[..., object], *, description: str | None = None) -> "Tool":
    """Wrap a callable as a :class:`Tool`, using ``description`` or the callable docstring."""
    if isinstance(func, Tool):
        return func
    if not callable(func):
        raise TypeError(f"Expected callable or Tool, got {type(func).__name__}.")
    if description is None:
        annotations_func = func if inspect.isfunction(func) or inspect.ismethod(func) else func.__call__
        doc = getattr(func, "__doc__", None) or getattr(annotations_func, "__doc__", None)
        if doc and doc.strip():
            description = doc.strip()
        else:
            name = getattr(func, "__name__", type(func).__name__)
            description = f"Invoke {name}."
    return Tool(func, description=description)


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
        *,
        description: str,
        name: str | None = None,
        args: dict[str, Any] | None = None,
        arg_types: dict[str, Any] | None = None,
        arg_desc: dict[str, str] | None = None,
    ) -> None:
        """Initialize the Tool class.

        Users can choose to specify the `name`, `args`, and `arg_types`, or let the `Tool`
        automatically infer them from the function. For values that are specified by the user, automatic inference
        will not be performed on them.

        Args:
            func (Callable): The actual function that is being wrapped by the tool.
            description (str): Required description of the tool sent to the language model.
            name (str | None, optional): The name of the tool. Defaults to None.
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

        tool = Tool(foo, description="Concatenate an integer and a string.")
        print(tool.args)
        # Expected output: {'x': {'type': 'integer'}, 'y': {'type': 'string', 'default': 'hello'}}
        ```
        """
        if not description:
            raise ValueError("Tool description is required and must be non-empty.")
        super().__init__(func=func, name=name, desc=description, args=args, arg_types=arg_types, arg_desc=arg_desc)  # ty: ignore[unknown-argument]
        self._parse_function(func=func, arg_desc=arg_desc)

    def _parse_function(self, func: Callable, arg_desc: dict[str, str] | None = None) -> None:
        """Helper method that parses a function to extract the name, description, and args.

        This is a helper function that automatically infers the name, description, and args of the tool from the
        provided function. In order to make the inference work, the function must have valid type hints.
        """
        annotations_func = func if inspect.isfunction(func) or inspect.ismethod(func) else func.__call__
        name = getattr(func, "__name__", type(func).__name__)
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

    def to_lm_tool_spec(self) -> LMToolSpec:
        if self.name is None:
            raise ValueError("Tool name is required to produce an LMToolSpec.")
        args_schema = self.args or {}
        return LMToolSpec(
            name=self.name,
            description=self.desc,
            parameters={
                "type": "object",
                "properties": args_schema,
                "required": list(args_schema.keys()),
            },
        )

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

        return convert_mcp_tool(session=session, tool=tool)

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
        return f"Tool(name={self.name}, description={self.desc}, args={self.args})"

    @override
    def __str__(self) -> str:
        desc = f", whose description is <desc>{self.desc}</desc>.".replace("\n", "  ") if self.desc else "."
        arg_desc = f"It takes arguments {self.args}."
        return f"{self.name}{desc} {arg_desc}"
