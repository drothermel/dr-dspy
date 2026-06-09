from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast

import pydantic
from pydantic import TypeAdapter
from pydantic.json_schema import GetJsonSchemaHandler, JsonSchemaValue
from pydantic_core import CoreSchema
from typing_extensions import override

from dspy.adapters.types.base_type import Type
from dspy.adapters.utils.json_loads import load_json

if TYPE_CHECKING:
    from .tool import Tool


def _is_tool_call_dict(data: dict[str, Any]) -> bool:
    return ("name" in data and ("args" in data or "arguments" in data)) or "function" in data


def normalize_tool_call_dict(data: dict[str, Any], *, repair: bool = False) -> dict[str, Any]:
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
        arguments = load_json(arguments, repair=repair)
    elif not isinstance(arguments, dict):
        arguments = {}
    return {"id": data.get("id") or data.get("call_id"), "name": name, "args": arguments}


class ToolCalls(Type):
    class ToolCall(Type):
        id: str | None = None
        name: str
        args: dict[str, Any]

        @classmethod
        @override
        def __get_pydantic_json_schema__(
            cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
        ) -> JsonSchemaValue:
            schema = super().__get_pydantic_json_schema__(core_schema, handler)
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
            data: dict[str, Any] = {"name": self.name, "args": self.args}
            if self.id is not None:
                data["id"] = self.id
            return data

        def execute(self, functions: dict[str, Callable[..., object]] | Sequence["Tool"]) -> object:
            func: Callable[..., object] | None = None
            if isinstance(functions, dict):
                func = cast("dict[str, Callable[..., object]]", functions).get(self.name)
            else:
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
    def __get_pydantic_json_schema__(cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        schema = super().__get_pydantic_json_schema__(core_schema, handler)
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
        tool_calls = [cls.ToolCall(**normalize_tool_call_dict(item)) for item in tool_calls_dicts]
        return cls(tool_calls=tool_calls)

    @classmethod
    @override
    def description(cls) -> str:
        return 'Tool calls must be a JSON object with `tool_calls`, a list of calls. Each call must include `name` and `args`. Example: {"tool_calls": [{"name": "search", "args": {"query": "cats"}}]}'

    @override
    def format(self) -> dict[str, Any]:
        return {"tool_calls": [tool_call.format() for tool_call in self.tool_calls]}

    @pydantic.model_serializer()
    @override
    def serialize_model(self) -> dict[str, Any]:
        data = self.format()
        if self.tool_call_results is not None:
            data["tool_call_results"] = TypeAdapter(type(self.tool_call_results)).dump_python(
                self.tool_call_results, mode="json"
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
                "tool_calls": [cls.ToolCall(**normalize_tool_call_dict(cast("dict[str, Any]", item))) for item in data]
            }
        if isinstance(data, dict):
            data = cast("dict[str, Any]", data)
            if "tool_calls" in data:
                tool_calls_data = data["tool_calls"]
                if isinstance(tool_calls_data, list):
                    normalized = {
                        "tool_calls": [
                            cls.ToolCall(**normalize_tool_call_dict(cast("dict[str, Any]", item)))
                            if isinstance(item, dict)
                            else item
                            for item in tool_calls_data
                        ]
                    }
                    if "tool_call_results" in data:
                        normalized["tool_call_results"] = data["tool_call_results"]
                    return normalized
            elif _is_tool_call_dict(data):
                return {"tool_calls": [cls.ToolCall(**normalize_tool_call_dict(data))]}
        raise ValueError(f"Received invalid value for `dspy.adapters.types.tool.ToolCalls`: {data}")
