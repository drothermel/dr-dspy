from typing import Any, Protocol, cast

from dspy.adapters.types.base_type import Type
from dspy.utils.lazy_import import require

_TYPE_MAPPING = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}
jsonschema = require("jsonschema", extra="tools", feature="dspy.adapters.types.tool.Tool argument validation")


class PydanticJsonSchemaHandler(Protocol):
    def resolve_ref_schema(self, schema: dict[str, Any]) -> dict[str, Any]: ...


def _resolve_json_schema_reference(schema: dict[str, Any]) -> dict[str, Any]:
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


def convert_input_schema_to_tool_args(schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Type], dict[str, str]]:
    args, arg_types, arg_desc = ({}, {}, {})
    properties = schema.get("properties")
    if properties is None:
        return (args, arg_types, arg_desc)
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
    return (args, arg_types, arg_desc)
