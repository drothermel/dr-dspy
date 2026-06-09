from typing import Any

from dspy.task_spec.field_spec import _UNSET, FieldSpec, input_field, output_field
from dspy.task_spec.type_registry import type_from_str, type_to_str

TASK_SPEC_VERSION = 3


def field_spec_to_dict(field: FieldSpec) -> dict[str, Any]:
    data = {
        "name": field.name,
        "type": type_to_str(field.type_),
        "desc": field.desc,
        "role": field.role.value,
        "prefix": field.prefix,
        "is_type_undefined": field.is_type_undefined,
        "constraints": field.constraints,
    }
    if field.has_default:
        data["has_default"] = True
        data["default"] = field.default
    return data


def field_spec_from_dict(data: dict[str, Any], *, custom_types: dict[str, type] | None = None) -> FieldSpec:
    type_ = type_from_str(data["type"], custom_types=custom_types)
    common = {"desc": data["desc"], "prefix": data["prefix"], "constraints": data.get("constraints")}
    if data["role"] == "input":
        return input_field(
            data["name"],
            type_,
            is_type_undefined=data.get("is_type_undefined", False),
            default=data["default"] if data.get("has_default") else _UNSET,
            **common,
        )
    return output_field(data["name"], type_, **common)
