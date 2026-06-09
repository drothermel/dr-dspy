from typing import Any

from dspy.task_spec.field_spec import _UNSET, FieldSpec, input_field, output_field
from dspy.task_spec.type_registry import type_from_str, type_to_str

TASK_SPEC_VERSION = 3

_REQUIRED_FIELD_SPEC_KEYS = ("type", "desc", "prefix", "role", "name")


def _require_field_spec_key(data: dict[str, Any], key: str, *, field_name: str | None = None) -> Any:
    if key not in data:
        context = f" for field {field_name!r}" if field_name is not None else ""
        raise ValueError(f"field_spec missing required key {key!r}{context}.")
    return data[key]


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
    field_name = data.get("name")
    for key in _REQUIRED_FIELD_SPEC_KEYS:
        _require_field_spec_key(data, key, field_name=field_name)
    role = data["role"]
    if role not in {"input", "output"}:
        raise ValueError(
            f"field_spec for field {field_name!r} has invalid role {role!r}; expected 'input' or 'output'."
        )
    has_default = data.get("has_default", False)
    if has_default and "default" not in data:
        raise ValueError(f"field_spec for field {field_name!r} has has_default=true but missing key 'default'.")
    type_ = type_from_str(data["type"], custom_types=custom_types)
    common = {"desc": data["desc"], "prefix": data["prefix"], "constraints": data.get("constraints")}
    if role == "input":
        return input_field(
            data["name"],
            type_,
            is_type_undefined=data.get("is_type_undefined", False),
            default=data["default"] if has_default else _UNSET,
            **common,
        )
    return output_field(data["name"], type_, **common)
