"""Serialization helpers for TaskSpec instances."""

from typing import Any, get_args, get_origin

from dspy.task_spec.field_spec import _UNSET, FieldSpec

TASK_SPEC_VERSION = 2


def field_spec_to_dict(field: FieldSpec) -> dict[str, Any]:
    data = {
        "name": field.name,
        "type": _type_to_str(field.type_),
        "desc": field.desc,
        "role": field.role,
        "prefix": field.prefix,
        "is_type_undefined": field.is_type_undefined,
        "constraints": field.constraints,
    }
    if field.has_default:
        data["has_default"] = True
        data["default"] = field.default
    return data


def field_spec_from_dict(data: dict[str, Any], *, custom_types: dict[str, type] | None = None) -> FieldSpec:
    type_ = _type_from_str(data["type"], custom_types=custom_types)
    common = {
        "desc": data["desc"],
        "prefix": data["prefix"],
        "constraints": data.get("constraints"),
    }
    if data["role"] == "input":
        return FieldSpec.input(
            data["name"],
            type_,
            is_type_undefined=data.get("is_type_undefined", False),
            default=data["default"] if data.get("has_default") else _UNSET,
            **common,
        )
    return FieldSpec.output(data["name"], type_, **common)


def _type_to_str(type_annotation: Any) -> str:
    if isinstance(type_annotation, type):
        return f"{type_annotation.__module__}.{type_annotation.__qualname__}"
    origin = get_origin(type_annotation)
    if origin is not None:
        args = get_args(type_annotation)
        origin_str = _type_to_str(origin)
        args_str = ", ".join(_type_to_str(arg) for arg in args)
        return f"{origin_str}[{args_str}]"
    return repr(type_annotation)


def _type_from_str(type_str: str, *, custom_types: dict[str, type] | None = None) -> Any:
    if custom_types and type_str in custom_types:
        return custom_types[type_str]

    builtins_map = {
        "builtins.str": str,
        "builtins.int": int,
        "builtins.float": float,
        "builtins.bool": bool,
        "builtins.list": list,
        "builtins.tuple": tuple,
        "builtins.dict": dict,
        "builtins.set": set,
        "builtins.frozenset": frozenset,
        "builtins.complex": complex,
        "builtins.bytes": bytes,
        "builtins.bytearray": bytearray,
    }
    if type_str in builtins_map:
        return builtins_map[type_str]

    if "[" in type_str and type_str.endswith("]"):
        origin_str, args_str = type_str.split("[", 1)
        args_str = args_str[:-1]
        origin = _type_from_str(origin_str, custom_types=custom_types)
        arg_parts = _split_generic_args(args_str)
        args = tuple(_type_from_str(part.strip(), custom_types=custom_types) for part in arg_parts)
        return origin[args]

    if type_str.startswith("builtins.") or "." in type_str:
        module_name, _, qualname = type_str.partition(".")
        if module_name == "builtins":
            return builtins_map.get(type_str, str)
        import importlib

        module = importlib.import_module(module_name)
        obj: Any = module
        for part in qualname.split("."):
            obj = getattr(obj, part)
        return obj

    if custom_types:
        for key, value in custom_types.items():
            if _type_to_str(value) == type_str or key == type_str:
                return value

    return str


def _split_generic_args(args_str: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for char in args_str:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts
