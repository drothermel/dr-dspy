"""Central registry for task-spec field type resolution (parse and serialize).

Adding a new DSPy adapter type:
1. Add ``"ShortName": "dspy.adapters.types.module"`` to ``DSPY_TYPE_MODULES``.
2. If the parse string uses an alias, add it to ``DSPY_TYPE_ALIASES``.
3. Add a round-trip test in ``tests/task_spec/test_serialize_parse.py``.
"""

from __future__ import annotations

import importlib
import types as types_module
from typing import Any, get_args, get_origin

BUILTIN_TYPES_BY_NAME: dict[str, type] = {
    "int": int,
    "str": str,
    "float": float,
    "bool": bool,
    "list": list,
    "tuple": tuple,
    "dict": dict,
    "set": set,
    "frozenset": frozenset,
    "complex": complex,
    "bytes": bytes,
    "bytearray": bytearray,
}

BUILTIN_TYPES_SERIALIZED: dict[str, type] = {f"builtins.{name}": type_ for name, type_ in BUILTIN_TYPES_BY_NAME.items()}
BUILTIN_TYPES_SERIALIZED["builtins.NoneType"] = type(None)

DSPY_TYPE_MODULES: dict[str, str] = {
    "Audio": "dspy.adapters.types.audio",
    "Code": "dspy.adapters.types.code",
    "File": "dspy.adapters.types.file",
    "TurnLog": "dspy.history.turn_log",
    "Image": "dspy.adapters.types.image",
    "Reasoning": "dspy.adapters.types.reasoning",
    "Tool": "dspy.adapters.types.tool",
}

# Parse-only aliases that resolve to a type attr in the module above.
DSPY_TYPE_ALIASES: dict[str, str] = {
    "ToolCalls": "ToolCalls",
    "ToolCallResults": "ToolCallResults",
}

_DSPY_TYPE_ALIASES_MODULES: dict[str, str] = dict.fromkeys(DSPY_TYPE_ALIASES, DSPY_TYPE_MODULES["Tool"])


def resolve_type_name(type_name: str, names: dict[str, Any]) -> Any:
    """Resolve a short type name during task-spec string parsing."""
    if type_name in names:
        return names[type_name]
    if type_name in BUILTIN_TYPES_BY_NAME:
        return BUILTIN_TYPES_BY_NAME[type_name]
    module_path = DSPY_TYPE_MODULES.get(type_name) or _DSPY_TYPE_ALIASES_MODULES.get(type_name)
    if module_path is not None:
        module = importlib.import_module(module_path)
        attr_name = DSPY_TYPE_ALIASES.get(type_name, type_name)
        resolved_type = getattr(module, attr_name)
        names[type_name] = resolved_type
        return resolved_type
    raise ValueError(f"Unknown type name: {type_name}. Provide it via custom_types=.")


def type_to_str(type_annotation: Any) -> str:
    if isinstance(type_annotation, type):
        return f"{type_annotation.__module__}.{type_annotation.__qualname__}"
    origin = get_origin(type_annotation)
    if origin is not None:
        args = get_args(type_annotation)
        origin_str = type_to_str(origin)
        args_str = ", ".join(type_to_str(arg) for arg in args)
        return f"{origin_str}[{args_str}]"
    return repr(type_annotation)


def _union_from_args(args: tuple[Any, ...]) -> Any:
    result = args[0]
    for arg in args[1:]:
        result = result | arg
    return result


def type_from_str(type_str: str, *, custom_types: dict[str, type] | None = None) -> Any:
    if custom_types and type_str in custom_types:
        return custom_types[type_str]
    if type_str == "Ellipsis":
        return Ellipsis
    if type_str in BUILTIN_TYPES_SERIALIZED:
        return BUILTIN_TYPES_SERIALIZED[type_str]
    if "[" in type_str and type_str.endswith("]"):
        origin_str, args_str = type_str.split("[", 1)
        args_str = args_str[:-1]
        origin = type_from_str(origin_str, custom_types=custom_types)
        arg_parts = split_generic_args(args_str)
        args = tuple(type_from_str(part.strip(), custom_types=custom_types) for part in arg_parts)
        if origin is types_module.UnionType:
            return _union_from_args(args)
        return origin[args]
    if type_str.startswith("builtins.") or "." in type_str:
        module_name, _, qualname = type_str.partition(".")
        if module_name == "builtins":
            builtin = BUILTIN_TYPES_SERIALIZED.get(type_str)
            if builtin is not None:
                return builtin
            raise ValueError(
                f"Unknown serialized field type {type_str!r}. Provide it via custom_types= or re-save with the current DSPy version."
            )
        try:
            if module_name == "types" and qualname == "UnionType":
                obj: Any = types_module.UnionType
            else:
                module = importlib.import_module(module_name)
                obj = module
                for part in qualname.split("."):
                    obj = getattr(obj, part)
        except (ImportError, AttributeError) as exc:
            raise ValueError(
                f"Unknown serialized field type {type_str!r}. Provide it via custom_types= or re-save with the current DSPy version."
            ) from exc
        return obj
    if custom_types:
        for key, value in custom_types.items():
            if type_to_str(value) == type_str or key == type_str:
                return value
    raise ValueError(
        f"Unknown serialized field type {type_str!r}. Provide it via custom_types= or re-save with the current DSPy version."
    )


def split_generic_args(args_str: str) -> list[str]:
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
