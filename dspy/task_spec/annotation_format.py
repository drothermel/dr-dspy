from __future__ import annotations

from typing import Any, Literal, get_args, get_origin


def get_type_name(type_annotation: Any) -> str:
    origin = get_origin(type_annotation)
    args = get_args(type_annotation)
    if origin is None:
        if hasattr(type_annotation, "__name__"):
            return type_annotation.__name__
        return str(type_annotation)
    if origin is Literal:
        literal_values = ", ".join(repr(arg) for arg in args)
        return f"Literal[{literal_values}]"
    if args:
        args_str = ", ".join("..." if arg is ... else get_type_name(arg) for arg in args)
        origin_name = getattr(origin, "__name__", str(origin))
        return f"{origin_name}[{args_str}]"
    return getattr(origin, "__name__", str(origin))
