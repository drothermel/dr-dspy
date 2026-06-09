from __future__ import annotations

from typing import Any, Literal, cast, get_args, get_origin


def get_annotation_name(annotation: object) -> str:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is None:
        if hasattr(annotation, "__name__"):
            return cast("str", annotation.__name__)
        return str(annotation)
    if origin is Literal:
        args_str = ", ".join(
            _quoted_string_for_literal_type_annotation(a) if isinstance(a, str) else get_annotation_name(a)
            for a in args
        )
        return f"{get_annotation_name(origin)}[{args_str}]"
    args_str = ", ".join(get_annotation_name(a) for a in args)
    return f"{get_annotation_name(origin)}[{args_str}]"


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


def _quoted_string_for_literal_type_annotation(s: str) -> str:
    has_single = "'" in s
    has_double = '"' in s
    if has_single and (not has_double):
        return f'"{s}"'
    if has_double and (not has_single):
        return f"'{s}'"
    if has_single and has_double:
        escaped = s.replace("'", "\\'")
        return f"'{escaped}'"
    return f"'{s}'"
