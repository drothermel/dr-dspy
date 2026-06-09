"""Render type annotations for diagnostics and adapter prompt display."""

from __future__ import annotations

from typing import Any, Literal, get_args, get_origin


def format_type_annotation(
    annotation: Any,
    *,
    quote_string_literals: bool = False,
) -> str:
    """Render a type annotation for diagnostics or prompt display."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is None:
        if hasattr(annotation, "__name__"):
            return annotation.__name__
        return str(annotation)
    if origin is Literal:
        if quote_string_literals:
            args_str = ", ".join(
                _quoted_string_for_literal(a)
                if isinstance(a, str)
                else format_type_annotation(a, quote_string_literals=True)
                for a in args
            )
        else:
            args_str = ", ".join(repr(arg) for arg in args)
        origin_name = getattr(origin, "__name__", str(origin))
        return f"{origin_name}[{args_str}]"
    if args:
        args_str = ", ".join(
            "..." if arg is ... else format_type_annotation(arg, quote_string_literals=quote_string_literals)
            for arg in args
        )
        origin_name = getattr(origin, "__name__", str(origin))
        return f"{origin_name}[{args_str}]"
    return getattr(origin, "__name__", str(origin))


def _quoted_string_for_literal(value: str) -> str:
    has_single = "'" in value
    has_double = '"' in value
    if has_single and not has_double:
        return f'"{value}"'
    if has_double and not has_single:
        return f"'{value}'"
    if has_single and has_double:
        escaped = value.replace("'", "\\'")
        return f"'{escaped}'"
    return f"'{value}'"
