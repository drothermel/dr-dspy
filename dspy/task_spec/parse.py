"""Parse string-based task spec declarations into FieldSpec lists."""

import ast
import importlib
import typing
from collections.abc import Iterator
from typing import Any, cast

from pydantic import Field

from dspy.task_spec.field_spec import FieldSpec


def parse_task_spec_string(
    spec: str,
    *,
    custom_types: dict[str, type] | None = None,
) -> tuple[tuple[FieldSpec, ...], tuple[FieldSpec, ...]]:
    """Parse ``"input1, input2 -> output"`` into input and output FieldSpec tuples."""
    if spec.count("->") != 1:
        raise ValueError(f"Invalid task spec format: '{spec}', must contain exactly one '->'.")

    inputs_str, outputs_str = spec.split("->")
    input_fields = list(_parse_field_string(inputs_str, custom_types))
    output_fields = list(_parse_field_string(outputs_str, custom_types))
    duplicate_field_names = sorted(
        {field_name for field_name, *_ in input_fields}.intersection(field_name for field_name, *_ in output_fields)
    )
    if duplicate_field_names:
        raise ValueError(
            "Input and output fields must have distinct names, but found duplicates: "
            f"'{', '.join(duplicate_field_names)}'."
        )

    inputs = tuple(
        FieldSpec.input(name, type_, is_type_undefined=is_type_undefined)
        for name, type_, is_type_undefined in input_fields
    )
    outputs = tuple(FieldSpec.output(name, type_) for name, type_, _ in output_fields)
    return inputs, outputs


def _parse_field_string(field_string: str, custom_types: dict[str, type] | None) -> Iterator[tuple[str, type, bool]]:
    names = _typing_names_for_parse(custom_types)
    function_def = cast("ast.FunctionDef", ast.parse(f"def f({field_string}): pass").body[0])
    args = function_def.args.args
    field_names: list[str] = [arg.arg for arg in args]
    types_list: list[type] = [
        str if arg.annotation is None else _parse_type_node(arg.annotation, names) for arg in args
    ]
    is_type_undefined: list[bool] = [arg.annotation is None for arg in args]
    return zip(field_names, types_list, is_type_undefined, strict=False)


def _typing_names_for_parse(custom_types: dict[str, type] | None) -> dict[str, Any]:
    names = dict(typing.__dict__)
    names.pop("Union", None)
    names.pop("Optional", None)
    names["NoneType"] = type(None)
    if custom_types:
        names.update(custom_types)
    return names


def _parse_type_node(node, names: dict[str, Any]) -> Any:
    dspy_type_modules = {
        "Audio": "dspy.adapters.types.audio",
        "Code": "dspy.adapters.types.code",
        "File": "dspy.adapters.types.file",
        "History": "dspy.adapters.types.history",
        "Image": "dspy.adapters.types.image",
        "Reasoning": "dspy.adapters.types.reasoning",
        "Tool": "dspy.adapters.types.tool",
        "ToolCalls": "dspy.adapters.types.tool",
        "ToolCallResults": "dspy.adapters.types.tool",
    }

    def resolve_name(type_name: str):
        if type_name in names:
            return names[type_name]
        builtin_types = [int, str, float, bool, list, tuple, dict, set, frozenset, complex, bytes, bytearray]
        for builtin_type in builtin_types:
            if builtin_type.__name__ == type_name:
                return builtin_type

        if type_name in dspy_type_modules:
            module = importlib.import_module(dspy_type_modules[type_name])
            resolved_type = getattr(module, type_name)
            names[type_name] = resolved_type
            return resolved_type

        try:
            mod = importlib.import_module(type_name)
            names[type_name] = mod
            return mod
        except ImportError:
            pass

        raise ValueError(f"Unknown type name: {type_name}. Provide it via custom_types=.")

    if isinstance(node, ast.Module):
        if len(node.body) != 1:
            raise ValueError(f"Code is not syntactically valid: {ast.dump(node)}")
        return _parse_type_node(node.body[0], names)

    if isinstance(node, ast.Expr):
        return _parse_type_node(node.value, names)

    if isinstance(node, ast.Name):
        return resolve_name(node.id)

    if isinstance(node, ast.Attribute):
        base = _parse_type_node(node.value, names)
        attr_name = node.attr
        if hasattr(base, attr_name):
            return getattr(base, attr_name)
        if isinstance(node.value, ast.Name):
            full_name = f"{node.value.id}.{attr_name}"
            if full_name in names:
                return names[full_name]
        raise ValueError(f"Unknown attribute: {attr_name} on {base}")

    if isinstance(node, ast.Subscript):
        base_type = _parse_type_node(node.value, names)
        slice_node = node.slice
        index_type = getattr(ast, "Index", None)
        if index_type is not None and isinstance(slice_node, index_type):
            slice_node = slice_node.value

        if isinstance(slice_node, ast.Tuple):
            arg_types = tuple(_parse_type_node(elt, names) for elt in slice_node.elts)
        else:
            arg_types = (_parse_type_node(slice_node, names),)
        return base_type[arg_types]

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _parse_type_node(node.left, names)
        right = _parse_type_node(node.right, names)
        return left | right

    if isinstance(node, ast.Tuple):
        return tuple(_parse_type_node(elt, names) for elt in node.elts)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Field":
        keys = [kw.arg for kw in node.keywords]
        values = []
        for kw in node.keywords:
            if isinstance(kw.value, ast.Constant):
                values.append(kw.value.value)
            else:
                values.append(_parse_type_node(kw.value, names))
        return Field(**dict(zip(keys, values, strict=False)))

    raise ValueError(
        f"Failed to parse task spec string due to unhandled AST node type: {ast.dump(node)}. "
        "Use make_task_spec with an explicit field dict for complex annotations."
    )
