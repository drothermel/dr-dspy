import inspect
import types
from typing import Any, Literal, Union, cast, get_args, get_origin

from pydantic import BaseModel
from typing_extensions import override

from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.utils import build_multimodal_user_message_content, inputs_include_multimodal_custom_type_values
from dspy.task_spec import TaskSpec
from dspy.task_spec import format_field_value as original_format_field_value

COMMENT_SYMBOL = "#"
INDENTATION = "  "


def _render_type_str(annotation: object, depth: int = 0, indent: int = 0, seen_models: set[type] | None = None) -> str:
    if annotation is str:
        return "string"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is bool:
        return "boolean"
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return _build_simplified_schema(pydantic_model=annotation, indent=indent, seen_models=seen_models)
    try:
        origin = get_origin(annotation)
        args = get_args(annotation)
    except Exception:
        return str(annotation)
    if origin in (types.UnionType, Union):
        non_none_args = [arg for arg in args if arg is not type(None)]
        type_render = " or ".join(
            [
                _render_type_str(annotation=arg, depth=depth + 1, indent=indent, seen_models=seen_models)
                for arg in non_none_args
            ]
        )
        if len(non_none_args) < len(args):
            return f"{type_render} or null"
        return type_render
    if origin is Literal:
        return " or ".join(f'"{arg}"' for arg in args)
    if origin is list:
        inner_type = args[0]
        if inspect.isclass(inner_type) and issubclass(inner_type, BaseModel):
            inner_schema = _build_simplified_schema(
                pydantic_model=inner_type, indent=indent + 1, seen_models=seen_models
            )
            current_indent = INDENTATION * indent
            return f"[\n{inner_schema}\n{current_indent}]"
        return f"{_render_type_str(annotation=inner_type, depth=depth + 1, indent=indent, seen_models=seen_models)}[]"
    if origin is dict:
        return f"dict[{_render_type_str(annotation=args[0], depth=depth + 1, indent=indent, seen_models=seen_models)}, {_render_type_str(annotation=args[1], depth=depth + 1, indent=indent, seen_models=seen_models)}]"
    if hasattr(annotation, "__name__"):
        return cast("str", annotation.__name__)
    return str(annotation)


def _build_simplified_schema(
    pydantic_model: type[BaseModel], indent: int = 0, seen_models: set[type] | None = None
) -> str:
    seen_models = seen_models or set()
    if pydantic_model in seen_models:
        raise ValueError("BAMLAdapter cannot handle recursive pydantic models, please use a different adapter.")
    seen_models.add(pydantic_model)
    lines = []
    current_indent = INDENTATION * indent
    next_indent = INDENTATION * (indent + 1)
    lines.append(f"{current_indent}{{")
    fields = pydantic_model.model_fields
    if not fields:
        lines.append(f"{next_indent}{COMMENT_SYMBOL} No fields defined")
    for name, field in fields.items():
        if field.description:
            lines.append(f"{next_indent}{COMMENT_SYMBOL} {field.description}")
        elif field.alias and field.alias != name:
            lines.append(f"{next_indent}{COMMENT_SYMBOL} alias: {field.alias}")
        field_annotation = field.annotation
        origin = get_origin(field_annotation)
        if origin in (types.UnionType, Union):
            args = get_args(field_annotation)
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                field_annotation = non_none_args[0]
        rendered_type = _render_type_str(annotation=field.annotation, indent=indent + 1, seen_models=seen_models)
        line = f"{next_indent}{name}: {rendered_type},"
        lines.append(line)
    lines.append(f"{current_indent}}}")
    return "\n".join(lines)


class BAMLAdapter(JSONAdapter):
    @override
    def format_field_structure(self, task_spec: TaskSpec) -> str:
        sections = []
        sections.append(
            "All interactions will be structured in the following way, with the appropriate values filled in.\n"
        )
        if task_spec.input_fields:
            for name in task_spec.input_fields:
                sections.append(f"[[ ## {name} ## ]]")
                sections.append(f"{{{name}}}")
                sections.append("")
        if task_spec.output_fields:
            for name, field in task_spec.output_fields.items():
                field_type = field.type_
                sections.append(f"[[ ## {name} ## ]]")
                if field.desc and field.desc != f"${{{name}}}":
                    sections.append(f"{COMMENT_SYMBOL} {field.desc}")
                sections.append(
                    f"Output field `{name}` should be of type: {_render_type_str(annotation=field_type, indent=0)}\n"
                )
        sections.append("[[ ## completed ## ]]")
        return "\n".join(sections)

    @override
    def format_user_message_content(
        self,
        task_spec: TaskSpec,
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str | list[dict[str, Any]]:
        if inputs_include_multimodal_custom_type_values(task_spec=task_spec, inputs=inputs):
            output_requirements = self.user_message_output_requirements(task_spec) if main_request else None
            return build_multimodal_user_message_content(
                task_spec=task_spec,
                inputs=inputs,
                prefix=prefix,
                suffix=suffix,
                main_request=main_request,
                output_requirements=output_requirements,
            )
        messages = [prefix]
        for key, field in task_spec.input_fields.items():
            if key in inputs:
                value = inputs.get(key)
                formatted_value = ""
                if isinstance(value, BaseModel):
                    formatted_value = value.model_dump_json(indent=2, by_alias=True)
                else:
                    formatted_value = original_format_field_value(field=field, value=value)
                messages.append(f"[[ ## {key} ## ]]\n{formatted_value}")
        if main_request:
            output_requirements = self.user_message_output_requirements(task_spec)
            if output_requirements is not None:
                messages.append(output_requirements)
        messages.append(suffix)
        return "\n\n".join(m for m in messages if m).strip()
