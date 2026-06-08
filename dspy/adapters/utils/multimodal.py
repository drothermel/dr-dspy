import json
from collections.abc import Mapping
from typing import Any, cast

import pydantic
from pydantic.fields import FieldInfo

from dspy.adapters.types.base_type import Type as DspyType
from dspy.adapters.utils.fields import format_field_value
from dspy.task_spec import TaskSpec


def _parse_serialized_content_block_string(value: object) -> list[dict[str, Any]] | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped.startswith("["):
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    if all(isinstance(block, dict) and "type" in block for block in parsed):
        return cast("list[dict[str, Any]]", parsed)
    return None


def value_contains_multimodal_custom_type(value: object) -> bool:
    if isinstance(value, DspyType):
        return value.renders_as_content_blocks()
    if _parse_serialized_content_block_string(value) is not None:
        return True
    if isinstance(value, list):
        return any(value_contains_multimodal_custom_type(item) for item in value)
    if isinstance(value, dict):
        return any(value_contains_multimodal_custom_type(item) for item in value.values())
    if isinstance(value, pydantic.BaseModel):
        return any(value_contains_multimodal_custom_type(getattr(value, name)) for name in type(value).model_fields)
    return False


def inputs_include_multimodal_custom_type_values(task_spec: TaskSpec, inputs: Mapping[str, Any]) -> bool:
    for field_name in task_spec.input_fields:
        if field_name in inputs and value_contains_multimodal_custom_type(inputs[field_name]):
            return True
    return False


def collect_multimodal_content_blocks(value: object) -> list[dict[str, Any]]:
    if isinstance(value, DspyType):
        return value.to_content_blocks() if value.renders_as_content_blocks() else []
    if blocks := _parse_serialized_content_block_string(value):
        return blocks
    if isinstance(value, list):
        blocks: list[dict[str, Any]] = []
        for item in value:
            blocks.extend(collect_multimodal_content_blocks(item))
        return blocks
    if isinstance(value, dict):
        blocks = []
        for item in value.values():
            blocks.extend(collect_multimodal_content_blocks(item))
        return blocks
    if isinstance(value, pydantic.BaseModel):
        blocks = []
        for name in type(value).model_fields:
            blocks.extend(collect_multimodal_content_blocks(getattr(value, name)))
        return blocks
    return []


def field_value_to_content_blocks(
    field_info: FieldInfo,
    field_name: str,
    value: object,
    *,
    prefix: str = "",
    field_wrapper: str | None = None,
) -> list[dict[str, Any]]:
    if field_wrapper == "xml":
        open_tag = f"{prefix}<{field_name}>\n"
        close_tag = f"\n</{field_name}>"
        if isinstance(value, DspyType) and value.renders_as_content_blocks():
            return [{"type": "text", "text": open_tag}, *value.to_content_blocks(), {"type": "text", "text": close_tag}]
        nested_blocks = collect_multimodal_content_blocks(value)
        if nested_blocks:
            return [{"type": "text", "text": open_tag}, *nested_blocks, {"type": "text", "text": close_tag}]
        formatted_field_value = format_field_value(field_info=field_info, value=value)
        return [{"type": "text", "text": f"{open_tag}{formatted_field_value}{close_tag}"}]

    header = f"{prefix}[[ ## {field_name} ## ]]\n"
    if isinstance(value, DspyType) and value.renders_as_content_blocks():
        return [{"type": "text", "text": header}, *value.to_content_blocks()]
    nested_blocks = collect_multimodal_content_blocks(value)
    if nested_blocks:
        return [{"type": "text", "text": header}, *nested_blocks]
    formatted_field_value = format_field_value(field_info=field_info, value=value)
    return [{"type": "text", "text": f"{header}{formatted_field_value}"}]


def build_multimodal_user_message_content(
    task_spec: TaskSpec,
    inputs: Mapping[str, Any],
    *,
    prefix: str = "",
    suffix: str = "",
    main_request: bool = False,
    output_requirements: str | None = None,
    field_wrapper: str | None = None,
) -> list[dict[str, Any]]:
    from dspy.task_spec.pydantic_bridge import task_spec_input_field_infos

    blocks: list[dict[str, Any]] = []
    if prefix:
        blocks.append({"type": "text", "text": prefix})

    field_blocks_added = False
    input_field_infos = task_spec_input_field_infos(task_spec)
    for field_name in task_spec.input_fields:
        if field_name not in inputs:
            continue
        field_prefix = "\n\n" if field_blocks_added else ""
        field_blocks_added = True
        blocks.extend(
            field_value_to_content_blocks(
                field_info=input_field_infos[field_name],
                field_name=field_name,
                value=inputs[field_name],
                prefix=field_prefix,
                field_wrapper=field_wrapper,
            )
        )

    if main_request and output_requirements is not None:
        blocks.append({"type": "text", "text": f"\n\n{output_requirements}"})

    if suffix:
        blocks.append({"type": "text", "text": suffix})
    return blocks
