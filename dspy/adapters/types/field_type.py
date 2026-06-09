from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any, Protocol, get_args, get_origin, runtime_checkable

import pydantic

if TYPE_CHECKING:
    from dspy.core.types import LMOutput


@runtime_checkable
class FieldType(Protocol):
    def format(self) -> list[dict[str, Any]] | dict[str, Any] | str: ...


class NativeResponseFieldType(Protocol):
    @classmethod
    def parse_lm_output(cls, output: LMOutput) -> FieldTypeMixin | None: ...


class FieldTypeMixin(pydantic.BaseModel):
    def format(self) -> list[dict[str, Any]] | dict[str, Any] | str:
        raise NotImplementedError

    @classmethod
    def description(cls) -> str:
        return ""

    def renders_as_content_blocks(self) -> bool:
        formatted = self.format()
        if not isinstance(formatted, list) or not formatted:
            return False
        return all(isinstance(block, dict) and "type" in block for block in formatted)

    def to_content_blocks(self) -> list[dict[str, Any]]:
        formatted = self.format()
        if isinstance(formatted, str):
            return [{"type": "text", "text": formatted}]
        if isinstance(formatted, list):
            return [block if isinstance(block, dict) else {"type": "text", "text": str(block)} for block in formatted]
        if isinstance(formatted, dict):
            return [formatted]
        return [{"type": "text", "text": str(formatted)}]

    @pydantic.model_serializer()
    def serialize_model(self) -> object:
        formatted = self.format()
        if isinstance(formatted, list):
            return json.dumps(formatted, ensure_ascii=False)
        return formatted


def is_field_type(value: object) -> bool:
    return isinstance(value, FieldType)


def is_field_type_class(cls: object) -> bool:
    try:
        return inspect.isclass(cls) and issubclass(cls, FieldTypeMixin)
    except TypeError:
        return False


def extract_field_types_from_annotation(annotation: object) -> list[type[FieldTypeMixin]]:
    if is_field_type_class(annotation):
        return [annotation]  # ty:ignore[invalid-return-type]
    origin = get_origin(annotation)
    if origin is None:
        return []
    result: list[type[FieldTypeMixin]] = []
    for arg in get_args(annotation):
        result.extend(extract_field_types_from_annotation(arg))
    return result


def implements_parse_lm_output(cls: type[object]) -> bool:
    return "parse_lm_output" in cls.__dict__


def renders_as_content_blocks_value(value: object) -> bool:
    if isinstance(value, FieldTypeMixin):
        return value.renders_as_content_blocks()
    return False


def to_content_blocks_value(value: object) -> list[dict[str, Any]]:
    if isinstance(value, FieldTypeMixin):
        return value.to_content_blocks()
    return []
