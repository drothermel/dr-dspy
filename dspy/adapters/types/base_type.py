from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, get_args, get_origin

import pydantic

from dspy.core.types import parts_from_openai_content
from dspy.core.types.parts import LMTextPart

if TYPE_CHECKING:
    from litellm import ModelResponseStream

    from dspy.core.types import LMOutput
    from dspy.core.types.parts import LMPart


class Type(pydantic.BaseModel):
    def format(self) -> list[dict[str, Any]] | dict[str, Any] | str:
        raise NotImplementedError

    @classmethod
    def description(cls) -> str:
        return ""

    @classmethod
    def extract_custom_type_from_annotation(cls, annotation: object) -> list[type[Type]]:
        try:
            if isinstance(annotation, type) and issubclass(annotation, cls):
                return [annotation]
        except TypeError:
            pass
        origin = get_origin(annotation)
        if origin is None:
            return []
        result: list[type[Type]] = []
        for arg in get_args(annotation):
            result.extend(cls.extract_custom_type_from_annotation(arg))
        return result

    def renders_as_content_blocks(self) -> bool:
        formatted = self.format()
        if not isinstance(formatted, list) or not formatted:
            return False
        return all(isinstance(block, dict) and "type" in block for block in formatted)

    def to_lm_parts(self) -> list[LMPart]:
        formatted = self.format()
        if isinstance(formatted, str):
            return [LMTextPart(text=formatted)]
        if isinstance(formatted, list):
            return parts_from_openai_content(formatted)
        if isinstance(formatted, dict):
            return parts_from_openai_content([formatted])
        return [LMTextPart(text=str(formatted))]

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

    @classmethod
    def is_streamable(cls) -> bool:
        return False

    @classmethod
    def parse_stream_chunk(cls, chunk: ModelResponseStream) -> Type | str | None:
        _ = chunk
        return None

    @classmethod
    def parse_lm_output(cls, output: LMOutput) -> Type | None:
        _ = output
        return None
