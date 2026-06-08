from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, get_args, get_origin

import pydantic

if TYPE_CHECKING:
    from litellm import ModelResponseStream

    from dspy.core.types import LMOutput, LMPart


class Type(pydantic.BaseModel):
    """Base class to support creating custom types for DSPy signatures.

    This is the parent class of DSPy custom types, e.g, Image. Subclasses must implement the `format` method to
    return a list of dictionaries (same as the Array of content parts in the OpenAI API user message's content field).

    Examples:

        ```python
        class Image(Type):
            url: str

            def format(self) -> list[dict[str, Any]]:
                return [{"type": "image_url", "image_url": {"url": self.url}}]
        ```
    """

    def format(self) -> list[dict[str, Any]] | dict[str, Any] | str:
        raise NotImplementedError

    @classmethod
    def description(cls) -> str:
        """Description of the custom type"""
        return ""

    @classmethod
    def extract_custom_type_from_annotation(cls, annotation: object) -> list[type[Type]]:
        """Extract all custom types from the annotation.

        This is used to extract all custom types from the annotation of a field, while the annotation can
        have arbitrary level of nesting. For example, we detect `Tool` is in `list[dict[str, Tool]]`.
        """
        # Direct match. Nested type like `list[dict[str, Event]]` passes `isinstance(annotation, type)` in python 3.10
        # while fails in python 3.11. To accommodate users using python 3.10, we need to capture the error and ignore it.
        try:
            if isinstance(annotation, type) and issubclass(annotation, cls):
                return [annotation]
        except TypeError:
            pass

        origin = get_origin(annotation)
        if origin is None:
            return []

        result: list[type[Type]] = []
        # Recurse into all type args
        for arg in get_args(annotation):
            result.extend(cls.extract_custom_type_from_annotation(arg))

        return result

    def renders_as_content_blocks(self) -> bool:
        formatted = self.format()
        if not isinstance(formatted, list) or not formatted:
            return False
        return all(isinstance(block, dict) and "type" in block for block in formatted)

    def to_lm_parts(self) -> list[LMPart]:
        """Render this custom type as normalized LM parts."""
        from dspy.core.types import LMTextPart, _parts_from_openai_content

        formatted = self.format()
        if isinstance(formatted, str):
            return [LMTextPart(text=formatted)]
        if isinstance(formatted, list):
            return _parts_from_openai_content(formatted)
        if isinstance(formatted, dict):
            return _parts_from_openai_content([formatted])
        return [LMTextPart(text=str(formatted))]

    def to_content_blocks(self) -> list[dict[str, Any]]:
        """Render this custom type as OpenAI chat content blocks."""
        formatted = self.format()
        if isinstance(formatted, str):
            return [{"type": "text", "text": formatted}]
        if isinstance(formatted, list):
            return [block if isinstance(block, dict) else {"type": "text", "text": str(block)} for block in formatted]
        if isinstance(formatted, dict):
            return [(formatted)]
        return [{"type": "text", "text": str(formatted)}]

    @pydantic.model_serializer()
    def serialize_model(self) -> object:
        formatted = self.format()
        if isinstance(formatted, list):
            return json.dumps(formatted, ensure_ascii=False)
        return formatted

    @classmethod
    def is_streamable(cls) -> bool:
        """Whether the custom type is streamable."""
        return False

    @classmethod
    def parse_stream_chunk(cls, chunk: ModelResponseStream) -> Type | str | None:
        """
        Parse a stream chunk into the custom type.

        Args:
            chunk: A stream chunk.

        Returns:
            A custom type object or None if the chunk is not for this custom type.
        """
        _ = chunk
        return None

    @classmethod
    def parse_lm_output(cls, output: LMOutput) -> Type | None:
        """Parse one typed LM output into the custom type."""
        text = output.text
        if text is not None:
            parsed = cls.parse_lm_response(text)
            if parsed is not None:
                return parsed

        output_dict = output.to_output_dict()
        if output_dict:
            parsed = cls.parse_lm_response(output_dict)
            if parsed is not None:
                return parsed

        provider_output = output.provider_output
        if isinstance(provider_output, (str, dict)):
            return cls.parse_lm_response(provider_output)
        return None

    @classmethod
    def parse_lm_response(cls, response: str | dict[str, Any]) -> Type | None:
        _ = response
        return None
