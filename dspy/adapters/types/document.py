from typing import Any, Literal

import pydantic
from typing_extensions import override

from dspy.adapters.types.base_type import Type
from dspy.utils.annotation import experimental


@experimental(version="3.0.4")
class Document(Type):
    data: str
    title: str | None = None
    media_type: Literal["text/plain", "application/pdf"] = "text/plain"
    context: str | None = None

    @override
    def format(self) -> list[dict[str, Any]]:
        document_block = {
            "type": "document",
            "source": {"type": "text", "media_type": self.media_type, "data": self.data},
            "citations": {"enabled": True},
        }
        if self.title:
            document_block["title"] = self.title
        if self.context:
            document_block["context"] = self.context
        return [document_block]

    @classmethod
    @override
    def description(cls) -> str:
        return "A document containing text content that can be referenced and cited. Include the full text content and optionally a title for proper referencing."

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: object) -> object:
        if isinstance(data, cls):
            return data
        if isinstance(data, str):
            return {"data": data}
        if isinstance(data, dict):
            return data
        raise ValueError(f"Received invalid value for `Document`: {data}")

    @override
    def __str__(self) -> str:
        title_part = f"'{self.title}': " if self.title else ""
        return f"Document({title_part}{len(self.data)} chars)"
