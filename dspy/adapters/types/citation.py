from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, cast

import pydantic
from typing_extensions import override

from dspy._meta.experimental import experimental
from dspy.adapters.types.base_type import Type

if TYPE_CHECKING:
    from collections.abc import Iterator

    from litellm import ModelResponseStream

    from dspy.core.types import LMOutput


@experimental(version="3.0.4")
class Citations(Type):
    class Citation(Type):
        type: str = "char_location"
        cited_text: str
        document_index: int
        document_title: str | None = None
        start_char_index: int
        end_char_index: int
        supported_text: str | None = None

        @override
        def format(self) -> dict[str, Any]:
            citation_dict = {
                "type": self.type,
                "cited_text": self.cited_text,
                "document_index": self.document_index,
                "start_char_index": self.start_char_index,
                "end_char_index": self.end_char_index,
            }
            if self.document_title:
                citation_dict["document_title"] = self.document_title
            if self.supported_text:
                citation_dict["supported_text"] = self.supported_text
            return citation_dict

    citations: list[Citation]

    @classmethod
    def from_dict_list(cls, citations_dicts: list[dict[str, Any]]) -> Self:
        citations = [cls.Citation(**item) for item in citations_dicts]
        return cls(citations=citations)

    @classmethod
    @override
    def description(cls) -> str:
        return "Citations with quoted text and source references. Include the exact text being cited and information about its source."

    @override
    def format(self) -> list[dict[str, Any]]:
        return [citation.format() for citation in self.citations]

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: object) -> object:
        if isinstance(data, cls):
            return data
        if isinstance(data, list) and all(isinstance(item, dict) and "cited_text" in item for item in data):
            return {"citations": [cls.Citation(**cast("dict[str, Any]", item)) for item in data]}
        if isinstance(data, dict):
            data = cast("dict[str, Any]", data)
            if "citations" in data:
                citations_data = data["citations"]
                if isinstance(citations_data, list):
                    return {
                        "citations": [
                            cls.Citation(**cast("dict[str, Any]", item)) if isinstance(item, dict) else item
                            for item in citations_data
                        ]
                    }
            elif "cited_text" in data:
                return {"citations": [cls.Citation(**data)]}
        raise ValueError(f"Received invalid value for `Citations`: {data}")

    @override
    def __iter__(self) -> Iterator[Citation]:  # ty:ignore[invalid-method-override]
        return iter(self.citations)

    def __len__(self) -> int:
        return len(self.citations)

    def __getitem__(self, index: int) -> Citation:
        return self.citations[index]

    @classmethod
    @override
    def is_streamable(cls) -> bool:
        return True

    @classmethod
    @override
    def parse_stream_chunk(cls, chunk: ModelResponseStream) -> Type | str | None:
        try:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "provider_specific_fields") and delta.provider_specific_fields:
                    citation_data = delta.provider_specific_fields.get("citation")
                    if citation_data:
                        return cls.from_dict_list([citation_data])
        except Exception:
            pass
        return None

    @classmethod
    @override
    def parse_lm_output(cls, output: LMOutput) -> Type | None:
        if output.citations:
            return cls.from_dict_list([cls._citation_part_to_dict(citation) for citation in output.citations])
        return None

    @staticmethod
    def _citation_part_to_dict(citation: Any) -> dict[str, Any]:
        data = dict(getattr(citation, "metadata", {}) or {})
        if getattr(citation, "text", None) is not None:
            data["cited_text"] = citation.text
        if getattr(citation, "title", None) is not None:
            data["document_title"] = citation.title
        if getattr(citation, "url", None) is not None:
            data["url"] = citation.url
        return data
