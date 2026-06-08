from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pydantic
from typing_extensions import override

from dspy.adapters.types.base_type import Type
from dspy.utils.annotation import experimental

if TYPE_CHECKING:
    from collections.abc import Iterator

    from litellm import ModelResponseStream

    from dspy.core.types import LMOutput


@experimental(version="3.0.4")
class Citations(Type):
    """Citations extracted from an LM response with source references.

    This type represents citations returned by language models that support
    citation extraction, particularly Anthropic's Citations API through LiteLLM.
    Citations include the quoted text and source information.

    Examples:
        ```python
        import asyncio
        import os

        from dspy.adapters.types.citation import Citations
        from dspy.adapters.types.document import Document
        from dspy.clients.lm import LM
        from dspy.predict.predict import Predict
        from dspy.task_spec import FieldSpec, make_task_spec

        os.environ["ANTHROPIC_API_KEY"] = "YOUR_ANTHROPIC_API_KEY"

        task_spec = make_task_spec(
            {
                "documents": FieldSpec.input("documents", type_=list[Document]),
                "question": FieldSpec.input("question"),
                "answer": FieldSpec.output("answer"),
                "citations": FieldSpec.output("citations", type_=Citations),
            },
            instructions="Answer questions using provided documents with citations.",
        )

        docs = [
            Document(
                data="The Earth orbits the Sun in an elliptical path.",
                title="Basic Astronomy Facts",
            ),
            Document(
                data="Water boils at 100°C at standard atmospheric pressure.",
                title="Physics Fundamentals",
                metadata={"author": "Dr. Smith", "year": 2023},
            ),
        ]

        lm = LM("anthropic/claude-opus-4-1-20250805")
        predictor = Predict(task_spec)
        result = asyncio.run(predictor(documents=docs, question="What temperature does water boil?", lm=lm))

        for citation in result.citations.citations:
            print(citation.format())
        ```
    """

    class Citation(Type):
        """Individual citation with character location information."""

        type: str = "char_location"
        cited_text: str
        document_index: int
        document_title: str | None = None
        start_char_index: int
        end_char_index: int
        supported_text: str | None = None

        @override
        def format(self) -> dict[str, Any]:
            """Format citation as dictionary for LM consumption.

            Returns:
                A dictionary in the format expected by citation APIs.
            """
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
    def from_dict_list(cls, citations_dicts: list[dict[str, Any]]) -> Citations:
        """Convert a list of dictionaries to a Citations instance.

        Args:
            citations_dicts: A list of dictionaries, where each dictionary should have 'cited_text' key
                and 'document_index', 'start_char_index', 'end_char_index' keys.

        Returns:
            A Citations instance.

        Examples:
            ```python
            citations_dict = [
                {
                    "cited_text": "The sky is blue",
                    "document_index": 0,
                    "document_title": "Weather Guide",
                    "start_char_index": 0,
                    "end_char_index": 15,
                    "supported_text": "The sky was blue yesterday."
                }
            ]
            citations = Citations.from_dict_list(citations_dict)
            ```
        """
        citations = [cls.Citation(**item) for item in citations_dicts]
        return cls(citations=citations)

    @classmethod
    @override
    def description(cls) -> str:
        """Description of the citations type for use in prompts."""
        return (
            "Citations with quoted text and source references. "
            "Include the exact text being cited and information about its source."
        )

    @override
    def format(self) -> list[dict[str, Any]]:
        """Format citations as a list of dictionaries."""
        return [citation.format() for citation in self.citations]

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: object) -> object:
        if isinstance(data, cls):
            return data

        # Handle case where data is a list of dicts with citation info
        if isinstance(data, list) and all(isinstance(item, dict) and "cited_text" in item for item in data):
            return {"citations": [cls.Citation(**cast("dict[str, Any]", item)) for item in data]}

        # Handle case where data is a dict
        if isinstance(data, dict):
            data = cast("dict[str, Any]", data)
            if "citations" in data:
                # Handle case where data is a dict with "citations" key
                citations_data = data["citations"]
                if isinstance(citations_data, list):
                    return {
                        "citations": [
                            cls.Citation(**cast("dict[str, Any]", item)) if isinstance(item, dict) else item
                            for item in citations_data
                        ]
                    }
            elif "cited_text" in data:
                # Handle case where data is a single citation dict
                return {"citations": [cls.Citation(**data)]}

        raise ValueError(f"Received invalid value for `Citations`: {data}")

    @override
    def __iter__(self) -> Iterator[Citation]:  # ty: ignore[invalid-method-override]
        """Allow iteration over citations."""
        return iter(self.citations)

    def __len__(self) -> int:
        """Return the number of citations."""
        return len(self.citations)

    def __getitem__(self, index: int) -> Citation:
        """Allow indexing into citations."""
        return self.citations[index]

    @classmethod
    @override
    def is_streamable(cls) -> bool:
        """Whether the Citations type is streamable."""
        return True

    @classmethod
    @override
    def parse_stream_chunk(cls, chunk: ModelResponseStream) -> Type | str | None:
        """
        Parse a stream chunk into Citations.

        Args:
            chunk: A stream chunk from the LM.

        Returns:
            A Citations object if the chunk contains citation data, None otherwise.
        """
        try:
            # Check if the chunk has citation data in provider_specific_fields
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
        """Parse a typed LM output into Citations."""
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
