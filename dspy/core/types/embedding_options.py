from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from dspy.core.types._merge_overlay import _merge_model_overlay


class EmbedderOptions(BaseModel):
    """Provider options forwarded to LiteLLM embedding calls or custom embedders."""

    model_config = ConfigDict(extra="forbid")

    dimensions: int | None = None
    encoding_format: str | None = None
    timeout: float | None = None

    def to_kwargs(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


def merge_embedder_options(
    left: EmbedderOptions | None,
    right: EmbedderOptions | None,
) -> EmbedderOptions:
    merged = _merge_model_overlay(
        left,
        right,
        model=EmbedderOptions,
        nested_fields=frozenset(),
    )
    return merged or EmbedderOptions()
