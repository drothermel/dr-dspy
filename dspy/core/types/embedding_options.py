from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


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
    if left is None:
        return right or EmbedderOptions()
    if right is None:
        return left
    merged = left.model_dump()
    merged.update(right.model_dump(exclude_none=True))
    return EmbedderOptions(**merged)
