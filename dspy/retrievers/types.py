from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel, ConfigDict

__all__ = ["QueryRetriever", "RetrievedPassage"]

QueryT = TypeVar("QueryT", contravariant=True)


class RetrievedPassage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    long_text: str
    score: float | None = None
    pid: int | str | None = None
    metadata: dict[str, object] | None = None


class QueryRetriever(Protocol[QueryT]):
    k: int

    async def __call__(self, query: QueryT, /) -> list[RetrievedPassage]: ...

    async def aforward(self, query: QueryT, /) -> list[RetrievedPassage]: ...
