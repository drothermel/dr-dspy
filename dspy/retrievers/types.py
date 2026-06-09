from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel, ConfigDict

__all__ = ["QueryRetriever", "RetrievedPassage"]

QueryT = TypeVar("QueryT", contravariant=True)
ResultT = TypeVar("ResultT", covariant=True)


class RetrievedPassage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    long_text: str
    score: float | None = None
    pid: int | None = None


class QueryRetriever(Protocol[QueryT, ResultT]):
    k: int

    async def __call__(self, query: QueryT, /) -> ResultT: ...

    async def aforward(self, query: QueryT, /) -> ResultT: ...
