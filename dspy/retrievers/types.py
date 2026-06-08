from __future__ import annotations

from typing import Protocol, TypeVar

QueryT = TypeVar("QueryT", contravariant=True)
ResultT = TypeVar("ResultT", covariant=True)


class QueryRetriever(Protocol[QueryT, ResultT]):
    """Structural protocol for direct-call query retrievers.

    Implementations may expose additional optional query parameters and may return
    retriever-specific result shapes.
    """

    k: int

    def __call__(self, query: QueryT, /) -> ResultT: ...

    def forward(self, query: QueryT, /) -> ResultT: ...


__all__ = ["QueryRetriever"]
