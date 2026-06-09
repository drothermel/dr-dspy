import asyncio
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from dspy._internal.lazy_import import _detect_dspy_dist
from dspy.retrievers.types import RetrievedPassage

if TYPE_CHECKING:
    import weaviate


def _require_weaviate() -> Any:
    try:
        import weaviate as weaviate_module

        return weaviate_module
    except ImportError as err:
        raise ImportError(
            f"The 'weaviate' extra is required to use WeaviateRM. "
            f"Install it with `pip install {_detect_dspy_dist()}[weaviate]`"
        ) from err


class WeaviateRM:
    def __init__(
        self,
        weaviate_collection_name: str,
        weaviate_client: "weaviate.WeaviateClient | weaviate.Client",
        weaviate_collection_text_key: str | None = "content",
        k: int = 3,
        tenant_id: str | None = None,
    ) -> None:
        _require_weaviate()
        self._weaviate_collection_name = weaviate_collection_name
        self._weaviate_client = weaviate_client
        self._weaviate_collection_text_key = weaviate_collection_text_key
        self._tenant_id = tenant_id
        if hasattr(weaviate_client, "collections"):
            self._client_type = "WeaviateClient"
            self._weaviate_collection = cast("Any", weaviate_client).collections.get(self._weaviate_collection_name)
        elif hasattr(weaviate_client, "query"):
            self._client_type = "Client"
            self._weaviate_collection = None
        else:
            raise ValueError("Unsupported Weaviate client type")
        self.k = k

    async def __call__(
        self, query_or_queries: str | list[str], k: int | None = None, **kwargs: object
    ) -> list[RetrievedPassage]:
        return await self.aforward(query_or_queries, k=k, **kwargs)

    async def aforward(
        self, query_or_queries: str | list[str], k: int | None = None, **kwargs: object
    ) -> list[RetrievedPassage]:
        return await asyncio.to_thread(self._search, query_or_queries, k, **kwargs)

    def _search(
        self, query_or_queries: str | list[str], k: int | None = None, **kwargs: object
    ) -> list[RetrievedPassage]:
        k = k if k is not None else self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]
        passages: list[RetrievedPassage] = []
        parsed_results = []
        tenant = kwargs.pop("tenant_id", self._tenant_id)
        for query in queries:
            if self._client_type == "WeaviateClient":
                collection = cast("Any", self._weaviate_collection)
                if tenant:
                    results = collection.query.with_tenant(tenant).hybrid(query=query, limit=k, **kwargs)
                else:
                    results = collection.query.hybrid(query=query, limit=k, **kwargs)
                parsed_results = [result.properties[self._weaviate_collection_text_key] for result in results.objects]
            elif self._client_type == "Client":
                q = cast("Any", self._weaviate_client).query.get(
                    self._weaviate_collection_name, [self._weaviate_collection_text_key]
                )
                if tenant:
                    q = q.with_tenant(tenant)
                results = q.with_hybrid(query=query).with_limit(k).do()
                results = results["data"]["Get"][self._weaviate_collection_name]
                parsed_results = [result[self._weaviate_collection_text_key] for result in results]
            passages.extend(RetrievedPassage(long_text=d) for d in parsed_results)
        return passages

    def get_objects(self, num_samples: int, fields: list[str]) -> list[dict[str, object]]:
        if self._client_type == "WeaviateClient":
            objects = []
            collection = cast("Any", self._weaviate_collection)
            for counter, item in enumerate(collection.iterator()):
                if counter >= num_samples:
                    break
                new_object = {}
                for key in item.properties:
                    if key in fields:
                        new_object[key] = item.properties[key]
                objects.append(new_object)
            return cast("list[dict[str, object]]", objects)
        raise ValueError("`get_objects` is not supported for the v3 Weaviate Python client, please upgrade to v4.")

    def insert(self, new_object_properties: dict[str, object]) -> None:
        if self._client_type == "WeaviateClient":
            _require_weaviate()
            from weaviate.util import get_valid_uuid

            cast("Any", self._weaviate_collection).data.insert(
                properties=cast("Any", new_object_properties), uuid=get_valid_uuid(uuid4())
            )
        else:
            raise AttributeError("`insert` is not supported for the v3 Weaviate Python client, please upgrade to v4.")
