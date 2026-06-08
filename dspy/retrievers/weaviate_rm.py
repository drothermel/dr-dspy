from typing import Any, cast

from dspy.dsp.utils.utils import dotdict

try:
    from uuid import uuid4

    import weaviate
    from weaviate.util import get_valid_uuid
except ImportError as err:
    raise ImportError(
        "The 'weaviate' extra is required to use WeaviateRM. Install it with `pip install dspy-ai[weaviate]`",
    ) from err


class WeaviateRM:
    """A retrieval module that uses Weaviate to return the top passages for a given query.

    Assumes that a Weaviate collection has been created and populated with the following payload:
        - content: The text of the passage

    Args:
        weaviate_collection_name (str): The name of the Weaviate collection.
        weaviate_client (WeaviateClient): An instance of the Weaviate client.
        k (int, optional): The default number of top passages to retrieve. Default to 3.
        tenant_id (str, optional): The tenant to retrieve objects from.

    Examples:
        Below is a code snippet that shows how to query Weaviate directly:
        ```python
        import weaviate

        weaviate_client = weaviate.connect_to_[local, wcs, custom, embedded]("your-path-here")
        retriever_model = WeaviateRM("my_collection_name", weaviate_client=weaviate_client, k=1)
        topK_passages = [
            passage.long_text
            for passage in retriever_model("what are the stages in planning, sanctioning and execution of public works")
        ]
        ```

        Below is a code snippet that shows how to use Weaviate in the forward() function of a module
        ```python
        self.retrieve = WeaviateRM("my_collection_name", weaviate_client=weaviate_client, k=num_passages)
        ```
    """

    def __init__(
        self,
        weaviate_collection_name: str,
        weaviate_client: weaviate.WeaviateClient | weaviate.Client,
        weaviate_collection_text_key: str | None = "content",
        k: int = 3,
        tenant_id: str | None = None,
    ) -> None:
        self._weaviate_collection_name = weaviate_collection_name
        self._weaviate_client = weaviate_client
        self._weaviate_collection_text_key = weaviate_collection_text_key
        self._tenant_id = tenant_id

        # Check the type of weaviate_client (this is added to support v3 and v4)
        if hasattr(weaviate_client, "collections"):
            self._client_type = "WeaviateClient"
            self._weaviate_collection = cast("Any", weaviate_client).collections.get(
                self._weaviate_collection_name
            )
        elif hasattr(weaviate_client, "query"):
            self._client_type = "Client"
            self._weaviate_collection = None
        else:
            raise ValueError("Unsupported Weaviate client type")

        self.k = k

    def __call__(self, query_or_queries: str | list[str], k: int | None = None, **kwargs: object) -> list[dotdict]:
        return self.forward(query_or_queries=query_or_queries, k=k, **kwargs)

    def forward(self, query_or_queries: str | list[str], k: int | None = None, **kwargs: object) -> list[dotdict]:
        """Search with Weaviate for self.k top passages for query or queries.

        Args:
            query_or_queries (Union[str, list[str]]): The query or queries to search for.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.
            kwargs :

        Returns:
            dspy.primitives.prediction.Prediction: An object containing the retrieved passages.
        """
        k = k if k is not None else self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]
        passages: list[dotdict] = []
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

            passages.extend(dotdict({"long_text": d}) for d in parsed_results)

        return passages

    def get_objects(self, num_samples: int, fields: list[str]) -> list[dict[str, object]]:
        """Get objects from Weaviate using the cursor API."""
        if self._client_type == "WeaviateClient":
            objects = []
            collection = cast("Any", self._weaviate_collection)
            for counter, item in enumerate(collection.iterator()): # TODO: Apply tenant_id to object iteration; search is tenant-scoped but get_objects is not.
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
            cast("Any", self._weaviate_collection).data.insert(
                properties=cast("Any", new_object_properties),
                uuid=get_valid_uuid(uuid4())
            ) # TODO: Apply tenant_id to inserts; search is tenant-scoped but insert is not.
        else:
            raise AttributeError("`insert` is not supported for the v3 Weaviate Python client, please upgrade to v4.")
