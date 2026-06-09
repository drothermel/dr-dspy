import asyncio
import json
import os
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

from dspy.primitives.prediction import Prediction


def _is_databricks_sdk_installed() -> bool:
    try:
        return find_spec("databricks.sdk") is not None
    except ModuleNotFoundError:
        return False


_databricks_sdk_installed = _is_databricks_sdk_installed()


class DatabricksRMError(Exception):
    pass


@dataclass
class Document:
    page_content: str
    metadata: dict[str, Any]
    type: str

    def to_dict(self) -> dict[str, Any]:
        return {"page_content": self.page_content, "metadata": self.metadata, "type": self.type}


class DatabricksRM:
    def __init__(
        self,
        databricks_index_name: str,
        databricks_endpoint: str | None = None,
        databricks_token: str | None = None,
        databricks_client_id: str | None = None,
        databricks_client_secret: str | None = None,
        columns: list[str] | None = None,
        filters_json: str | None = None,
        k: int = 3,
        docs_id_column_name: str = "id",
        docs_uri_column_name: str | None = None,
        text_column_name: str = "text",
        use_with_databricks_agent_framework: bool = False,
    ) -> None:
        self.databricks_token = databricks_token if databricks_token is not None else os.environ.get("DATABRICKS_TOKEN")
        self.databricks_endpoint = (
            databricks_endpoint if databricks_endpoint is not None else os.environ.get("DATABRICKS_HOST")
        )
        self.databricks_client_id = (
            databricks_client_id if databricks_client_id is not None else os.environ.get("DATABRICKS_CLIENT_ID")
        )
        self.databricks_client_secret = (
            databricks_client_secret
            if databricks_client_secret is not None
            else os.environ.get("DATABRICKS_CLIENT_SECRET")
        )
        if not _databricks_sdk_installed and (self.databricks_token, self.databricks_endpoint).count(None) > 0:
            raise ValueError(
                "To retrieve documents with Databricks Vector Search, you must install the databricks-sdk Python library, supply the databricks_token and databricks_endpoint parameters, or set the DATABRICKS_TOKEN and DATABRICKS_HOST environment variables. You may also supply a service principal the databricks_client_id and databricks_client_secret parameters, or set the DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET"
            )
        self.databricks_index_name = databricks_index_name
        self.columns = list({docs_id_column_name, text_column_name, *(columns or [])})
        self.filters_json = filters_json
        self.k = k
        self.docs_id_column_name = docs_id_column_name
        self.docs_uri_column_name = docs_uri_column_name
        self.text_column_name = text_column_name
        self.use_with_databricks_agent_framework = use_with_databricks_agent_framework
        if self.use_with_databricks_agent_framework:
            try:
                import mlflow

                mlflow.models.set_retriever_schema(primary_key="doc_id", text_column="page_content", doc_uri="doc_uri")
            except ImportError as err:
                raise ImportError(
                    "To use the `DatabricksRM` retriever module with the Databricks Mosaic Agent Framework, you must install the mlflow Python library. Please install mlflow via `pip install mlflow`."
                ) from err

    async def __call__(
        self, query: str | list[float], query_type: str = "ANN", filters_json: str | None = None
    ) -> Prediction | list[dict[str, Any]]:
        return await self.aforward(query=query, query_type=query_type, filters_json=filters_json)

    async def aforward(
        self, query: str | list[float], query_type: str = "ANN", filters_json: str | None = None
    ) -> Prediction | list[dict[str, Any]]:
        return await asyncio.to_thread(self._query, query=query, query_type=query_type, filters_json=filters_json)

    def _extract_doc_ids(self, item: dict[str, Any]) -> str:
        if self.docs_id_column_name == "metadata":
            docs_dict = json.loads(item["metadata"])
            return docs_dict["document_id"]
        return item[self.docs_id_column_name]

    def _get_extra_columns(self, item: dict[str, Any]) -> dict[str, Any]:
        extra_columns = {
            k: v
            for k, v in item.items()
            if k not in [self.docs_id_column_name, self.text_column_name, self.docs_uri_column_name]
        }
        if self.docs_id_column_name == "metadata":
            extra_columns = {
                **extra_columns,
                "metadata": {k: v for k, v in json.loads(item["metadata"]).items() if k != "document_id"},
            }
        return extra_columns

    def _query(
        self, query: str | list[float], query_type: str = "ANN", filters_json: str | None = None
    ) -> Prediction | list[dict[str, Any]]:
        if isinstance(query, str):
            query_text = query
            query_vector = None
        elif isinstance(query, list):
            query_vector = query
            query_text = None
        else:
            raise TypeError("Query must be a string or a list of floats.")
        if _databricks_sdk_installed:
            results = self._query_via_databricks_sdk(
                index_name=self.databricks_index_name,
                k=self.k,
                columns=self.columns,
                query_type=query_type,
                query_text=query_text,
                query_vector=query_vector,
                databricks_token=self.databricks_token,
                databricks_endpoint=self.databricks_endpoint,
                databricks_client_id=self.databricks_client_id,
                databricks_client_secret=self.databricks_client_secret,
                filters_json=filters_json or self.filters_json,
            )
        else:
            if self.databricks_token is None or self.databricks_endpoint is None:
                raise DatabricksRMError("Databricks token and endpoint are required for request-based querying.")
            results = self._query_via_requests(
                index_name=self.databricks_index_name,
                k=self.k,
                columns=self.columns,
                databricks_token=self.databricks_token,
                databricks_endpoint=self.databricks_endpoint,
                query_type=query_type,
                query_text=query_text,
                query_vector=query_vector,
                filters_json=filters_json or self.filters_json,
            )
        col_names = [column["name"] for column in results["manifest"]["columns"]]
        if self.docs_id_column_name not in col_names:
            raise DatabricksRMError(
                f"docs_id_column_name: '{self.docs_id_column_name}' is not in the index columns: \n {col_names}"
            )
        if self.text_column_name not in col_names:
            raise DatabricksRMError(
                f"text_column_name: '{self.text_column_name}' is not in the index columns: \n {col_names}"
            )
        items = []
        if "data_array" in results["result"]:
            for data_row in results["result"]["data_array"]:
                item = dict(zip(col_names, data_row, strict=False))
                items += [item]
        sorted_docs = sorted(items, key=lambda x: x["score"], reverse=True)[: self.k]
        if self.use_with_databricks_agent_framework:
            return [
                Document(
                    page_content=doc[self.text_column_name],
                    metadata={
                        "doc_id": self._extract_doc_ids(doc),
                        "doc_uri": doc[self.docs_uri_column_name] if self.docs_uri_column_name else None,
                    }
                    | self._get_extra_columns(doc),
                    type="Document",
                ).to_dict()
                for doc in sorted_docs
            ]
        return Prediction(
            docs=[doc[self.text_column_name] for doc in sorted_docs],
            doc_ids=[self._extract_doc_ids(doc) for doc in sorted_docs],
            doc_uris=[doc[self.docs_uri_column_name] for doc in sorted_docs] if self.docs_uri_column_name else None,
            extra_columns=[self._get_extra_columns(item) for item in sorted_docs],
        )

    @staticmethod
    def _query_via_databricks_sdk(
        index_name: str,
        k: int,
        columns: list[str],
        query_type: str,
        query_text: str | None,
        query_vector: list[float] | None,
        databricks_token: str | None,
        databricks_endpoint: str | None,
        databricks_client_id: str | None,
        databricks_client_secret: str | None,
        filters_json: str | None,
    ) -> dict[str, Any]:
        from databricks.sdk import WorkspaceClient

        if (query_text, query_vector).count(None) != 1:
            raise ValueError("Exactly one of query_text or query_vector must be specified.")
        if databricks_client_secret and databricks_client_id:
            databricks_client = WorkspaceClient(client_id=databricks_client_id, client_secret=databricks_client_secret)
        else:
            databricks_client = WorkspaceClient(host=databricks_endpoint, token=databricks_token)
        return databricks_client.vector_search_indexes.query_index(
            index_name=index_name,
            query_type=query_type,
            query_text=query_text,
            query_vector=query_vector,
            columns=columns,
            filters_json=filters_json,
            num_results=k,
        ).as_dict()

    @staticmethod
    def _query_via_requests(
        index_name: str,
        k: int,
        columns: list[str],
        databricks_token: str,
        databricks_endpoint: str,
        query_type: str,
        query_text: str | None,
        query_vector: list[float] | None,
        filters_json: str | None,
    ) -> dict[str, Any]:
        import requests

        if (query_text, query_vector).count(None) != 1:
            raise ValueError("Exactly one of query_text or query_vector must be specified.")
        headers = {"Authorization": f"Bearer {databricks_token}", "Content-Type": "application/json"}
        payload: dict[str, object] = {"columns": columns, "num_results": k, "query_type": query_type}
        if filters_json is not None:
            payload["filters_json"] = filters_json
        if query_text is not None:
            payload["query_text"] = query_text
        elif query_vector is not None:
            payload["query_vector"] = query_vector
        response = requests.post(
            f"{databricks_endpoint}/api/2.0/vector-search/indexes/{index_name}/query",
            json=payload,
            headers=headers,
            timeout=60,
        )
        results = response.json()
        if "error_code" in results:
            raise DatabricksRMError(f"ERROR: {results['error_code']} -- {results['message']}")
        return results
