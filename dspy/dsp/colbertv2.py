import importlib
from typing import Any

import requests

from dspy.utils.dotdict import dotdict


class ColBERTv2:
    def __init__(self, url: str = "http://0.0.0.0", port: str | int | None = None, post_requests: bool = False) -> None:
        self.post_requests = post_requests
        self.url = f"{url}:{port}" if port else url

    def __call__(self, query: str, k: int = 10, simplify: bool = False) -> list[str] | list[dotdict]:
        if self.post_requests:
            topk: list[dict[str, Any]] = colbertv2_post_request(url=self.url, query=query, k=k)
        else:
            topk: list[dict[str, Any]] = colbertv2_get_request(url=self.url, query=query, k=k)
        if simplify:
            return [psg["long_text"] for psg in topk]
        return [dotdict(psg) for psg in topk]


def colbertv2_get_request(url: str, query: str, k: int):
    assert k <= 100, "Only k <= 100 is supported for the hosted ColBERTv2 server at the moment."
    payload = {"query": query, "k": k}
    res = requests.get(url, params=payload, timeout=10)
    res.raise_for_status()
    res_json = res.json()
    if res_json.get("error"):
        error_message = res_json.get("message", "Unknown error")
        raise ValueError(f"ColBERTv2 server returned an error: {error_message}")
    if "topk" not in res_json:
        raise ValueError(f"ColBERTv2 server returned an unexpected response: {res_json}")
    topk = res_json["topk"][:k]
    topk = [{**d, "long_text": d["text"]} for d in topk]
    return topk[:k]


def colbertv2_post_request(url: str, query: str, k: int):
    headers = {"Content-Type": "application/json; charset=utf-8"}
    payload = {"query": query, "k": k}
    res = requests.post(url, json=payload, headers=headers, timeout=10)
    res.raise_for_status()
    res_json = res.json()
    if res_json.get("error"):
        error_message = res_json.get("message", "Unknown error")
        raise ValueError(f"ColBERTv2 server returned an error: {error_message}")
    if "topk" not in res_json:
        raise ValueError(f"ColBERTv2 server returned an unexpected response: {res_json}")
    return res_json["topk"][:k]


class ColBERTv2RetrieverLocal:
    def __init__(self, passages: list[str], colbert_config: Any = None, load_only: bool = False) -> None:
        assert colbert_config is not None, (
            "Please pass a valid colbert_config, which you can import from colbert.infra.config import ColBERTConfig and modify it"
        )
        self.colbert_config = colbert_config
        assert self.colbert_config.checkpoint is not None, (
            "Please pass a valid checkpoint like colbert-ir/colbertv2.0, which you can modify in the ColBERTConfig with attribute name checkpoint"
        )
        self.passages = passages
        assert self.colbert_config.index_name is not None, (
            "Please pass a valid index_name, which you can modify in the ColBERTConfig with attribute name index_name"
        )
        self.passages = passages
        if not load_only:
            self.build_index()
        self.searcher = self.get_index()

    def build_index(self) -> None:
        colbert = importlib.import_module("colbert")
        Indexer = colbert.Indexer
        infra = importlib.import_module("colbert.infra")
        Run = infra.Run
        RunConfig = infra.RunConfig
        with Run().context(RunConfig(nranks=self.colbert_config.nranks, experiment=self.colbert_config.experiment)):
            indexer = Indexer(checkpoint=self.colbert_config.checkpoint, config=self.colbert_config)
            indexer.index(name=self.colbert_config.index_name, collection=self.passages, overwrite=True)

    def get_index(self):
        colbert = importlib.import_module("colbert")
        Searcher = colbert.Searcher
        infra = importlib.import_module("colbert.infra")
        Run = infra.Run
        RunConfig = infra.RunConfig
        with Run().context(RunConfig(experiment=self.colbert_config.experiment)):
            return Searcher(index=self.colbert_config.index_name, collection=self.passages)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, query: str, k: int = 7, **kwargs):
        torch = importlib.import_module("torch")
        filtered_pids: list[int] = kwargs.get("filtered_pids") or []
        if filtered_pids:
            assert isinstance(filtered_pids, list) and all(isinstance(pid, int) for pid in filtered_pids), (
                "The filtered pids should be a list of integers"
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            searcher_results = self.searcher.search(
                query,
                k=k,
                filter_fn=lambda pids: torch.tensor(
                    [pid for pid in pids if pid in filtered_pids], dtype=torch.int32
                ).to(device),
            )
        else:
            searcher_results = self.searcher.search(query, k=k)
        results = []
        for pid, _rank, score in zip(*searcher_results, strict=False):
            results.append(dotdict({"long_text": self.searcher.collection[pid], "score": score, "pid": pid}))
        return results


class ColBERTv2RerankerLocal:
    def __init__(self, colbert_config: Any = None, checkpoint: str = "bert-base-uncased") -> None:
        self.colbert_config = colbert_config
        self.checkpoint = checkpoint
        self.colbert_config.checkpoint = checkpoint

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, query: str, passages: list[str] | None = None):
        passages = passages or []
        assert len(passages) > 0, "Passages should not be empty"
        import numpy as np

        colbert = importlib.import_module("colbert")
        ColBERT = colbert.modeling.colbert.ColBERT
        DocTokenizer = colbert.modeling.tokenization.doc_tokenization.DocTokenizer
        QueryTokenizer = colbert.modeling.tokenization.query_tokenization.QueryTokenizer
        self.colbert_config.nway = len(passages)
        query_tokenizer = QueryTokenizer(self.colbert_config, verbose=1)
        doc_tokenizer = DocTokenizer(self.colbert_config)
        query_ids, query_masks = query_tokenizer.tensorize([query])
        doc_ids, doc_masks = doc_tokenizer.tensorize(passages)
        col = ColBERT(self.checkpoint, self.colbert_config)
        q = col.query(query_ids, query_masks)
        doc_ids, doc_masks = col.doc(doc_ids, doc_masks, keep_dims="return_mask")
        q_duplicated = q.repeat_interleave(len(passages), dim=0).contiguous()
        tensor_scores = col.score(q_duplicated, doc_ids, doc_masks)
        return np.array([score.cpu().detach().numpy().tolist() for score in tensor_scores])
