from __future__ import annotations

from typing import Any, Callable

from dspy.clients._litellm import get_litellm
from dspy.utils.lazy_import import require

np = require("numpy")


def _get_litellm():
    return get_litellm(feature="dspy.clients.embedding.Embedder")


class Embedder:
    def __init__(self, model: str | Callable, batch_size: int = 200, **kwargs: dict[str, Any]) -> None:
        self.model = model
        self.batch_size = batch_size
        self.default_kwargs = kwargs

    def _preprocess(self, inputs, batch_size=None, **kwargs):
        if isinstance(inputs, str):
            is_single_input = True
            inputs = [inputs]
        else:
            is_single_input = False
        if not all(isinstance(inp, str) for inp in inputs):
            raise ValueError("All inputs must be strings.")
        batch_size = batch_size or self.batch_size
        merged_kwargs = self.default_kwargs.copy()
        merged_kwargs.update(kwargs)
        input_batches = [inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)]
        return (input_batches, merged_kwargs, is_single_input)

    def _postprocess(self, embeddings_list, is_single_input):
        embeddings = np.array(embeddings_list, dtype=np.float32)
        if is_single_input:
            return embeddings[0]
        return np.array(embeddings, dtype=np.float32)

    async def __call__(
        self, inputs: str | list[str], batch_size: int | None = None, **kwargs: dict[str, Any]
    ) -> np.ndarray:
        input_batches, kwargs, is_single_input = self._preprocess(inputs=inputs, batch_size=batch_size, **kwargs)
        embeddings_list = []
        for batch in input_batches:
            embeddings_list.extend(await _acompute_embeddings(model=self.model, batch_inputs=batch, **kwargs))
        return self._postprocess(embeddings_list=embeddings_list, is_single_input=is_single_input)

    async def acall(self, inputs, batch_size=None, **kwargs):
        return await self.__call__(inputs, batch_size=batch_size, **kwargs)


async def _acompute_embeddings(model, batch_inputs, **kwargs):
    if isinstance(model, str):
        embedding_response = await _get_litellm().aembedding(model=model, input=batch_inputs, caching=False, **kwargs)
        return [data["embedding"] for data in embedding_response.data]
    if callable(model):
        return model(batch_inputs, **kwargs)
    raise ValueError(
        f"`model` in `dspy.clients.embedding.Embedder` must be a string or a callable, but got {type(model)}."
    )
