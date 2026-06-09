from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dspy.predict.knn import KNN  # noqa: TC001 — held for demo retrieval at runtime
from dspy.primitives import Module

if TYPE_CHECKING:
    from dspy.runtime.call_options import ModuleCallOptions
    from dspy.runtime.run_context import RunContext


class KNNFewShotModule(Module):
    """Wraps a program to inject KNN-retrieved demos before each forward pass."""

    def __init__(self, inner: Module, *, knn: KNN, k: int) -> None:
        super().__init__()
        self._inner = inner
        self._knn = knn
        self._k = k

    def predictors(self) -> list[Any]:
        return self._inner.predictors()

    def named_predictors(self) -> list[tuple[str, Any]]:
        return self._inner.named_predictors()

    def set_lm(self, lm: Any) -> None:
        self._inner.set_lm(lm)

    def optional_lm(self) -> Any:
        return self._inner.optional_lm()

    def deepcopy(self) -> KNNFewShotModule:
        return KNNFewShotModule(self._inner.deepcopy(), knn=self._knn, k=self._k)

    async def _aforward_impl(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **kwargs: Any,
    ) -> Any:
        knn_trainset = await self._knn(inputs=kwargs)
        for predictor in self._inner.predictors():
            predictor.demos = knn_trainset[: self._k]
        return await self._inner.aforward(run=run, options=options, **kwargs)
