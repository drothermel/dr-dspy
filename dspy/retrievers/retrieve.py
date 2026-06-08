import random
from collections.abc import Mapping, Sequence

from dspy.dsp.utils.settings import settings
from dspy.predict.parameter import Parameter
from dspy.primitives.prediction import Prediction
from dspy.utils.callback import with_callbacks


def single_query_passage(passages: Sequence[Mapping[str, object]]) -> Prediction:
    passages_dict = {key: [] for key in list(passages[0].keys())}
    for docs in passages:
        for key, value in docs.items():
            passages_dict[key].append(value)
    if "long_text" in passages_dict:
        passages_dict["passages"] = passages_dict.pop("long_text")
    return Prediction(**passages_dict)


class Retrieve(Parameter):
    name = "Search"
    input_variable = "query"
    desc = "takes a search query and returns one or more potentially relevant passages from a corpus"

    def __init__(self, k: int = 3, callbacks: list[object] | None = None) -> None:
        self.stage = random.randbytes(8).hex()
        self.k = k
        self.callbacks = callbacks or []

    def reset(self) -> None:
        pass

    def dump_state(self) -> dict[str, object]:
        state_keys = ["k"]
        return {k: getattr(self, k) for k in state_keys}

    def load_state(self, state: Mapping[str, object]) -> None:
        for name, value in state.items():
            setattr(self, name, value)

    @with_callbacks
    def __call__(self, *args: object, **kwargs: object) -> list[str] | Prediction | list[Prediction]:
        return self.forward(*args, **kwargs)

    def forward(
        self,
        query: str,
        k: int | None = None,
        **kwargs: object,
    ) -> list[str] | Prediction | list[Prediction]:
        k = k if k is not None else self.k


        if not settings.rm:
            raise AssertionError("No RM is loaded.")

        passages = settings.rm(query, k=k, **kwargs)

        from collections.abc import Iterable
        if not isinstance(passages, Iterable):
            # TODO: Normalize retriever return types so single-document results do not need wrapping here.
            passages = [passages]
        passages = [psg.long_text for psg in passages]

        return Prediction(passages=passages)

# TODO: Preserve per-query passage groups when constructing Predictions for batched retrieval.
