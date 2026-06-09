import importlib
import random
import re
from typing import Protocol, cast

from dspy.integrations.datasets.import_ import import_datasets
from dspy.primitives.example import Example


class HasAnswer(Protocol):
    answer: object


class MATH:
    def __init__(self, subset: str) -> None:
        load_dataset = import_datasets(feature="MATH").load_dataset

        ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", subset)
        dataset = [
            Example.from_record(
                {
                    "question": example["problem"],
                    "reasoning": example["solution"],
                    "answer": extract_answer(example["solution"]),
                },
                input_keys=("question",),
            )
            for example in ds["test"]
        ]
        size = min(350, len(dataset) // 3)
        random.Random(0).shuffle(dataset)
        self.train, self.dev, self.test = (dataset[:size], dataset[size : 2 * size], dataset[2 * size :])

    def metric(self, example: HasAnswer, pred: HasAnswer, _trace: object | None = None) -> bool:
        try:
            math_equivalence = cast("MathEquivalenceModule", importlib.import_module("math_equivalence"))
        except ImportError as err:
            raise ImportError("MATH's metric requires `pip install git+https://github.com/hendrycks/math.git`") from err
        return math_equivalence.is_equiv(example.answer, pred.answer)


def extract_answer(s: str) -> str | None:
    start = s.find("\\boxed{")
    if start == -1:
        return None
    idx = start + len("\\boxed{")
    brace_level = 1
    answer = ""
    while idx < len(s) and brace_level > 0:
        c = s[idx]
        if c == "{":
            brace_level += 1
        elif c == "}":
            brace_level -= 1
            if brace_level == 0:
                break
        answer += c
        idx += 1
    answer = re.sub("\\\\text\\{[^}]*\\}", "", answer)
    answer = re.sub("\\\\!", "", answer)
    return answer.strip()


class MathEquivalenceModule(Protocol):
    def is_equiv(self, left: object, right: object) -> bool: ...
