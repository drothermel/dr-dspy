import importlib
import random
import re
from typing import TYPE_CHECKING, Protocol, cast

from dspy.primitives.example import Example

if TYPE_CHECKING:
    from datasets import DatasetDict


class HasAnswer(Protocol):
    answer: object


class MATH:
    def __init__(self, subset: str) -> None:
        from datasets import load_dataset

        ds = cast("DatasetDict", load_dataset("DigitalLearningGmbH/MATH-lighteval", subset))

        # NOTE: Defaults to sub-splitting MATH's 'test' split into train/dev/test, presuming that current
        # LMs are trained on MATH's train. Makes no difference for gpt-4o-mini, but might for other models.

        dataset = [
            Example(
                question=example["problem"], reasoning=example["solution"], answer=extract_answer(example["solution"])
            ).with_inputs("question")
            for example in ds["test"]
        ]

        size = min(350, len(dataset) // 3)
        random.Random(0).shuffle(dataset)
        self.train, self.dev, self.test = dataset[:size], dataset[size : 2 * size], dataset[2 * size :]

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

    answer = re.sub(r"\\text\{[^}]*\}", "", answer)
    answer = re.sub(r"\\!", "", answer)
    return answer.strip()


class MathEquivalenceModule(Protocol):
    def is_equiv(self, left: object, right: object) -> bool: ...


"""
NOTE: MATH's official math_equivalence.is_equiv does not seem to have perfect recall.
Consider its behavior on reference values like `left[\frac{1}{2}, \frac{4}{3}\right]`.
"""
