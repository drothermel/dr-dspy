import random
import re
from typing import TYPE_CHECKING, ClassVar

from dspy._internal.lazy_import import import_optional
from dspy.datasets.dataset import Dataset

if TYPE_CHECKING:
    from dspy.evaluate.metric_contract import OptimizerMetric

__all__ = ["MATH", "extract_answer"]


class MATH(Dataset):
    default_metric: ClassVar["OptimizerMetric"]
    default_input_keys: ClassVar[tuple[str, ...]] = ("question",)

    def __init__(
        self,
        subset: str,
        train_seed: int = 0,
        train_size: int | None = None,
        dev_seed: int = 0,
        dev_size: int | None = None,
        test_seed: int = 0,
        test_size: int | None = None,
        input_keys: list[str] | None = None,
    ) -> None:
        load_dataset = import_optional("datasets", extra="datasets", feature="MATH").load_dataset
        ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", subset)
        records = [
            {
                "question": example["problem"],
                "reasoning": example["solution"],
                "answer": extract_answer(example["solution"]),
            }
            for example in ds["test"]
        ]
        size = min(350, len(records) // 3)
        random.Random(0).shuffle(records)
        super().__init__(
            train_seed=train_seed,
            train_size=train_size or size,
            dev_seed=dev_seed,
            dev_size=dev_size or size,
            test_seed=test_seed,
            test_size=test_size,
            input_keys=input_keys or list(self.default_input_keys),
        )
        self._train = records[:size]
        self._dev = records[size : 2 * size]
        self._test = records[2 * size :]


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
