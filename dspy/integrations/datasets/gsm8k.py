from __future__ import annotations

import random
from typing import TYPE_CHECKING, ClassVar

import tqdm

from dspy._internal.lazy_import import import_optional
from dspy.datasets.dataset import Dataset

if TYPE_CHECKING:
    from dspy.evaluate.metric_contract import OptimizerMetric

__all__ = ["GSM8K"]


class GSM8K(Dataset):
    default_metric: ClassVar[OptimizerMetric]
    default_input_keys: ClassVar[tuple[str, ...]] = ("question",)

    def __init__(
        self,
        train_seed: int = 0,
        train_size: int = 200,
        dev_seed: int = 0,
        dev_size: int = 300,
        test_seed: int = 0,
        test_size: int | None = None,
        input_keys: list[str] | None = None,
    ) -> None:
        super().__init__(
            train_seed=train_seed,
            train_size=train_size,
            dev_seed=dev_seed,
            dev_size=dev_size,
            test_seed=test_seed,
            test_size=test_size,
            input_keys=input_keys or list(self.default_input_keys),
        )
        datasets = import_optional("datasets", extra="datasets", feature="GSM8K")
        load_dataset = datasets.load_dataset
        dataset = load_dataset("gsm8k", "main")
        hf_official_train = dataset["train"]
        hf_official_test = dataset["test"]
        official_train = []
        official_test = []
        for example in tqdm.tqdm(hf_official_train):
            question = example["question"]
            answer = example["answer"].strip().split()
            if answer[-2] != "####":
                raise ValueError("GSM8K answer is missing the #### delimiter.")
            gold_reasoning = " ".join(answer[:-2])
            answer = str(int(answer[-1].replace(",", "")))
            official_train.append({"question": question, "gold_reasoning": gold_reasoning, "answer": answer})
        for example in tqdm.tqdm(hf_official_test):
            question = example["question"]
            answer = example["answer"].strip().split()
            if answer[-2] != "####":
                raise ValueError("GSM8K answer is missing the #### delimiter.")
            gold_reasoning = " ".join(answer[:-2])
            answer = str(int(answer[-1].replace(",", "")))
            official_test.append({"question": question, "gold_reasoning": gold_reasoning, "answer": answer})
        rng = random.Random(0)
        rng.shuffle(official_train)
        self._train = official_train[:train_size]
        self._dev = official_train[train_size : train_size + dev_size]
        rng = random.Random(0)
        rng.shuffle(official_test)
        self._test = official_test
