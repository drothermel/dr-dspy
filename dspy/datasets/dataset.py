from __future__ import annotations

import random
import uuid
from typing import TYPE_CHECKING, cast

from dspy.primitives.example import Example
from dspy.utils.dotdict import dotdict

if TYPE_CHECKING:
    from collections.abc import Iterable


class Dataset:
    def __init__(
        self,
        train_seed: int = 0,
        train_size: int | None = None,
        eval_seed: int = 0,
        dev_size: int | None = None,
        test_size: int | None = None,
        input_keys: list[str] | None = None,
    ) -> None:
        self.train_size = train_size
        self.train_seed = train_seed
        self.dev_size = dev_size
        self.dev_seed = eval_seed
        self.test_size = test_size
        self.test_seed = eval_seed
        self.input_keys = input_keys or []
        self._train: list[dict[str, object]] | list[dict[str, str]] | list[Example] | None = None
        self._dev: list[dict[str, object]] | list[dict[str, str]] | list[Example] | None = None
        self._test: list[dict[str, object]] | list[dict[str, str]] | list[Example] | None = None
        self._train_cache: list[Example] | None = None
        self._dev_cache: list[Example] | None = None
        self._test_cache: list[Example] | None = None
        self.do_shuffle = True
        self.name = self.__class__.__name__

    def reset_seeds(
        self,
        train_seed: int | None = None,
        train_size: int | None = None,
        eval_seed: int | None = None,
        dev_size: int | None = None,
        test_size: int | None = None,
    ) -> None:
        self.train_size = train_size or self.train_size
        self.train_seed = train_seed or self.train_seed
        self.dev_size = dev_size or self.dev_size
        self.dev_seed = eval_seed or self.dev_seed
        self.test_size = test_size or self.test_size
        self.test_seed = eval_seed or self.test_seed
        self._train_cache = None
        self._dev_cache = None
        self._test_cache = None

    @property
    def train(self) -> list[Example]:
        if self._train_cache is None:
            if self._train is None:
                raise ValueError("Train split has not been initialized.")
            self._train_cache = self._shuffle_and_sample(
                split="train",
                data=cast("Iterable[dict[str, object]]", self._train),
                size=self.train_size,
                seed=self.train_seed,
            )
        return self._train_cache

    @property
    def dev(self) -> list[Example]:
        if self._dev_cache is None:
            if self._dev is None:
                raise ValueError("Dev split has not been initialized.")
            self._dev_cache = self._shuffle_and_sample(
                split="dev", data=cast("Iterable[dict[str, object]]", self._dev), size=self.dev_size, seed=self.dev_seed
            )
        return self._dev_cache

    @property
    def test(self) -> list[Example]:
        if self._test_cache is None:
            if self._test is None:
                raise ValueError("Test split has not been initialized.")
            self._test_cache = self._shuffle_and_sample(
                split="test",
                data=cast("Iterable[dict[str, object]]", self._test),
                size=self.test_size,
                seed=self.test_seed,
            )
        return self._test_cache

    def _shuffle_and_sample(
        self, split: str, data: Iterable[dict[str, object]], size: int | None, seed: int = 0
    ) -> list[Example]:
        data_list = list(data)
        base_rng = random.Random(seed)
        if self.do_shuffle:
            base_rng.shuffle(data_list)
        data_list = data_list[:size]
        output: list[Example] = []
        for example in data_list:
            example_obj = Example(**example, dspy_uuid=str(uuid.uuid4()), dspy_split=split)
            if self.input_keys:
                example_obj = example_obj.with_inputs(*self.input_keys)
            output.append(example_obj)
        return output

    @classmethod
    def prepare_by_seed(
        cls,
        train_seeds: list[int] | None = None,
        train_size: int = 16,
        dev_size: int = 1000,
        divide_eval_per_seed: bool = True,
        eval_seed: int = 2023,
        **kwargs: object,
    ) -> dotdict:
        train_seeds = train_seeds or [1, 2, 3, 4, 5]
        data_args = dotdict(train_size=train_size, eval_seed=eval_seed, dev_size=dev_size, test_size=0, **kwargs)
        dataset = cls(**data_args)
        eval_set = dataset.dev
        eval_sets: list[list[Example]] = []
        train_sets: list[list[Example]] = []
        examples_per_seed = dev_size // len(train_seeds) if divide_eval_per_seed else dev_size
        eval_offset = 0
        for train_seed in train_seeds:
            data_args.train_seed = train_seed
            dataset.reset_seeds(**data_args)
            eval_sets.append(eval_set[eval_offset : eval_offset + examples_per_seed])
            train_sets.append(dataset.train)
            if len(eval_sets[-1]) != examples_per_seed:
                raise ValueError(len(eval_sets[-1]))
            if len(train_sets[-1]) != train_size:
                raise ValueError(len(train_sets[-1]))
            if divide_eval_per_seed:
                eval_offset += examples_per_seed
        return dotdict(train_sets=train_sets, eval_sets=eval_sets)
