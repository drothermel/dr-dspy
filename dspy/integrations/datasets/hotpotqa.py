import random

from dspy.datasets.dataset import Dataset
from dspy.integrations.datasets.import_ import import_datasets


class HotPotQA(Dataset):
    def __init__(
        self,
        train_seed: int = 0,
        train_size: int | None = None,
        dev_seed: int = 0,
        dev_size: int | None = None,
        test_seed: int = 0,
        test_size: int | None = None,
        input_keys: list[str] | None = None,
        only_hard_examples: bool = True,
        keep_details: bool | str = "dev_titles",
        unofficial_dev: bool = True,
    ) -> None:
        super().__init__(
            train_seed=train_seed,
            train_size=train_size,
            dev_seed=dev_seed,
            dev_size=dev_size,
            test_seed=test_seed,
            test_size=test_size,
            input_keys=input_keys,
        )
        if not only_hard_examples:
            raise ValueError(
                "Care must be taken when adding support for easy examples.Dev must be all hard to match official dev, but training can be flexible."
            )
        load_dataset = import_datasets(feature="HotPotQA").load_dataset
        hf_official_train = load_dataset("hotpot_qa", "fullwiki", split="train")
        hf_official_dev = load_dataset("hotpot_qa", "fullwiki", split="validation")
        official_train = []
        for raw_example in hf_official_train:
            if raw_example["level"] == "hard":
                if keep_details is True:
                    keys = ["id", "question", "answer", "type", "supporting_facts", "context"]
                elif keep_details == "dev_titles":
                    keys = ["question", "answer", "supporting_facts"]
                else:
                    keys = ["question", "answer"]
                example = {k: raw_example[k] for k in keys}
                if "supporting_facts" in example:
                    example["gold_titles"] = set(example["supporting_facts"]["title"])
                    del example["supporting_facts"]
                official_train.append(example)
        rng = random.Random(self.train_seed)
        rng.shuffle(official_train)
        train_split = official_train[: len(official_train) * 75 // 100]
        self._train = train_split
        if unofficial_dev:
            dev_split = official_train[len(official_train) * 75 // 100 :]
            self._dev = dev_split
        else:
            self._dev = None
        for example in train_split:
            if keep_details == "dev_titles":
                del example["gold_titles"]
        test = []
        for raw_example in hf_official_dev:
            if raw_example["level"] != "hard":
                raise ValueError("HotPotQA validation split must contain hard examples only.")
            example = {k: raw_example[k] for k in ["id", "question", "answer", "type", "supporting_facts"]}
            if "supporting_facts" in example:
                example["gold_titles"] = set(example["supporting_facts"]["title"])
                del example["supporting_facts"]
            test.append(example)
        self._test = test


if __name__ == "__main__":
    dataset = HotPotQA(train_seed=1, train_size=16, dev_seed=2023, dev_size=200 * 5, test_size=0)
