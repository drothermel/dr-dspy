import random
from typing import Protocol

import tqdm

from dspy.primitives.example import Example


class HasAnswer(Protocol):
    answer: object


class GSM8K:
    def __init__(self) -> None:
        self.do_shuffle = False

        from datasets import load_dataset

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

        rng = random.Random(0)
        rng.shuffle(official_test)

        trainset = official_train[:200]
        devset = official_train[200:500]
        testset = official_test[:]


        trainset = [Example(**x).with_inputs("question") for x in trainset]
        devset = [Example(**x).with_inputs("question") for x in devset]
        testset = [Example(**x).with_inputs("question") for x in testset]

        self.train = trainset
        self.dev = devset
        self.test = testset


def parse_integer_answer(answer: str, only_first_line: bool = True) -> int:
    try:
        if only_first_line:
            answer = answer.strip().split("\n")[0]

        answer = [token for token in answer.split() if any(c.isdigit() for c in token)][-1]
        answer = answer.split(".")[0]
        answer = "".join([c for c in answer if c.isdigit()])
        answer = int(answer)

    except (ValueError, IndexError):
        answer = 0

    return answer


def gsm8k_metric(gold: HasAnswer, pred: HasAnswer, trace: object | None = None) -> bool:
    _ = trace
    return int(parse_integer_answer(str(gold.answer))) == int(parse_integer_answer(str(pred.answer)))
