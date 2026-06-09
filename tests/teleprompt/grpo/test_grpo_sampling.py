import random
from collections import Counter
from typing import Any, cast

from dspy.teleprompt.grpo.sampling import select_training_sample
from dspy.teleprompt.grpo.session import GRPOCompileSession


def test_grpo_dataset_shuffler():
    dataset = [1, 2, 3]
    session = GRPOCompileSession()
    rng = random.Random(0)
    trainset_instances = []
    for i in range(4):
        trainset_instances.append(
            select_training_sample(
                session,
                original_trainset=cast("Any", dataset),
                train_step_idx=i,
                num_dspy_examples_per_grpo_step=3,
                rng=rng,
            )
        )
        assert len(trainset_instances[-1]) == 3
        assert set(trainset_instances[-1]) == set(dataset)


def test_grpo_dataset_shuffler_with_num_ex_per_step_less_dataset():
    dataset = [1, 2, 3]
    session = GRPOCompileSession()
    rng = random.Random(0)
    trainset_instances = []
    for i in range(15):
        trainset_instances.append(
            select_training_sample(
                session,
                original_trainset=cast("Any", dataset),
                train_step_idx=i,
                num_dspy_examples_per_grpo_step=2,
                rng=rng,
            )
        )
        assert len(trainset_instances[-1]) == 2
    counter = Counter()
    for instance in trainset_instances:
        counter.update(instance)
    assert len(counter) == 3
    for i in counter:
        assert counter[i] == 10


def test_grpo_dataset_shuffler_with_num_ex_per_step_greater_dataset():
    dataset = [1, 2, 3]
    session = GRPOCompileSession()
    rng = random.Random(0)
    trainset_instances = []
    for i in range(6):
        trainset_instances.append(
            select_training_sample(
                session,
                original_trainset=cast("Any", dataset),
                train_step_idx=i,
                num_dspy_examples_per_grpo_step=5,
                rng=rng,
            )
        )
        assert len(trainset_instances[-1]) == 5
    counter = Counter()
    for instance in trainset_instances:
        counter.update(instance)
    assert len(counter) == 3
    for i in counter:
        assert counter[i] == 10
