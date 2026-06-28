from __future__ import annotations

from dr_dspy import eval_set

_TEST_BODY = (
    "def check(candidate):\n"
    "    inputs = [[1, 2], [0, 0]]\n"
    "    results = [3, 0]\n"
    "    for inp, expected in zip(inputs, results):\n"
    "        assert candidate(*inp) == expected\n"
)


def _rows(count: int) -> list[dict[str, str]]:
    return [
        {
            "task_id": f"T{index}",
            "prompt": f"def f{index}(a, b):\n",
            "canonical_solution": f"    return a + b + {index}\n",
            "entry_point": f"f{index}",
            "test": _TEST_BODY,
        }
        for index in range(count)
    ]


def test_split_is_disjoint_and_sized() -> None:
    split = eval_set.build_eval_split_from_rows(
        _rows(10), seed=0, train=2, val=3, test=4, repetitions=2
    )
    assert len(split.train_ids) == 2
    assert len(split.val_ids) == 3
    assert len(split.test_ids) == 4
    all_ids = split.all_ids()
    assert len(all_ids) == len(set(all_ids))  # disjoint
    assert set(all_ids) <= {f"T{i}" for i in range(10)}
    assert split.repetitions == 2


def test_split_is_deterministic_for_seed() -> None:
    rows = _rows(10)
    a = eval_set.build_eval_split_from_rows(
        rows, seed=7, train=2, val=3, test=3, repetitions=1
    )
    b = eval_set.build_eval_split_from_rows(
        rows, seed=7, train=2, val=3, test=3, repetitions=1
    )
    assert a == b


def test_split_changes_with_seed() -> None:
    rows = _rows(20)
    a = eval_set.build_eval_split_from_rows(
        rows, seed=1, train=4, val=4, test=4, repetitions=1
    )
    b = eval_set.build_eval_split_from_rows(
        rows, seed=2, train=4, val=4, test=4, repetitions=1
    )
    assert a.all_ids() != b.all_ids()


def test_split_requires_enough_tasks() -> None:
    try:
        eval_set.build_eval_split_from_rows(
            _rows(3), seed=0, train=2, val=2, test=2, repetitions=1
        )
    except ValueError as error:
        assert "only" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")
