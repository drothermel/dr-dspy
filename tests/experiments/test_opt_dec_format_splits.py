from __future__ import annotations

import json
from typing import TYPE_CHECKING

from dspy.experiments.opt_dec_format.splits import build_split_from_source

if TYPE_CHECKING:
    from pathlib import Path


def test_build_split_excludes_perfect_worst_tasks(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    sample_stats = {
        f"HumanEval/{index}": {
            "all": {
                "average_perf": 1.0 if index == 0 else index / 100,
                "n_average": 1,
                "perf_variance": 0.0,
            }
        }
        for index in range(90)
    }
    source.write_text(
        json.dumps(
            {
                "metadata": {"model_name": "demo"},
                "sample_stats_by_id": sample_stats,
                "selection_rank_by_id": {task_id: index for index, task_id in enumerate(sample_stats, start=1)},
                "worst_sample_ids_by_n": {
                    "100": list(sample_stats),
                },
            }
        ),
        encoding="utf-8",
    )

    split = build_split_from_source(source, rng_seed=7)

    assert "HumanEval/0" in split.excluded_perfect_task_ids
    assert "HumanEval/0" not in split.task_ids_by_split["train"]
    assert len(split.task_ids_by_split["train"]) == 15
    assert len(split.task_ids_by_split["dev"]) == 15
    assert len(split.task_ids_by_split["test"]) == 30
