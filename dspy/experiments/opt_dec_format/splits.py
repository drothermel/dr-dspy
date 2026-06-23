"""Dataset split construction for opt-dec-format experiments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SPLIT_NAME = "humanevalplus_gpt5nano_nonperfect_stratified_v0"
DEFAULT_RNG_SEED = 20260623


class HumanEvalSplit(BaseModel):
    """Persisted model-conditioned HumanEval+ split."""

    model_config = ConfigDict(extra="forbid")

    split_name: str = SPLIT_NAME
    rng_seed: int
    source_path: str
    source_metadata: dict[str, Any]
    task_ids_by_split: dict[str, list[str]]
    task_ids_by_difficulty_bucket: dict[str, list[str]]
    average_performance_by_task_id: dict[str, float]
    excluded_perfect_task_ids: list[str] = Field(default_factory=list)


def build_split_from_source(
    source_path: Path | str,
    *,
    rng_seed: int = DEFAULT_RNG_SEED,
) -> HumanEvalSplit:
    """Build the fixed nonperfect stratified split from source rankings."""
    path = Path(source_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    sample_stats = raw["sample_stats_by_id"]
    worst_ids = list(raw["worst_sample_ids_by_n"]["100"])
    average_by_id = {task_id: float(stats["all"]["average_perf"]) for task_id, stats in sample_stats.items()}
    excluded = sorted(task_id for task_id in worst_ids if average_by_id.get(task_id, 1.0) >= 1.0)
    nonperfect_worst = [task_id for task_id in worst_ids if task_id not in excluded]
    worst_25 = nonperfect_worst[:25]
    worst_50_not_25 = nonperfect_worst[25:50]
    nonperfect_not_50 = [
        task_id
        for task_id, average in average_by_id.items()
        if average < 1.0 and task_id not in set(nonperfect_worst[:50])
    ]

    rng = random.Random(rng_seed)
    train = _sample_sorted(rng, worst_25, 15)
    dev = _sample_sorted(rng, worst_50_not_25, 15)
    test = _sample_sorted(rng, nonperfect_not_50, 30)
    return HumanEvalSplit(
        rng_seed=rng_seed,
        source_path=str(path),
        source_metadata=dict(raw["metadata"]),
        task_ids_by_split={
            "train": train,
            "dev": dev,
            "test": test,
        },
        task_ids_by_difficulty_bucket={
            "worst_25": worst_25,
            "worst_50_not_25": worst_50_not_25,
            "nonperfect_not_50": sorted(nonperfect_not_50),
        },
        average_performance_by_task_id={
            task_id: average_by_id[task_id] for task_id in sorted({*train, *dev, *test, *excluded})
        },
        excluded_perfect_task_ids=excluded,
    )


def save_split(split: HumanEvalSplit, path: Path | str) -> Path:
    """Persist a split JSON artifact."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(split.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out


def _sample_sorted(
    rng: random.Random,
    population: list[str],
    count: int,
) -> list[str]:
    if len(population) <= count:
        return sorted(population)
    return sorted(rng.sample(population, count))
