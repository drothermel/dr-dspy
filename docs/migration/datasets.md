# Dataset migration guide

## Spine vs integrations

| Layer | Module | Purpose |
| --- | --- | --- |
| Spine | `dspy.datasets.dataset` | `Dataset` base for benchmark splits |
| Spine | `dspy.datasets.rows` | `rows_to_examples` for dict rows → `Example` |
| Integrations | `dspy.integrations.datasets.*` | Vendor loaders (Hugging Face, HotPotQA, GSM8K, MATH, AlfWorld) |

Install optional extras as needed: `pip install dspy[datasets]` for Hugging Face helpers.

## Hugging Face loading

```python
from dspy.integrations.datasets.huggingface import (
    examples_from_csv,
    examples_from_huggingface,
    examples_from_json,
    examples_from_parquet,
)

examples = examples_from_huggingface("squad", split="train", input_keys=("question",))
rows = examples_from_csv("data.csv", input_keys=("text",))
```

## Benchmark datasets

All benchmark loaders subclass spine `Dataset` with lazy `train` / `dev` / `test` properties,
configurable seeds, and `dspy_uuid` / `dspy_split` metadata on access.

Paired metrics live in `dspy.integrations.datasets.metrics` and on `default_metric` where applicable.

```python
from dspy.evaluate.evaluator import Evaluate
from dspy.integrations.datasets.gsm8k import GSM8K
from dspy.integrations.datasets.hotpotqa import HotPotQA
from dspy.integrations.datasets.metrics import gsm8k_metric, hotpotqa_metric

gsm8k = GSM8K(train_seed=0, dev_seed=0)
evaluate = Evaluate(devset=gsm8k.dev, metric=gsm8k_metric)
# or: metric=gsm8k.default_metric

hotpot = HotPotQA(train_seed=0, dev_seed=0, test_seed=0)
evaluate = Evaluate(devset=hotpot.dev, metric=hotpotqa_metric)
```

```python
from dspy.integrations.datasets.math import MATH
from dspy.integrations.datasets.alfworld.alfworld import AlfWorld

math = MATH("algebra", train_seed=0)
alf = AlfWorld(train_seed=0)
alf_train = alf.train
alf_dev = alf.dev
```

## Breaking changes

- `GSM8K`, `MATH`, and `AlfWorld` are `Dataset` subclasses — use lazy `.train` / `.dev` / `.test` with seeds, not eager lists materialized at construction.
- `MATH.metric(...)` removed — use `math_metric(example, pred, trace)` from `dspy.integrations.datasets.metrics`.
- `gsm8k_metric` now uses `(example, pred, trace)` instead of `(gold, pred, trace)` protocols.

## Removed spine utilities

- `dspy.datasets.dataloader.DataLoader` — removed; use `rows_to_examples` or integration loaders.
- `dspy.datasets.colors.Colors` — removed.
- `Dataset.prepare_by_seed` — removed.
- `Dataset.eval_seed` — removed; use separate `dev_seed` and `test_seed` on `Dataset` / integration loaders.
