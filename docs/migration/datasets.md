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

```python
from dspy.integrations.datasets.hotpotqa import HotPotQA
from dspy.integrations.datasets.gsm8k import GSM8K
from dspy.integrations.datasets.math import MATH
from dspy.integrations.datasets.alfworld.alfworld import AlfWorld

hotpot = HotPotQA()
train = hotpot.train

alf = AlfWorld()
alf_train = alf.train
alf_dev = alf.dev
```

## Removed spine utilities

- `dspy.datasets.dataloader.DataLoader` — removed; use `rows_to_examples` or integration loaders.
- `dspy.datasets.colors.Colors` — removed.
- `Dataset.prepare_by_seed` — removed.
