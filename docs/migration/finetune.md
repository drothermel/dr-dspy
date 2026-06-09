# Finetune provider migration guide

Fine-tuning spans two packages:

| Layer | Import path | Contents |
| --- | --- | --- |
| Spine | `dspy.clients.finetune` | `TrainDataFormat`, `TrainingJob`, `validate_data_format`, protocol types |
| Vendor providers | `dspy.integrations.finetune.{openai,databricks,local}` | Provider namespace classes |

There is no `dspy.integrations.finetune` barrel — import vendor submodules directly.

## Provider auto-inference

When `LM(..., provider=None)`, `infer_provider(model)` selects a finetune provider from the model id:

| Model ID pattern | Inferred provider | Finetunable |
| --- | --- | --- |
| `databricks/{endpoint}` | `DatabricksProvider` | Yes |
| `local:{path}` | `LocalProvider` | Yes |
| `openai/{model}` | `OpenAIProvider` | Yes |
| `ft:{id}` | `OpenAIProvider` | Yes |
| Everything else | `DefaultFinetuneProvider` | No |

Precedence is Databricks → Local → OpenAI → default (most specific vendor prefix first).

## Explicit provider override

Pass `provider=` when the model id is ambiguous or inference should be overridden:

```python
from dspy.clients.lm import LM
from dspy.integrations.finetune.databricks import DatabricksProvider
from dspy.integrations.finetune.local import LocalProvider
from dspy.integrations.finetune.openai import OpenAIProvider

lm = LM("meta-llama/Llama-3.2-1B", provider=DatabricksProvider())
lm = LM("meta-llama/Llama-3.2-1B", provider=LocalProvider())
lm = LM("openai/gpt-4.1-mini", provider=OpenAIProvider())
```

## Training data formats

Shared validation lives in `dspy.clients.finetune.validate_data_format`.

| Provider | Supported formats |
| --- | --- |
| `OpenAIProvider` | `TrainDataFormat.CHAT`, `TrainDataFormat.COMPLETION` |
| `DatabricksProvider` | `TrainDataFormat.CHAT`, `TrainDataFormat.COMPLETION` |
| `LocalProvider` | `TrainDataFormat.CHAT` only |

Completion records may use `response` instead of `completion` (normalized at validation).

## Post-finetune model IDs

| Provider | Returned model id |
| --- | --- |
| OpenAI | `ft:...` (OpenAI fine-tuned model id) |
| Databricks | `databricks/{endpoint_name}` |
| Local | `local:{output_dir}` |

### Breaking change: local finetune output

Local finetune now returns `local:{output_dir}` instead of `openai/local:{output_dir}`. Update persisted model ids and re-infer or pass `provider=LocalProvider()` explicitly when reloading checkpoints.

## Finetune entry points

```python
job = lm.finetune(train_data=..., train_data_format=TrainDataFormat.CHAT)
lm.launch()   # LocalProvider SGLang server
lm.kill()     # LocalProvider teardown
```

Orchestration is implemented in `dspy.clients.finetune.lm` and delegated from `dspy.clients.lm.LM`.
