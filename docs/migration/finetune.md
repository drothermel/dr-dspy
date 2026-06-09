# Finetune provider migration guide

Fine-tuning spans two packages:

| Layer | Import path | Contents |
| --- | --- | --- |
| Spine | `dspy.clients.finetune` | `FinetuneService`, `infer_finetune_provider`, `TrainDataFormat`, `TrainingJob`, `validate_data_format`, protocol types |
| Vendor providers | `dspy.integrations.finetune.{openai,databricks,local}` | Provider namespace classes |

There is no `dspy.integrations.finetune` barrel — import vendor submodules directly.

Inference `LM` (`dspy.clients.lm.LM`) is inference-only. Finetune lifecycle lives on `FinetuneService`.

## Provider auto-inference

When `FinetuneService(lm)` is constructed without `finetune_provider=`, `infer_finetune_provider(model)` selects a finetune provider from the model id:

| Model ID pattern | Inferred provider | Finetunable |
| --- | --- | --- |
| `databricks/{endpoint}` | `DatabricksProvider` | Yes |
| `local:{path}` | `LocalProvider` | Yes |
| `openai/{model}` | `OpenAIProvider` | Yes |
| `ft:{id}` | `OpenAIProvider` | Yes |
| Everything else | `DefaultFinetuneProvider` | No |

Precedence is Databricks → Local → OpenAI → default (most specific vendor prefix first).

## Explicit provider override

Pass `finetune_provider=` when the model id is ambiguous or inference should be overridden:

```python
from dspy.clients.finetune import FinetuneService
from dspy.clients.lm import LM
from dspy.integrations.finetune.databricks import DatabricksProvider
from dspy.integrations.finetune.local import LocalProvider
from dspy.integrations.finetune.openai import OpenAIProvider

lm = LM("meta-llama/Llama-3.2-1B")
FinetuneService(lm, finetune_provider=DatabricksProvider())
FinetuneService(lm, finetune_provider=LocalProvider())
FinetuneService(LM("openai/gpt-4.1-mini"), finetune_provider=OpenAIProvider())
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

Local finetune now returns `local:{output_dir}` instead of `openai/local:{output_dir}`. Update persisted model ids and re-infer or pass `finetune_provider=LocalProvider()` explicitly when reloading checkpoints.

## Finetune entry points

```python
from dspy.clients.finetune import FinetuneService, TrainDataFormat
from dspy.clients.lm import LM

lm = LM("openai/gpt-4.1-mini")
service = FinetuneService(lm, train_kwargs={"n_epochs": 3})
job = service.finetune(train_data=..., train_data_format=TrainDataFormat.CHAT)
finetuned_lm = job.result()  # inference LM with updated model id

service.launch()   # LocalProvider SGLang server
service.kill()     # LocalProvider teardown
```

Orchestration is implemented in `dspy.clients.finetune.service.FinetuneService`. Provider lookup is lazy in `dspy.clients.finetune.registry`.

## Breaking changes (LM finetune surface removed)

- `LM(provider=...)` removed — use `FinetuneService(lm, finetune_provider=...)`
- `lm.launch()` / `lm.kill()` / `lm.finetune()` / `lm.reinforce()` / `lm.infer_provider()` removed
- `lm.provider` attribute removed
- `LM` serialized state no longer includes `finetuning_model`, `launch_kwargs`, `train_kwargs`
- `dspy.clients.finetune.lm` module removed
