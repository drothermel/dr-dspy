# Memoization removal migration guide

DSPy no longer memoizes LM, embedding, or ColBERT HTTP responses. Every call goes to the provider (or your callable) unless the provider itself caches prompts.

## What was removed

| Removed | Migration |
| --- | --- |
| `LM(..., cache=True/False)` | Remove the kwarg; every call is live |
| `configure_cache()`, `DSPY_CACHE`, `Cache` | Remove calls and imports |
| `LMCacheConfig`, `config={"cache": ...}` | Remove from `LMConfig` / call config |
| `rollout_id` on LM / Predict / teleprompters | Remove; use `temperature > 0` (e.g. `lm.copy(temperature=1.0)`) for diversity |
| `Embedder(..., caching=...)` | Remove the kwarg |
| `LMResponse.cache_hit` | Remove checks; responses are always from a live call |
| `DSPY_CACHE_LIMIT` env var | Remove (unused) |
| `~/.dspy_cache` LM memo entries | Orphaned; safe to delete manually |

## What is unchanged

- **`LMPromptCacheConfig`** — provider-side prefix/prompt caching (OpenAI, Anthropic, etc.) still works via `config={"prompt_cache": ...}`.
- **`LMUsage.cache_read_tokens` / `cache_write_tokens`** — provider billing fields on usage objects.
- **`DSPY_CACHEDIR`** — still used for finetune artifact paths under `~/.dspy_cache` (not LM memoization).
- **LiteLLM's own cache** — remains disabled inside DSPy (`litellm.cache = None`).

## Behavioral changes

**Cost and latency.** Re-running optimizers, evals, and notebooks hits the provider every time. Budget accordingly.

**Diversity.** Repeated identical calls may return different answers when `temperature > 0`. At `temperature=0`, some providers still return stable outputs, but DSPy no longer freezes the first answer in a local cache.

**Serialized programs.** Saved LM state no longer includes `"cache": true`. Loading older programs that contain a `"cache"` key is safe — `BaseLM.load_state` ignores it.

**Teleprompters and multi-sample predictors.** Modules that previously set `rollout_id` to partition cache buckets now use `lm.copy(temperature=1.0)` (or your chosen temperature) so sampling drives diversity.

## Example updates

Before:

```python
import dspy

dspy.configure_cache(enable_disk_cache=True)
lm = dspy.LM("openai/gpt-4o-mini", cache=True)
predict = dspy.Predict("q -> a", config={"rollout_id": 3})
```

After:

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
predict = dspy.Predict(QATaskSpec(), config={"temperature": 1.0})
```

Before:

```python
from dspy.clients.embedding import Embedder

embedder = Embedder("text-embedding-3-small", caching=True)
```

After:

```python
from dspy.clients.embedding import Embedder

embedder = Embedder("text-embedding-3-small")
```

Before:

```python
if response.cache_hit:
    ...
```

After:

Remove the branch; treat every response as a fresh provider result.
