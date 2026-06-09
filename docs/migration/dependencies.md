# Dependency requirements (3.3+)

## Core runtime floors and caps

| Package | Requirement |
|---------|-------------|
| `litellm` | `>=1.88.0` |
| `openai` | `>=2.20.0,<3.0.0` |
| `pydantic` | `>=2.13.0,<3.0.0` |
| `tenacity` | `>=8.2.3` (direct dep; still required by LiteLLM retry) |

OpenAI Python SDK **1.x is no longer supported**. Pydantic **2.12 and below** are below the supported floor.

## LiteLLM retry behavior

`LM(..., num_retries=N)` delegates to LiteLLM with `retry_strategy="exponential_backoff_retry"`. Under litellm 1.88+, the number of provider attempts for a failing call is **higher** than under 1.68 for the same `num_retries` value. Re-tune `num_retries` if you relied on the old attempt counts.

## Optional extras

`litellm` is a **core** dependency (not an optional extra). If LiteLLM is missing at runtime, install with `pip install litellm` or reinstall `dspy`.

Other optional groups (`anthropic`, `mcp`, `tools`, etc.) have tightened bounds; see `pyproject.toml` for current specifiers.
