The tests in this directory are primarily concerned with code correctness and Adapter reliability.

If you're looking for testing the end-to-end quality of DSPy modules and optimizer, refer to [LangProBe](https://github.com/Shangyint/langProBe).

## Running tests

The default pytest command runs tests in parallel with pytest-xdist:

```bash
uv run pytest
```

Pytest is configured to use `-n auto --dist=loadfile` by default. File-level distribution keeps all tests from the same file on one worker, which is a safer default for tests that use module-level state or shared fixtures.

Test module basenames must be unique across the entire `tests/` tree (for example `test_teleprompt_metrics.py`, not `test_metrics.py` in multiple packages). Duplicate basenames break pytest-xdist collection even with `--dist=loadfile`.

To force a serial run, disable xdist for that invocation:

```bash
uv run pytest -n 0
```

Pytest's built-in cache can rerun or prioritize failures from the previous run:

```bash
uv run pytest --lf
uv run pytest --ff
```

For affected-test reruns, use pytest-testmon. The first run builds `.testmondata`; later runs use it to select tests affected by code changes.

```bash
uv run pytest --testmon
```

## Opt-in test categories

Markers are defined in `pyproject.toml`. By default, pytest skips tests marked
`integration`, `llm_call`, `deno`, or `slow`. Pass the matching CLI flag to opt
in (`--integration`, `--llm_call`, `--deno`, `--slow`). You can also select by
marker expression (for example `-m integration`).

Credential and dependency checks still apply as secondary skip reasons after
opt-in. Use `require_env` from `tests.test_utils` in live tests.

| Marker | Flag | Requires | Examples |
|--------|------|----------|----------|
| `integration` | `--integration` | External infra (Postgres, HuggingFace downloads, Databricks workspace, MCP stdio server, local litellm proxy) | `tests/clients/dr_llm/test_integration_pool.py`, `tests/integrations/datasets/test_benchmark_datasets.py` |
| `llm_call` | `--llm_call` | Live LLM provider credentials (`OPENAI_API_KEY`, `LM_FOR_TEST`, provider-specific `LM_FOR_TEST_*`) | `tests/clients/test_lm_direct_live.py` |
| `deno` | `--deno` | Deno runtime installed | `tests/primitives/python_interpreter/`, RLM integration tests |
| `slow` | `--slow` | (none) | Multi-round bootstrap compile tests under `tests/persistence/` |

Example commands:

```bash
uv run pytest                                          # default unit suite
uv run pytest --integration -n 0                       # Postgres pool, HF datasets, litellm proxy, MCP
uv run pytest --llm_call -n 0 tests/clients/test_lm_direct_live.py
uv run pytest --integration --llm_call -n 0 tests/integrations/finetune/test_databricks_live.py
uv run pytest --deno -n 0 tests/primitives/python_interpreter/
uv run pytest --slow -n 0 tests/persistence/
```

## Test doubles

Shared LM and retrieval doubles live under `tests/test_utils/`:

| Double | Use when |
|--------|----------|
| `DummyLM` | End-to-end module/adapter tests needing structured field answers via an adapter |
| `SequentialTextLM` | Plain sequential text responses with optional request recording |
| `SpyLM` | LiteLLM-path tests that need call recording or fixed response text |
| `NativeToolCallLM` | Native provider tool-call loop tests (`parallel_first_turn=True` for parallel first turn) |
| `CapabilityStubLM` | Adapter capability/adaptation-mode tests without LM forwarding |
| `FailingLM` | Error-path tests where the LM must raise |
| `DummyVectorizer` | Retrieval/KNN tests needing deterministic embeddings |

Keep adapter-pipeline helpers such as `CapturingLM` in `tests/adapters/conftest.py` when they depend on adapter internals. Use local nested LM classes only for test-specific contract assertions.
