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
