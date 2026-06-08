The tests in this directory are primarily concerned with code correctness and Adapter reliability.

If you're looking for testing the end-to-end quality of DSPy modules and optimizer, refer to [LangProBe](https://github.com/Shangyint/langProBe).

## Running tests

The default pytest command runs tests in parallel with pytest-xdist:

```bash
uv run pytest
```

Pytest is configured to use `-n auto --dist=loadfile` by default. File-level distribution keeps all tests from the same file on one worker, which is a safer default for tests that use module-level state or shared fixtures.

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
