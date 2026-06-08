# Async-Only Public API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse DSPy to a single async execution path end-to-end and expose an async-only public API (`await module(...)`) with streaming removed for now.

**Architecture:** Delete streaming and sync/async duplicate paths at LM → Adapter → Predict. Make `Module.__call__`, `BaseLM.__call__`, and `Adapter.acall` the only runtime entry points (all async). Migrate composite modules, batch utilities, teleprompt, and tests to `await` + `asyncio.gather`/semaphore. Preserve `LMStream` types in `dspy/core/types.py` for a future streaming reimplementation.

**Tech Stack:** Python 3.10+, asyncio, pytest-asyncio (already in dev deps), LiteLLM via `acompletion`, uv/ruff/ty per `AGENTS.md`.

**Branch:** `06-08-cleanup`

---

## Pre-Phase 0: Baseline the branch

The branch currently has unrelated WIP. Start from a clean baseline so each phase commit is reviewable.

**Files:** working tree on `06-08-cleanup`

- [ ] **Step 1: Review existing changes**

Run:
```bash
git status
git diff
```

- [ ] **Step 2: Commit or stash WIP**

Either finish/commit the in-flight cleanup separately, or:
```bash
git stash push -m "pre-async-migration WIP"
```

- [ ] **Step 3: Verify tests pass before migration**

Run:
```bash
uv run pytest tests/ -q --ignore=tests/reliability --ignore=tests/streaming
```

Expected: green (or note known failures).

---

## Phase 1: Remove streaming and sync-bridge utilities

**Commit message:** `refactor: remove streaming and sync/async bridge utilities`

**Why first:** Streaming is isolated (~5 files + hooks). Removing it eliminates `asyncify`, `send_stream`, and the worst sync/async bridging in `lm.py` before the core migration.

### Files to delete

- `dspy/streaming/__init__.py`
- `dspy/streaming/streamify.py`
- `dspy/streaming/streaming_listener.py`
- `dspy/streaming/messages.py`
- `dspy/utils/asyncify.py`
- `dspy/utils/syncify.py`
- `tests/streaming/test_streaming.py`
- `tests/utils/test_asyncify.py`
- `tests/utils/test_syncify.py`

### Files to modify

**`dspy/dsp/utils/settings.py`**
- Remove from `DEFAULT_CONFIG`: `send_stream`, `caller_predict`, `stream_listeners`
- Remove any `settings.context` docs referencing streaming propagation (keep `async_max_workers` for now or rename in Phase 5)

**`dspy/clients/lm.py`**
- Delete `_get_stream_completion_fn` entirely
- In `litellm_completion` / `alitellm_completion`: always call non-streaming completion paths (`completion` / `acompletion` without `stream=True`)
- Remove `anyio.from_thread` import if no longer needed

**`dspy/predict/predict.py`**
- Delete `_should_stream()` and both `settings.context(caller_predict=...)` / `send_stream=None` branches
- Both `forward` and `aforward` should call adapter without streaming conditionals (consolidated in Phase 3)

**`dspy/adapters/types/base_type.py`, `reasoning.py`, `citation.py`**
- Remove or stub `parse_stream_chunk` classmethods (keep stubs returning `None` if referenced elsewhere; delete if unused after streaming removal)

**`tests/clients/test_lazy_litellm_import.py`**
- Remove `streamify` import test

### Preserve (do not delete)

- `dspy/core/types.py`: `LMStream`, `AsyncLMStream`, `LMStreamEvent*`, `LMOutputBuilder` — future streaming API

### Verification

```bash
uv run ruff check --fix dspy/ tests/
uv run ty check --fix
uv run ruff format
uv run pytest tests/ -q --ignore=tests/reliability
```

- [x] **Commit Phase 1** (`2d104b89`)

---

## Phase 1.5: Split `dspy/core/types` into a package

**Commit message:** `refactor: split dspy.core.types into package`

**Why here:** `types.py` is ~2k lines and mostly data models — orthogonal to async execution. Split before Phase 2 so LM/Adapter work touches stable import paths.

**Rule:** Pure move + re-export only. No behavior changes in this commit.

### Target layout

```
dspy/core/types/
  __init__.py      # re-exports everything: from dspy.core.types import LMRequest still works
  messages.py      # LMMessage, parts, deltas
  request.py       # LMRequest, LMConfig, tool specs, coercion helpers
  response.py      # LMResponse, LMUsage, LMOutput, LMHistoryEntry
  stream.py        # LMStreamEvent*, LMStream, AsyncLMStream, LMOutputBuilder
  builders.py      # LMOutputBuilder helpers, message construction utilities (if needed)
```

Use judgment on exact file boundaries — goal is readable `tree`, not perfect taxonomy.

### Verification

```bash
uv run ruff check --fix && uv run ty check --fix && uv run ruff format
uv run pytest tests/core/ tests/clients/test_lm.py -q
```

- [x] **Commit Phase 1.5** (`a4afdc75`)

---

## Phase 2: LM and Adapter async-only spine

**Commit message:** `refactor: make LM and Adapter async-only`

**Target API:**

```python
# BaseLM / LM — only async entry
async def __call__(self, request: LMRequest) -> LMResponse: ...
async def aforward(self, request: LMRequest) -> LMResponse: ...  # implements logic

# Adapter — only async entry
async def acall(self, lm, config, signature, demos, inputs) -> list[dict[str, Any]]: ...
```

### `dspy/clients/base_lm.py`

- [ ] Delete `forward()` and sync `__call__` body that calls it
- [ ] Make `async def __call__` delegate to `aforward` + `_finalize_lm_response` (move current `acall` logic to `__call__`, or keep `acall` as alias):

```python
@with_callbacks
async def __call__(self, request: LMRequest) -> LMResponse:
    if not isinstance(request, LMRequest):
        raise TypeError(...)
    response = await self.aforward(request)
    if not isinstance(response, LMResponse):
        raise TypeError(...)
    return self._finalize_lm_response(request, response)

async def acall(self, request: LMRequest) -> LMResponse:
    return await self.__call__(request)
```

- [ ] Delete sync `forward()` abstract method; keep `async def aforward()` as the subclass contract

### `dspy/clients/lm.py`

- [ ] Delete sync `forward()` and `litellm_completion` / `litellm_text_completion` / `litellm_responses_completion` if unused after migration
- [ ] Keep only `aforward()` → `alitellm_*` helpers
- [ ] Update `utils/dummies.py` `DummyLM` / test LMs: async-only

### `dspy/adapters/base.py`

- [ ] Delete sync `__call__`, `_call_lm`, and `forward`-named paths
- [ ] Rename `_acall_lm` → `_call_lm` (async only):

```python
async def _call_lm(self, lm: BaseLM, request: LMRequest) -> LMResponse:
    return await lm(request)
```

- [ ] Single public method `async def acall(...)` (keep name for now; all callers use `await adapter.acall(...)`)

### Keyword-only internal APIs (fold into Phase 2)

While rewriting call sites, make spine methods keyword-only. **Do not** repo-wide strictness on public constructors (`Predict("q -> a")`) or user program kwargs — that is Phase 9.

**`dspy/adapters/base.py`:**

```python
async def acall(
    self,
    *,
    lm: BaseLM,
    config: LMConfig | Mapping[str, Any] | None,
    signature: type[Signature],
    demos: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    ...
```

**`dspy/utils/async_parallel.py`** (Phase 5): `run_bounded(..., *, max_concurrency: int)`

Update Predict and any touched internal callers to use keyword args at the adapter boundary.

### Adapter subclasses

Update any override of sync `__call__` to async-only:
- `dspy/adapters/chat_adapter.py`
- `dspy/adapters/json_adapter.py`
- `dspy/adapters/two_step_adapter.py`
- `dspy/adapters/baml_adapter.py` (if applicable)

Pattern: delete sync `__call__` overrides; keep async `acall` overrides calling `super().acall(...)`.

### `dspy/clients/embedding.py` (same phase — same pattern)

- [ ] Delete sync `__call__`; make `async def acall` the only entry (or async `__call__` + `acall` alias)
- [ ] Update `tests/clients/test_embedding.py`

### Tests to update in this phase

- `tests/clients/test_lm.py` — replace `lm(request)` with `await lm(request)` or `await lm.acall(request)`
- `tests/adapters/test_chat_adapter.py`
- `tests/adapters/test_json_adapter.py`
- `tests/adapters/test_two_step_adapter.py`
- `tests/adapters/test_baml_adapter.py`
- `tests/test_utils/spy_lm.py`

Use `@pytest.mark.asyncio` on converted tests.

### Verification

```bash
uv run pytest tests/clients/ tests/adapters/ -q
uv run ruff check --fix && uv run ty check --fix && uv run ruff format
```

- [x] **Commit Phase 2** (pending hash below)

```bash
git commit -m "$(cat <<'EOF'
refactor: make LM and Adapter async-only

Remove sync forward/__call__ paths from BaseLM, LM, Adapter, and Embedding.
All LM I/O now goes through await lm(request).
EOF
)"
```

---

## Phase 3: Module and Predict async-only entry points

**Commit message:** `refactor: make Module and Predict async-only`

**Target public API:**

```python
# User code
result = await predict(question="...")

# Subclass contract
class MyProgram(Module):
    async def aforward(self, question: str) -> Prediction:
        return await self.predictor(question=question)
```

### `dspy/primitives/module.py`

- [ ] Replace sync `__call__` with async `__call__`:

```python
@with_callbacks
async def __call__(self, *args, **kwargs) -> Prediction:
    from dspy.dsp.utils.settings import thread_local_overrides
    caller_modules = settings.caller_modules or []
    caller_modules = list(caller_modules)
    caller_modules.append(self)
    with settings.context(caller_modules=caller_modules):
        if settings.track_usage and thread_local_overrides.get().get("usage_tracker") is None:
            with track_usage() as usage_tracker:
                output = await self.aforward(*args, **kwargs)
            tokens = usage_tracker.get_total_tokens()
            self._set_lm_usage(tokens, output)
            return output
        return await self.aforward(*args, **kwargs)
```

- [ ] Keep `acall` as alias: `acall = __call__` (or thin wrapper) for transitional grep stability
- [ ] Add base `aforward` that raises `NotImplementedError` with message pointing to async `aforward` requirement (optional — can leave abstract via convention)

### `dspy/predict/predict.py`

- [ ] Delete `forward()`
- [ ] Consolidate logic into `aforward()` only:

```python
async def aforward(self, **kwargs):
    lm, config, signature, demos, kwargs = self._forward_preprocess(**kwargs)
    adapter = settings.adapter or ChatAdapter()
    completions = await adapter.acall(lm, config, signature, demos, kwargs)
    return self._forward_postprocess(completions, signature, **kwargs)
```

- [ ] Rename `_forward_preprocess` / `_forward_postprocess` → `_preprocess` / `_postprocess` (optional cleanup)
- [ ] Update `Parameter` base if it has sync `__call__` — check `dspy/predict/parameter.py`

### `dspy/predict/chain_of_thought.py`, `react.py`, `rlm.py`

- [ ] Delete sync `forward()`; keep only `aforward()` delegating with `await self.predict(...)` / `await self.predict.acall(...)`

Prefer `await self.predict(...)` once `Predict.__call__` is async.

### Verification

```bash
uv run pytest tests/predict/test_predict.py tests/predict/test_chain_of_thought.py tests/predict/test_react.py tests/primitives/ -q
```

- [ ] **Commit Phase 3**

```bash
git commit -m "$(cat <<'EOF'
refactor: make Module and Predict async-only

Module.__call__ is now async; Predict implements a single aforward path.
EOF
)"
```

---

## Phase 4: Migrate composite Predict modules

**Commit message:** `refactor: migrate composite predict modules to async-only`

Each module: delete `forward()`, implement `async def aforward()`, replace internal `mod(**kwargs)` / `self.foo(**kwargs)` with `await mod(...)` / `await self.foo(...)`.

### Files (all under `dspy/predict/`)

| File | Notes |
|------|-------|
| `best_of_n.py` | Loop: `pred = await mod(**kwargs)` |
| `refine.py` | Await nested module calls |
| `retry.py` | Uncomment/update if still used |
| `multi_chain_comparison.py` | `await self.predict(...)` |
| `program_of_thought.py` | Multiple internal predict calls |
| `code_act.py` | Tool + predict loops |
| `react_v2.py` | Agent loop |
| `parallel.py` | **Defer batch rewrite to Phase 5** — for now make `aforward` call async gather stub or raise `NotImplementedError` with message, OR do minimal async port here |
| `avatar/avatar.py` | Actor loops |

### `dspy/utils/dummies.py`

- [ ] `DummyLM`, dummy modules: async-only

### Tests

- `tests/predict/test_best_of_n.py`
- `tests/predict/test_refine.py`
- `tests/predict/test_code_act.py`
- `tests/predict/test_parallel.py` (partial — full fix in Phase 5)
- Any other predict tests failing after migration

### Verification

```bash
uv run pytest tests/predict/ -q
```

- [ ] **Commit Phase 4**

```bash
git commit -m "$(cat <<'EOF'
refactor: migrate composite predict modules to async-only

Convert BestOfN, Refine, ProgramOfThought, CodeAct, and related modules
to aforward with awaited internal composition.
EOF
)"
```

---

## Phase 5: Async batch utilities (Evaluate, Parallel, parallelizer)

**Commit message:** `refactor: replace thread pools with async batch execution`

### New helper: `dspy/utils/async_parallel.py`

Create a focused utility (don't over-abstract):

```python
import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")

async def run_bounded(
    items: Sequence[T],
    fn: Callable[[T], Awaitable[R]],
    *,
    max_concurrency: int,
) -> list[R]:
    sem = asyncio.Semaphore(max_concurrency)

    async def run_one(item: T) -> R:
        async with sem:
            return await fn(item)

    return list(await asyncio.gather(*(run_one(item) for item in items)))
```

### `dspy/evaluate/evaluate.py`

- [ ] Replace `ParallelExecutor` with `run_bounded`
- [ ] Make `Evaluate.__call__` async:

```python
async def __call__(self, program, devset=None, ..., callback_metadata=None) -> EvaluationResult:
    ...
    async def process_item(example):
        with settings.context(trace=[]):
            prediction = await program(**example.inputs())
            trace = list(settings.trace)
        score = self.metric(example, prediction, trace)
        return prediction, score

    results = await run_bounded(
        devset,
        process_item,
        max_concurrency=num_threads or settings.num_threads,
    )
    ...
```

- [ ] Rename param `num_threads` → `max_concurrency` in `Evaluate.__init__` (keep `num_threads` as deprecated alias accepting same value — optional but nice)

### `dspy/predict/parallel.py`

- [ ] Replace `ParallelExecutor` + sync `forward` with async API:

```python
class Parallel:
    async def __call__(self, exec_pairs, *args, **kwargs) -> list:
        async def run_pair(pair):
            module, example = pair
            inputs = example.inputs() if self.access_examples else example
            return await module(**inputs, *args, **kwargs)
        return await run_bounded(exec_pairs, run_pair, max_concurrency=self.num_threads)
```

- [ ] Delete sync `forward()`

### `dspy/utils/parallelizer.py`

- [ ] Either delete if unused after Evaluate migration, or add `AsyncParallelExecutor` and migrate callers
- [ ] Grep for `ParallelExecutor` and update all call sites

### `dspy/dsp/utils/settings.py`

- [ ] Consider renaming `num_threads` → `max_concurrency` in defaults (optional; can alias)

### Tests

- `tests/evaluate/test_evaluate.py` — all tests become `@pytest.mark.asyncio`
- `tests/predict/test_parallel.py`
- `tests/utils/test_parallelizer.py` — rewrite or delete

### Verification

```bash
uv run pytest tests/evaluate/ tests/predict/test_parallel.py tests/utils/test_parallelizer.py -q
```

- [ ] **Commit Phase 5**

```bash
git commit -m "$(cat <<'EOF'
refactor: replace thread pools with async batch execution

Evaluate and Parallel now use asyncio.Semaphore-bounded gather instead
of ParallelExecutor thread pools.
EOF
)"
```

---

## Phase 6: Teleprompt, propose, retrievers, evaluate metrics

**Commit message:** `refactor: migrate teleprompt and remaining modules to async-only`

### Teleprompt — key patterns

**`dspy/teleprompt/bootstrap_trace.py`**
- Patch `aforward` instead of `forward`:

```python
original_aforward = object.__getattribute__(program, "aforward")

async def patched_aforward(program_to_use: Module, **kwargs):
    with settings.context(trace=[]):
        return await original_aforward(**kwargs), settings.trace.copy()

program.aforward = MethodType(patched_aforward, program)
```

**`dspy/teleprompt/knn_fewshot.py`**
- Replace `forward_pass` / `student_copy.forward = ...` with `aforward` patching
- `return await compiled_program(**kwargs)`

**`dspy/teleprompt/simba_utils.py`**
- `wrap_program`: async wrapper, `prediction = await program(**example.inputs())`

**`dspy/teleprompt/simba.py`**
- `Parallel(...)` → `await Parallel(...)(...)`
- All optimizer loops that call `evaluate(...)` → `await evaluate(...)`

**`dspy/teleprompt/utils.py`**
- `eval_candidate_program` → async, await evaluate

**Other teleprompt files** — grep and fix:
```bash
rg 'program\(\*\*|\.forward|evaluate\(' dspy/teleprompt/
```

Files likely needing updates:
- `mipro_optimizer_v2.py`
- `bettertogether.py`
- `infer_rules.py` (+ `RulesInduction` module `aforward`)
- `ensemble.py`
- `gepa/instruction_proposal.py`
- `gepa/gepa_utils.py`
- `avatar_optimizer.py`
- `grpo.py`
- `bootstrap.py`, `random_search.py`, `copro_optimizer.py`, etc.

Strategy: migrate shared helpers first (`utils.py`, `simba_utils.py`, `bootstrap_trace.py`), then fix optimizers top-down.

### `dspy/evaluate/auto_evaluation.py`

- [ ] Metrics modules with `forward` → `aforward` if they call LMs (grep file)

### `dspy/propose/grounded_proposer.py`

- [ ] Async `aforward`

### Retrievers (lower priority if not used in your experiments — still migrate for consistency)

- `dspy/retrievers/embeddings.py`
- `dspy/retrievers/types.py`
- `dspy/retrievers/weaviate_rm.py`
- `dspy/retrievers/databricks_rm.py`
- `dspy/dsp/colbertv2.py`

Retrievers may stay sync if they only hit local/vector DB — **only async-ify those that call modules or LMs**. Document any intentionally-sync retrievers.

### Tests

- `tests/teleprompt/` — bulk `@pytest.mark.asyncio` + await
- `tests/propose/`
- `tests/evaluate/test_metrics.py`

### Verification

```bash
uv run pytest tests/teleprompt/ tests/propose/ tests/evaluate/ -q
```

- [ ] **Commit Phase 6**

```bash
git commit -m "$(cat <<'EOF'
refactor: migrate teleprompt and remaining modules to async-only

Update optimizers, bootstrap tracing, and eval helpers to await programs
and async Evaluate.
EOF
)"
```

---

## Phase 7: Full test suite migration and pytest config

**Commit message:** `test: migrate test suite to async-only public API`

### Pytest configuration

**`pyproject.toml`** — add to `[tool.pytest.ini_options]`:

```toml
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
```

Note: project uses `@pytest.mark.anyio` in some tests — standardize on `@pytest.mark.asyncio` OR keep anyio if already configured; pick one style and apply consistently.

### Bulk test migration checklist

Run to find remaining sync callers:
```bash
rg '\bprogram\(|\bmodule\(|\bpredict\(|\breact\(|\bcot\(|\.forward\(' tests/ --glob '*.py'
rg 'from dspy.utils.asyncify|from dspy.utils.syncify|from dspy.streaming' tests/
```

High-volume files:
- `tests/adapters/test_json_adapter.py` (~21 call sites)
- `tests/adapters/test_chat_adapter.py`
- `tests/predict/test_predict.py` (~100 references — many are constructors)
- `tests/primitives/test_base_module.py`
- `tests/utils/test_saving.py`
- `tests/signatures/`

### Test helper pattern

Add to `tests/conftest.py` if not present:

```python
import pytest

@pytest.fixture
def anyio_backend():
    return "asyncio"
```

For module invocation:
```python
@pytest.mark.asyncio
async def test_foo(program):
    result = await program(question="...")
    assert result.answer
```

### Callback tests

**`tests/callback/test_callback.py`**
- Callbacks already support async via `with_callbacks` — update tests to `await target(...)` not `target.forward(...)`

### Delete obsolete tests

Already deleted in Phase 1: streaming, asyncify, syncify

### Full verification

```bash
uv run ruff check --fix
uv run ty check --fix
uv run ruff format
uv run pytest tests/ -q --ignore=tests/reliability
```

- [ ] **Commit Phase 7**

```bash
git commit -m "$(cat <<'EOF'
test: migrate test suite to async-only public API

Configure pytest-asyncio and update remaining tests to await module calls.
EOF
)"
```

---

## Phase 9 (future / separate plan): Public API strictness and more file splits

**Not part of the async migration.** Do after Phase 8 when the test suite is fully async.

- Keyword-only public constructors where appropriate (`Predict(signature="q -> a")`) — large breaking change
- Split `dspy/adapters/base.py`, `dspy/clients/lm.py` if still desired
- Broader pass-by-name lint or ruff rule for internal call sites

---

## Phase 8: API cleanup, docs, and changelog

**Commit message:** `docs: document async-only API and remove deprecated entry points`

### API hardening

- [ ] Remove `Module.acall` alias if redundant with async `__call__` — **OR** keep `acall` as permanent alias (recommend keeping alias one release, then remove)
- [ ] Remove `BaseLM.acall` alias similarly
- [ ] Grep for any remaining sync `forward` in `dspy/`:

```bash
rg 'def forward\(' dspy/
```

Expected: none (except comments/docstrings).

- [ ] Grep for sync LM invocation:

```bash
rg '\blm\(request\)|litellm_completion\(' dspy/ tests/
```

Expected: none.

### Documentation

- [ ] Update docstrings in `Module`, `Predict`, `Evaluate`, `Parallel` with async examples
- [ ] Add `CHANGELOG.md` entry (file exists untracked — create/update):

```markdown
## [Unreleased]

### Breaking

- DSPy modules are async-only. Use `await program(...)` instead of `program(...)`.
- `Evaluate` and `Parallel` are async: `await evaluate(program, devset=...)`, `await parallel(pairs)`.
- Removed streaming (`streamify`, `StreamListener`) and sync bridges (`asyncify`, `syncify`).
- `BaseLM.forward` and sync `Adapter.__call__` removed; use `await lm(request)` and `await adapter.acall(...)`.
```

### Optional: migration guide snippet for README/AGENTS.md

```python
# Before
result = program(question="What is DSPy?")

# After
result = await program(question="What is DSPy?")

# Scripts
import asyncio
asyncio.run(main())
```

### Final verification

```bash
uv run ruff check --fix && uv run ty check --fix && uv run ruff format
uv run pytest tests/ -q --ignore=tests/reliability
```

- [ ] **Commit Phase 8**

```bash
git commit -m "$(cat <<'EOF'
docs: document async-only API and remove deprecated entry points

Add changelog and update docstrings for await-based module invocation.
EOF
)"
```

---

## Phase summary

| Phase | Scope | Commit |
|-------|-------|--------|
| 0 | Baseline branch | (optional stash/commit) |
| 1 | Remove streaming + asyncify/syncify | `refactor: remove streaming...` |
| 1.5 | Split `dspy/core/types/` package | `refactor: split dspy.core.types into package` |
| 2 | LM + Adapter + Embedding async-only + keyword-only spine | `refactor: make LM and Adapter async-only` |
| 3 | Module + Predict entry points | `refactor: make Module and Predict async-only` |
| 4 | Composite predict modules | `refactor: migrate composite predict modules...` |
| 5 | Evaluate + Parallel + async batch | `refactor: replace thread pools...` |
| 6 | Teleprompt + propose + retrievers | `refactor: migrate teleprompt...` |
| 7 | Full test suite | `test: migrate test suite...` |
| 8 | Docs + changelog | `docs: document async-only API...` |

---

## Risk notes

1. **Teleprompt is the long pole** — Phase 6 may take as long as Phases 2–4 combined. Consider splitting teleprompt into 6a (shared helpers) and 6b (optimizers) with two commits if review size matters.

2. **Metric callables in Evaluate** — today metrics are sync `(example, pred, trace) -> score`. Keep sync metrics unless you have async metrics; only the program invocation needs await.

3. **Nested event loops** — after migration, never call sync wrappers. No `asyncio.run` inside running loops.

4. **pytest-xdist** — `asyncio_mode = auto` works with xdist; if flakiness appears, run problematic tests with `-n0`.

5. **Intentionally sync code** — file I/O, pandas display in Evaluate, logging: keep sync. Only LM/module invocation paths go async.

---

## Future streaming re-entry (out of scope)

When re-adding streaming:
- Implement `BaseLM.astream(request) -> AsyncLMStream` using `LMStreamEvent` types
- Add optional `Module.astream()` wrapper
- Do **not** restore old `streamify` / LiteLLM chunk tagging via `settings.send_stream`

---

## Self-review (spec coverage)

| Requirement | Phase |
|-------------|-------|
| Async-only public API | 3, 7, 8 |
| No streaming (preserve option) | 1 (preserve types), Future section |
| LM/Adapter single path | 2 |
| All modules migrated | 4, 6 |
| Evaluate/Parallel async | 5 |
| Tests green | Each phase partial; 7 full |
| Commit per phase | All phases |
