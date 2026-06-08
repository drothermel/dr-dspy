# Internal Named-Args Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert multi-argument **internal** call sites in `dspy/` and `tests/` to keyword argument passing for safety, without changing public API signatures or breaking downstream callers.

**Architecture:** Work package-by-package in dependency order. Each phase converts call sites only (no `*` on public constructors or user-facing protocols). Optionally harden **private or module-local** function definitions with keyword-only parameters where callers are entirely in-repo. Verify with the existing pytest suite after each commit; no new behavior.

**Tech Stack:** Python 3.10+, uv, ruff, ty, pytest (per `AGENTS.md`).

**Note on ordering:** File-split work was originally recommended first. If file splits land later, re-run the Phase 0 audit on touched modules — splits tend to reintroduce positional call sites.

---

## Scope

### In scope

- Multi-arg calls inside `dspy/**/*.py` and `tests/**/*.py` where argument meaning is not obvious from position alone.
- Internal helper functions that are not part of the stable public import surface (may add `*,` to their **definitions**).
- Boolean and flag arguments at call sites (`cache=True` not bare `True` as 3rd positional).

### Out of scope (no signature changes, no forced keyword at call sites)

| Category | Examples | Why leave alone |
|----------|----------|-----------------|
| Single-argument calls | `await lm(request)`, `len(x)`, `isinstance(x, T)` | Positional is unambiguous |
| User module invocation | `await program(**example.inputs())`, `await cot(question="...")` | Public API; kwargs already explicit |
| `Predict` / module kwargs | Signature field names as kwargs | Already keyword-only at runtime (`Predict` rejects `*args`) |
| Metric callback protocol | `metric(example, prediction, trace)` | User-supplied functions; documented contract |
| GEPA metric | `metric(gold, pred, trace, pred_name, pred_trace)` | Validated by `inspect.signature().bind(...)` |
| `Example` construction | `Example(**record)`, `Example(a=1, b=2)` | Data container; kwargs pattern |
| `.with_inputs(...)` | `.with_inputs("question")`, `.with_inputs(*keys)` | Variadic field names; positional is idiomatic |
| Public constructors | `LM("gpt-4")`, `Predict("q -> a")`, `ChainOfThought("...")` | Breaking change if forced keyword-only |
| `make_signature(...)` string form | `make_signature("a, b -> c")` | Public API; first arg is naturally positional |
| `Signature.append(...)` **definition** | Keep `append(name, field, type_=None)` | Public; convert **internal call sites** only |
| Stdlib / third-party | `zip(a, b, strict=False)`, `max(...)`, `dict(...)` | Not our API |
| `super().__init__(...)` | Parent init forwarding | Convention |
| Test assertions | `assert x == y`, `pytest.raises(...)` | Not production call sites |

### Already done (verify only in Phase 1)

These were partially addressed during the async migration; confirm no regressions:

- `Adapter.acall(*, lm=, config=, signature=, demos=, inputs=)` in `dspy/adapters/base.py`, `chat_adapter.py`, `json_adapter.py`
- `Predict` / module calls reject positional args
- Most `adapter.acall(...)` test call sites already use keywords

### Still needs work (known hotspots)

| Area | Pattern | Example fix |
|------|---------|-------------|
| `run_bounded` call sites | `run_bounded(items, fn, max_concurrency=...)` | `run_bounded(items=items, fn=fn, ...)` |
| `two_step_adapter.acall` override | Positional params in definition (public) | Call sites use keywords; optionally align override body calls |
| `openai_format` helpers | `completion_to_lm_response(response, request)` | `response=..., request=...` |
| `merge_lm_request_config` | `merge_lm_request_config(lm, config)` | `lm=lm, config=config` |
| `evaluate/metrics.py` | `em_score(prediction, ground_truth)` | Named internal calls |
| `Signature.append` call sites | `.append("trajectory", InputField(), type_=str)` | `.append(name="trajectory", field=InputField(), type_=str)` |
| `react.py`, `code_act.py`, `rlm.py` | Chained `.append(...)` | Name all args |
| `embedding` tests | `await embedding.acall(inputs)` | Single-arg OK; multi-arg calls get names |
| Teleprompt helpers | Multi-arg internal helpers | Per-function audit |
| `mipro_optimizer_v2.py` | `eval_candidate_program(batch_size, valset, ...)` | Named at call site and definition if private |

---

## Phase 0: Baseline and audit

**Commit message:** (no commit — working notes only, or `chore: add named-args audit baseline`)

### Step 0.1: Verify green baseline

```bash
uv run ruff check --fix
uv run ty check --fix
uv run ruff format
uv run pytest tests/ -q --ignore=tests/reliability
```

Expected: green (record any pre-existing failures).

### Step 0.2: Generate audit lists

Run these from repo root. Results are **candidates** — apply the exemption table before editing.

```bash
# Files with comma-separated args (manual review required)
rg -n '\w+\([^)\n]*, [^=)]' dspy --glob '*.py' > /tmp/named-args-dspy.txt
rg -n '\w+\([^)\n]*, [^=)]' tests --glob '*.py' > /tmp/named-args-tests.txt

# Known high-value internal patterns
rg -n 'run_bounded\(\s*\w+,\s*\w+' dspy tests --glob '*.py'
rg -n '\.append\(["\']' dspy/predict --glob '*.py'
rg -n '(completion_to_lm_response|responses_to_lm_response|merge_lm_request_config|em_score|f1_score)\(' dspy --glob '*.py'

# Positional adapter calls (should be empty after async migration)
rg -n 'adapter\.acall\(\s*(lm|self|\w+),' dspy tests --glob '*.py'
```

### Step 0.3: Triage checklist

For each candidate line, ask:

1. Is it a **call site** (not a `def` line)?
2. Does it have **2+ positional args** (excluding `self`/`cls`)?
3. Is it **exempt** per the table above?
4. Would naming args **clarify** intent (especially bools and same-type adjacent args)?

Mark files as: **convert**, **skip**, or **harden-def** (internal function gets `*,`).

- [ ] Baseline tests recorded
- [ ] Audit files generated and triaged

---

## Phase 1: Core utilities and LM spine

**Commit message:** `refactor: use keyword args at internal utility call sites`

**Files:**
- Modify: `dspy/utils/async_parallel.py` (optional: harden definition)
- Modify: `dspy/evaluate/evaluate.py`
- Modify: `dspy/predict/parallel.py`
- Modify: `tests/utils/test_parallelizer.py`
- Modify: `dspy/clients/lm.py`
- Modify: `dspy/clients/openai_format.py`
- Modify: `dspy/core/types/config.py` (call sites only unless `_`-prefixed helpers)
- Modify: `dspy/adapters/base.py`, `dspy/adapters/two_step_adapter.py`

### Task 1.1: Harden `run_bounded` (internal module — safe to harden definition)

**File:** `dspy/utils/async_parallel.py`

Change the signature so `items` and `fn` are keyword-only:

```python
async def run_bounded(
    *,
    items: Sequence[T],
    fn: Callable[[T], Awaitable[R]],
    max_concurrency: int,
    max_errors: int | None = None,
    provide_traceback: bool | None = None,
    disable_progress_bar: bool = False,
    compare_results: bool = False,
) -> tuple[list[R | None], BoundedRunStats]:
```

Update all call sites:

**`dspy/evaluate/evaluate.py`** (~line 167):

```python
results, _stats = await run_bounded(
    items=devset,
    fn=process_item,
    max_concurrency=concurrency or settings.num_threads,
    disable_progress_bar=not display_progress,
    max_errors=(self.max_errors if self.max_errors is not None else settings.max_errors),
    provide_traceback=self.provide_traceback,
    compare_results=True,
)
```

**`dspy/predict/parallel.py`** (~line 90):

```python
results, stats = await run_bounded(
    items=exec_pairs,
    fn=self._run_pair,
    max_concurrency=concurrency,
    max_errors=self.max_errors,
    provide_traceback=self.provide_traceback,
    disable_progress_bar=self.disable_progress_bar,
)
```

**`tests/utils/test_parallelizer.py`** — update `_run_bounded` wrapper and every `run_bounded(...)` / `_run_bounded(...)` call to pass `items=` and `fn=`.

### Task 1.2: OpenAI format and config helpers

**`dspy/clients/lm.py`:**

```python
lm_response = responses_to_lm_response(response=response, request=request)
# ...
lm_response = completion_to_lm_response(response=response, request=request)
```

**`dspy/adapters/base.py` / `two_step_adapter.py`:**

```python
config=merge_lm_request_config(lm=lm, config=config),
```

**Tests** (`tests/clients/test_disk_serialization.py`):

```python
lm_response = completion_to_lm_response(response=response, request=LMRequest(...))
```

Do **not** add `*` to `completion_to_lm_response` / `merge_lm_request_config` public definitions unless all external callers are in-repo (prefer call-site-only conversion if unsure).

### Task 1.3: `two_step_adapter.acall` call-site alignment

Do **not** add `*` to `TwoStepAdapter.acall` (public override). Ensure **body** calls use keywords (nested `ChatAdapter().acall(...)` already does). Fix `await lm.acall(request)` → `await lm.acall(request=request)` only if you adopt a repo-wide single-arg naming rule; **optional** — single-arg positional is exempt.

### Verification

```bash
uv run ruff check --fix && uv run ty check --fix && uv run ruff format
uv run pytest tests/utils/test_parallelizer.py tests/evaluate/ tests/predict/test_parallel.py tests/clients/test_lm.py tests/clients/test_disk_serialization.py tests/adapters/test_two_step_adapter.py -q
```

- [ ] **Commit Phase 1**

---

## Phase 2: Evaluate metrics and scoring helpers

**Commit message:** `refactor: use keyword args in evaluate metrics internals`

**Files:**
- Modify: `dspy/evaluate/metrics.py`
- Modify: `dspy/evaluate/auto_evaluation.py` (internal calls only)
- Test: `tests/evaluate/test_metrics.py`

### Task 2.1: Convert internal metric helper calls

Public functions `EM`, `F1`, `HotPotF1` keep positional signatures (documented public API). Convert **internal** calls:

```python
# Before
return max(em_score(prediction, ans) for ans in answers_list)

# After
return max(em_score(prediction=prediction, ground_truth=ans) for ans in answers_list)
```

Apply the same pattern for `f1_score`, `hotpot_f1_score`, `normalize_text` call chains, and any helper invoked with 2+ positional args inside `metrics.py`.

Leave doctest examples that demonstrate public positional usage unchanged unless doctests are considered internal (they document public API — **keep positional in docstring examples**).

### Verification

```bash
uv run pytest tests/evaluate/ -q
```

- [ ] **Commit Phase 2**

---

## Phase 3: Predict modules and signature manipulation

**Commit message:** `refactor: use keyword args in predict signature construction`

**Files:**
- Modify: `dspy/predict/react.py`
- Modify: `dspy/predict/react_v2.py`
- Modify: `dspy/predict/code_act.py`
- Modify: `dspy/predict/rlm.py`
- Modify: `dspy/predict/refine.py`
- Modify: `dspy/predict/multi_chain_comparison.py`
- Modify: `dspy/predict/program_of_thought.py`
- Test: `tests/predict/test_react.py`, `test_code_act.py`, `test_rlm.py`, etc.

### Task 3.1: `Signature.append` call sites

Pattern (do **not** change `append` definition):

```python
# Before
.append("trajectory", InputField(), type_=str)

# After
.append(name="trajectory", field=InputField(), type_=str)
```

Apply to all `.append(` calls in the files listed above where `name=` is not already used.

### Task 3.2: `make_signature` internal calls with multiple args

Convert when **both** signature and instructions are passed positionally from **internal** code:

```python
# Before
make_signature(fields, instructions)

# After
make_signature(signature=fields, instructions=instructions)
```

Leave string-literal first-arg calls as-is:

```python
make_signature("question, context -> answer")  # OK — public idiomatic usage
```

### Verification

```bash
uv run pytest tests/predict/ -q
```

- [ ] **Commit Phase 3**

---

## Phase 4: Adapters (remaining)

**Commit message:** `refactor: use keyword args at remaining adapter internal call sites`

**Files:**
- Modify: `dspy/adapters/utils.py`
- Modify: `dspy/adapters/json_adapter.py`
- Modify: `dspy/adapters/chat_adapter.py`
- Modify: `dspy/adapters/baml_adapter.py`
- Modify: `dspy/adapters/types/tool.py`
- Test: `tests/adapters/`

### Task 4.1: Audit adapter internals

Focus on:

- Multi-arg calls to `_`-prefixed helpers in `adapters/utils.py`
- `format()` / `parse()` chains with bool or same-type adjacent args
- Any remaining positional `adapter.acall(` in tests (grep from Phase 0 should be empty for `lm,` pattern)

Example bool call-site fix:

```python
# Before
some_helper(data, True, "prefix")

# After
some_helper(data, enabled=True, prefix="prefix")
```

(Use actual parameter names from each function definition.)

### Verification

```bash
uv run pytest tests/adapters/ -q
```

- [ ] **Commit Phase 4**

---

## Phase 5: Teleprompt and propose

**Commit message:** `refactor: use keyword args in teleprompt internal call sites`

**Why separate:** Largest internal surface area; many multi-arg helpers. Do **not** rename metric callback invocations.

**Files (audit all, convert where triaged):**
- `dspy/teleprompt/utils.py`
- `dspy/teleprompt/simba_utils.py`
- `dspy/teleprompt/simba.py`
- `dspy/teleprompt/mipro_optimizer_v2.py`
- `dspy/teleprompt/bettertogether.py`
- `dspy/teleprompt/bootstrap.py`
- `dspy/teleprompt/bootstrap_trace.py`
- `dspy/teleprompt/copro_optimizer.py`
- `dspy/teleprompt/infer_rules.py`
- `dspy/teleprompt/gepa/gepa_utils.py`
- `dspy/teleprompt/gepa/instruction_proposal.py`
- `dspy/teleprompt/avatar_optimizer.py`
- `dspy/teleprompt/grpo.py`
- `dspy/propose/grounded_proposer.py`
- `dspy/propose/dataset_summary_generator.py`
- Test: `tests/teleprompt/`, `tests/propose/`

### Task 5.1: Shared teleprompt helpers first

Start with `teleprompt/utils.py` and `simba_utils.py` — downstream optimizers import these.

**Keep positional (do not convert):**

```python
score = metric(example, prediction, trace)
score = self.metric(example, prediction, trace)
```

**Convert internal helpers**, e.g. `mipro_optimizer_v2.py`:

```python
# Before
eval_candidate_program(batch_size, valset, candidate_program, evaluate, self.rng)

# After — use names from function definition
eval_candidate_program(
    batch_size=batch_size,
    valset=valset,
    candidate_program=candidate_program,
    evaluate=evaluate,
    rng=self.rng,
)
```

If `eval_candidate_program` is module-private (`_` prefix or only used in teleprompt), consider adding `*,` to its definition in the same commit.

### Task 5.2: Optimizer-by-optimizer pass

Work in this order (dependencies):

1. `utils.py`, `simba_utils.py`, `bootstrap_trace.py`
2. `bootstrap.py`, `random_search.py`, `knn_fewshot.py`
3. `copro_optimizer.py`, `infer_rules.py`, `ensemble.py`
4. `simba.py`, `mipro_optimizer_v2.py`, `bettertogether.py`
5. `gepa/`, `grpo.py`, `avatar_optimizer.py`

After each sub-batch:

```bash
uv run pytest tests/teleprompt/test_<relevant>.py -q
```

### Verification (full teleprompt)

```bash
uv run pytest tests/teleprompt/ tests/propose/ -q
```

- [ ] **Commit Phase 5** (or split into 5a helpers + 5b optimizers if diff > ~500 lines)

---

## Phase 6: Remaining `dspy/` packages

**Commit message:** `refactor: use keyword args in remaining internal call sites`

**Files (audit and convert):**
- `dspy/clients/` (except already done in Phase 1)
- `dspy/primitives/` (`base_module.py`, `python_interpreter.py`, `example.py` internal calls only)
- `dspy/utils/` (`callback.py`, `caching.py`, `magicattr.py`, etc.)
- `dspy/retrievers/`
- `dspy/datasets/` (internal loader helpers)
- `dspy/dsp/`
- `dspy/signatures/signature.py` — internal `make_signature` calls only; **no** public API signature changes

### Task 6.1: `callback.py` and `base_module.py`

High file count in grep results — likely many false positives (`isinstance`, `getattr`). Convert only genuine multi-arg **project** function calls.

### Verification

```bash
uv run pytest tests/primitives/ tests/utils/ tests/clients/ tests/retrievers/ tests/signatures/ -q
```

- [ ] **Commit Phase 6**

---

## Phase 7: Test suite sweep

**Commit message:** `test: use keyword args at internal multi-arg call sites`

**Files:** Remaining `tests/**/*.py` from Phase 0 audit (`/tmp/named-args-tests.txt`).

### Rules for tests

- Convert helper/setup calls with 2+ positional args.
- Leave `assert`, `pytest.raises`, fixture params, and public API demos unchanged.
- Prefer keywords in `DummyLM([...])` companion calls only when multiple args exist.

Large files — work in chunks:

| File | Lines | Strategy |
|------|-------|----------|
| `tests/adapters/test_chat_adapter.py` | ~3150 | Split into 2–3 commits by test class section |
| `tests/adapters/test_json_adapter.py` | ~1880 | Same |
| `tests/predict/test_predict.py` | ~1760 | Same |
| `tests/clients/test_lm.py` | ~1585 | Same |

### Verification

```bash
uv run ruff check --fix && uv run ty check --fix && uv run ruff format
uv run pytest tests/ -q --ignore=tests/reliability
```

- [ ] **Commit Phase 7**

---

## Phase 8: Guardrails (optional, same PR or follow-up)

**Commit message:** `chore: document internal named-arg conventions`

### Task 8.1: Add `AGENTS.md` note

Append a short section:

```markdown
## Internal call-site conventions

- Use keyword arguments for multi-arg calls to DSPy-internal functions when meaning is not obvious from position.
- Do not add keyword-only `*` to public constructors or documented callback protocols.
- `run_bounded`, adapter `acall`, and other spine APIs require keywords at call sites.
```

### Task 8.2: Optional ruff enforcement (internal modules only)

Do **not** enable repo-wide `FBT001`/`FBT002` on definitions — that pushes public signature changes.

Optional per-path rule in `pyproject.toml` for **new** internal modules only (defer until after this migration):

```toml
# Example — only if team wants mechanical enforcement later
# "dspy/utils/async_parallel.py" = ["FBT001"]
```

Skip unless explicitly requested.

- [ ] **Commit Phase 8**

---

## Phase summary

| Phase | Scope | Commit |
|-------|-------|--------|
| 0 | Baseline + audit | (notes) |
| 1 | `run_bounded`, LM/config helpers | `refactor: use keyword args at internal utility call sites` |
| 2 | `evaluate/metrics.py` | `refactor: use keyword args in evaluate metrics internals` |
| 3 | Predict + `Signature.append` | `refactor: use keyword args in predict signature construction` |
| 4 | Adapters remainder | `refactor: use keyword args at remaining adapter internal call sites` |
| 5 | Teleprompt + propose | `refactor: use keyword args in teleprompt internal call sites` |
| 6 | Other `dspy/` packages | `refactor: use keyword args in remaining internal call sites` |
| 7 | Tests sweep | `test: use keyword args at internal multi-arg call sites` |
| 8 | AGENTS.md guardrails | `chore: document internal named-arg conventions` |

---

## Risk notes

1. **False positives in regex audit** — Always read context; do not bulk-rewrite `isinstance`, `getattr`, or stdlib calls.

2. **Metric callbacks** — Uniform `(example, prediction, trace)` positional protocol is intentional; converting to keywords at invoke sites would be inconsistent with user-implemented metrics.

3. **File splits later** — If module splits happen after this work, re-run Phase 0 greps on new files.

4. **Definition hardening vs call sites** — Prefer call-site conversion for anything exported or documented. Use `*,` on definitions only for `_`-private helpers or modules not imported from `dspy.__init__`.

5. **Review size** — Phase 5 and Phase 7 are the long poles. Split commits if diffs exceed ~400–500 lines.

---

## Self-review (spec coverage)

| Requirement | Phase |
|-------------|-------|
| Internal call sites only | All phases; exemption table |
| No public breaking changes | Exemptions; no `*` on public constructors |
| Leave positional where idiomatic | Exemption table |
| Verify each phase | Verification blocks |
| Commit per phase | Phase summary |
| Works before file splits | Note at top; re-audit after splits |
