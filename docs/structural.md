# Structural Review: Spine Submodules

Manual structural review of `dspy/_internal`, `dspy/_meta`, `dspy/core`,
`dspy/primitives`, `dspy/runtime`, `dspy/task_spec`, `dspy/testing`, and
`dspy/transparency`.

Additional manual pass covers `dspy/history`, `dspy/serialization`,
`dspy/clients`, `dspy/adapters`, `dspy/integrations`, `dspy/persistence`,
`dspy/predict`, and `dspy/teleprompt`.

Scope: behavior-preserving improvements to boundaries, flow, robustness,
simplicity, and ease of modification. This is not a feature-change list or a
mechanical style backlog.

Source tags:

- **Both reviews**: independently identified in both manual passes.
- **Manual review**: identified in the Codex manual pass in this thread.
- **External review**: identified in the provided second-agent review.

Path note: `dspy/_meta/` and `dspy/transparency/` have no tracked Python source
in this checkout. They are empty package shells with local `__pycache__` only.
The live transparency implementation is `dspy/runtime/transparency.py`; metadata
appears to live outside the reviewed paths.

## Tier 0: Correctness-Preserving Robustness Fixes

These are small structural hardening changes that should happen before larger
file moves because they remove latent hangs or ambiguous state.

### 0.1 Harden `_internal.Unbatchify`

**Source:** Both reviews.

`dspy/_internal/unbatchify.py` can leave callers blocked forever:

- `__call__` accepts new work after `close()`, but the worker has already joined,
  so `future.result()` never resolves.
- `_worker()` zips batch outputs to futures with `strict=False`; if `batch_fn`
  returns fewer outputs than inputs, remaining futures never resolve.

Refactor:

- Reject calls after close with a clear exception.
- Validate output count before setting futures.
- Prefer `zip(..., strict=True)` or explicit length checks so mismatch failures
  are immediate and diagnosable.
- Consider moving this helper to `dspy/retrievers/_unbatchify.py` if
  `dspy/retrievers/embeddings.py` remains its only production consumer.

### 0.2 Stop using `None` as the completion sentinel in `run_bounded`

**Source:** Manual review.

`dspy/runtime/async_parallel.py` initializes `results` with `None` and counts
completed work by non-`None` entries. A valid task result of `None` is therefore
indistinguishable from an unfinished or failed task.

Refactor:

- Track completion separately from result values, or use a private sentinel.
- Keep the public return shape unchanged.
- Preserve current cancellation and error-count behavior.

## Tier 1: Layer Boundaries and Hidden Coupling

These have the highest leverage for making the reviewed paths a stable
foundation. They reduce upward dependencies and clarify which modules own which
contracts.

### 1.1 Break the `dspy/core` to `dspy/clients` dependency

**Source:** Both reviews.

`dspy/core/types/openai_compat.py` and `dspy/core/types/parts/openai.py` import
client-layer helpers such as media URI handling and OpenAI binary serialization.
This inverts the intended spine direction: client/provider code should depend on
core types, not the reverse.

Refactor:

- Move direction-neutral helpers such as `data_uri` and `split_data_uri` into a
  core-owned module, for example `dspy/core/media.py`.
- Make `openai_compat.py` assemble OpenAI messages but delegate per-part
  conversion to one canonical serializer.
- Keep existing parity coverage, such as history/live serializer agreement
  tests, as the guardrail while consolidating.

### 1.2 Break the `dspy/task_spec` to `dspy/adapters` dependency

**Source:** Both reviews.

`dspy/task_spec/field_format.py` imports adapter-specific types such as `Code`
and `Reasoning`; `dspy/task_spec/formatting.py` imports adapter prompt-formatting
helpers; `dspy/task_spec/type_registry.py` hard-codes adapter module paths.

Why it matters: task specs should describe field contracts. Adapter prompt
formatting and provider-specific type interpretation should sit at adapter
boundaries.

Refactor:

- Keep generic value formatting in task-spec code.
- Move adapter-aware prompt descriptions to an adapter module such as
  `dspy/adapters/task_spec_prompt.py` or existing adapter formatting code.
- For serialization, replace hard-coded adapter imports with a registration
  mechanism or lazy lookup that does not make task-spec import adapter modules.

### 1.3 Split `runtime/transparency.py` by responsibility

**Source:** Both reviews.

`dspy/runtime/transparency.py` currently owns DTOs, call-site resolution, LM
config resolution, validation, and logging/reporting side effects. Several
production files import helper functions via the submodule path, while only part
of the module is exported from `dspy/runtime/__init__.py`.

Refactor:

- Split into focused modules or a subpackage:
  - `types.py`: `CompiledCall`, `TransparencyViolation`
  - `resolve.py`: `resolve_call`, `resolve_adapter`, `resolve_lm_config`,
    `resolve_call_site`
  - `validate.py`: pure violation collection and validation
  - `report.py`: warning, verbose logging, and raising behavior
- Re-export the de-facto public helpers from `dspy/runtime/__init__.py` if they
  are intended spine APIs.

### 1.4 Give LM config resolution two explicit names

**Source:** External review, supported by manual review of `runtime/transparency.py`.

`resolve_lm_config` appears to serve both "merge call config with LM defaults"
and "trace provenance for audit" roles. That makes duplicated calls look
accidental at call sites.

Refactor:

- Split into named helpers such as `merge_call_config(lm, config)` and
  `trace_config_provenance(lm, merged_config)`.
- Use the merge-only helper before adapter preprocessing.
- Use provenance tracing only where the compiled/audit call is built.

### 1.5 Relocate `PredictOptions` out of core LM wire types

**Source:** External review.

`dspy/core/types/call_options.py` imports `TaskSpec` and has predict-layer
model-rebuild behavior. That makes core LM type imports carry predict/runtime
contracts.

Refactor:

- Move `ModuleCallOptions` and `PredictOptions` to `dspy/runtime/call_options.py`
  or `dspy/predict/call_options.py`.
- Re-export from `dspy.core.types` temporarily if compatibility requires it.
- Mirror the deferred rebuild pattern used by `run_context_model.py` instead of
  import-time side effects.

### 1.6 Delete or intentionally shim ghost package shells

**Source:** Both reviews.

`dspy/_meta/` and `dspy/transparency/` are empty in tracked source. Local
`__pycache__` artifacts make them look like migrated but still meaningful
packages.

Refactor:

- Delete empty directories and local cache artifacts if no compatibility path is
  required.
- If compatibility is required, add explicit thin re-export shims and document
  canonical locations.

## Tier 2: Runtime, Logging, and Execution Ownership

These changes make operational behavior easier to trace without changing what
gets logged or when execution happens.

### 2.1 Centralize call-log fan-out

**Source:** External review, adjacent to manual review of `RunContext` and
`Module.inspect_call_log`.

Call logs are surfaced on `RunContext`, modules, and LMs. Fan-out reportedly
lives in client code, while inspection helpers are duplicated across runtime,
module, and LM surfaces.

Refactor:

- Add a runtime-owned call-log coordinator, for example
  `dspy/runtime/call_log.py`.
- Put bounded append and fan-out policy in one helper.
- Make `RunContext.inspect_call_log`, `Module.inspect_call_log`, and LM
  inspection delegate to the same implementation.

### 2.2 Split `run_log.py` into paths, redaction, and session I/O

**Source:** Both reviews.

`dspy/runtime/run_log.py` mixes environment/path resolution, redaction, session
creation, and JSONL append behavior.

Refactor:

- `log_paths.py`: `DSPY_LOG_DIR`, `DSPY_RUN_ID`, slugging, and path resolution.
- `log_redaction.py`: pure redaction helpers.
- `run_log_session.py`: `RunLogSession`, session creation, append behavior.

### 2.3 Keep `RunContext` focused on runtime data

**Source:** Both reviews.

`RunContext.create()`, `fork()`, and log-session handling mix config storage with
log lifecycle policy and environment-derived path decisions.

Refactor:

- Move log-session resolution/fork policy behind runtime log helpers.
- Preserve the explicit override validation in `fork()`.
- Lazy-import debug pretty-printing inside `inspect_call_log` so ordinary
  `RunContext` imports do not pull terminal formatting helpers.

### 2.4 Simplify callback dispatch tables

**Source:** Manual review.

`dspy/runtime/callback.py` has duplicated sync/async wrapper logic and branchy
handler lookup for each callback kind. This is readable today but expensive to
modify when adding a callback surface.

Refactor:

- Factor shared wrapper state handling into one helper used by sync and async
  wrappers.
- Replace callback-kind branching with small mapping tables for start/end
  handlers, keeping adapter `format`/`parse` special handling explicit.

## Tier 3: Core Type Flow and Serialization Drift

These changes reduce drift across provider compatibility, config merging, and
core object views.

### 3.1 Consolidate OpenAI serialization and parsing paths

**Source:** Both reviews.

OpenAI-compatible conversion is split across:

- `dspy/core/types/openai_compat.py`
- `dspy/core/types/parts/openai.py`
- `dspy/core/types/parts/serialize.py`
- `dspy/clients/openai_format/serialize.py`

The existence of parity tests is a signal that drift is plausible.

Refactor:

- Establish one canonical serializer for `LMPart` to provider blocks.
- Keep `openai_compat.py` as message assembly and request kwargs only.
- Route tool-call dict shaping through `tool_call_part_to_openai` in one place.

### 3.2 Extract core type submodules by domain concept

**Source:** External review, partially supported by manual review.

Several core files combine multiple public concepts:

- `response.py`: `LMResponse`, `LMOutput`, and `CallRecord`
- `config.py`: generation config, tool spec, and adaptation mode
- `coercion.py`: message normalization and tool coercion
- `stream.py`: sync and async stream wrappers with duplicated state

Refactor:

- Move `CallRecord` to `call_record.py`.
- Move `LMToolSpec` and tool coercion to tool-focused modules.
- Move adaptation mode out of general generation config.
- Extract common stream state used by `LMStream` and `AsyncLMStream`.
- Keep package-barrel re-exports stable during migration.

### 3.3 Replace triplicated merge-overlay logic with one private helper

**Source:** External review.

`merge_lm_config`, `merge_provider_options`, and embedder option merging share
overlay semantics: `model_fields_set`, extension union, right-wins behavior, and
nested model dumping.

Refactor:

- Add a private helper such as `_merge_model_overlay(...)`.
- Keep public merge functions as thin wrappers with domain names.
- Preserve each function's current edge-case semantics with focused tests before
  consolidating.

### 3.4 Make `LMRequestPatch` a Pydantic model

**Source:** External review, aligned with project preference for Pydantic models.

`dspy/core/types/request.py` defines `LMRequestPatch` as a dataclass among
Pydantic core request models.

Refactor:

- Convert it to a frozen or ordinary Pydantic model with the same defaults and
  `merge()` semantics.
- Preserve current list/tuple merge behavior and config overlay behavior.

### 3.5 Reconsider orphaned `dspy/core/hashing.py`

**Source:** External review.

`dspy/core/hashing.py` sits at the core root but appears closer to persistence or
internal utility ownership.

Refactor:

- Move to `dspy/_internal/hashing.py` or `dspy/persistence/hashing.py`, depending
  on its actual consumers.
- Re-export or update callers without changing hash output.

## Tier 4: Task-Spec Internals and Persistence Contracts

These changes clarify the task-spec spine without changing public task-spec
behavior.

### 4.1 Move task-spec validation to `validation.py`

**Source:** External review, supported by manual review.

`parse.py` imports validation helpers from `task_spec.py`, which makes the model
module both the model owner and a validation utility module.

Refactor:

- Move `validate_task_spec` and `validate_task_spec_field_names` to
  `dspy/task_spec/validation.py` or a focused invariants module.
- Keep direction clear: `field_spec` -> validation -> parse/factory/task_spec.

### 4.2 Model serialized task specs explicitly

**Source:** Manual review.

`TaskSpec.from_dict()` and `field_spec_from_dict()` pass raw dictionaries through
procedural helpers. The code is readable, but a strict serialized model would
make schema drift detection more explicit.

Refactor:

- Add strict Pydantic ingest models for serialized task specs and fields.
- Keep `TaskSpec` and `FieldSpec` as the domain models.
- Remove redundant post-construction validation once model validators cover the
  same invariants.

### 4.3 Untangle overlapping task-spec formatting modules

**Source:** External review.

`field_format.py`, `formatting.py`, `annotation_format.py`, and
`json_serialize.py` have overlapping naming or very small scopes.

Refactor:

- Rename `formatting.py` to a more specific name such as
  `field_descriptions.py`, or move adapter-aware pieces out with Tier 1.2.
- Consolidate annotation stringifiers behind one private formatter with thin
  public wrappers.
- Merge the tiny JSON helper into a nearby field-formatting or JSON-coercion
  module if it remains task-spec-owned.

## Tier 5: Primitives and Test Helper Modifiability

These are lower-risk cleanup targets that would make future changes less
surprising.

### 5.1 Split `Module` into graph, state, and execution helpers

**Source:** Both reviews.

`dspy/primitives/module.py` handles async invocation, callbacks, usage tracking,
graph traversal, persistence, batching, LM propagation, and call-log inspection.

Refactor:

- `module_graph.py`: BFS traversal, predictor discovery, submodule discovery.
- `module_state.py`: dump/load/save/load persistence boundary.
- Keep `Module` public methods as thin delegations.
- Keep `batch()` as a thin wrapper around `Parallel` unless there is pressure to
  move it.

### 5.2 Guard dynamic record APIs against method collisions

**Source:** Manual review, adjacent to external record-storage note.

`RecordBacked.__getattribute__` checks the backing store before normal public
attributes. Keys like `keys`, `items`, `to_dict`, or `completions` can mask real
methods and properties.

Refactor:

- Reserve class attributes and public methods from dynamic field lookup, or make
  dynamic lookup a fallback after normal attribute lookup.
- Document intentional differences between `Completions` and the facade pattern.
- Extract shared `dspy_` key filtering helpers for `RecordStore` and facades.

### 5.3 Consolidate interpreter/sandbox layout and response handling

**Source:** Both reviews.

Interpreter code is split between root primitive files and
`python_interpreter/`. JSON-RPC response handling is duplicated between
execution and generic request helpers.

Refactor:

- Consolidate interpreter-related files under a single subpackage such as
  `dspy/primitives/interpreter/`.
- Keep re-exports from `dspy.primitives` and current import paths for API
  stability.
- Extract one JSON-RPC response pump that handles skipped lines, ID checks, tool
  calls, and error translation.

### 5.4 Split `testing/dummy_lm.py` by concern

**Source:** External review, consistent with manual review of test helper
coupling.

`DummyLM` handles answer routing, adapter-specific formatting, dynamic
`FieldSpec` synthesis, and output assembly while importing production adapter
details.

Refactor:

- `_dummy_answers.py`: answer lookup and follow-example behavior.
- `_dummy_format.py`: field spec synthesis and adapter formatting.
- `_dummy_output.py`: `LMOutput` and `LMResponse` assembly.
- Keep `DummyLM` as the thin orchestrator exported from `dspy.testing`.

### 5.5 Unify optional dependency import patterns

**Source:** External review, supported by manual review of `_internal/lazy_import.py`.

The repo has the lazy `require()` pattern and separate eager try/except install
hint patterns. Some integrations reportedly import private helpers.

Refactor:

- Add a public internal helper such as
  `import_optional(top_level, *, extra, feature)` for eager entrypoints.
- Keep `_detect_dspy_dist()` private.
- Migrate integrations incrementally.

## Tier 6: Public Spine Imports and Verification Gaps

These are cleanup and guardrail items to apply while touching the tiers above.

### 6.1 Normalize public imports through package spines

**Source:** External review.

Production code frequently imports deeply from `dspy.core.types.config`,
`dspy.task_spec.field_spec`, and `dspy.runtime.transparency`, creating two
public APIs: package barrels and submodules.

Refactor:

- Re-export stable symbols from package `__init__` files.
- Migrate callers incrementally to documented spine imports.
- Keep direct submodule imports only for genuinely private internals.

### 6.2 Add focused tests when refactoring these areas

**Source:** External review plus manual review additions.

Useful guardrails:

- `Unbatchify` close-after-call and output-count mismatch behavior.
- `run_bounded` with valid `None` results.
- `coerce_tool_spec` for OpenAI wrapper and flat dict inputs.
- Merge behavior for LM, provider, and embedder options.
- `LMStream` and `AsyncLMStream` result/state behavior.
- `LMRequest.from_call` exclusion of messages plus direct-call inputs.
- Task-spec serialized model drift and field default round trips.
- Dynamic record key collisions with public methods.
- GRPO validation behavior under optimized Python.
- Adapter JSON parsing with unexpected output fields.
- Trace capture around concurrent optimizer calls.
- Local finetune server timeout and cleanup behavior.

## Additional Manual Pass: Adapters, Clients, Integrations, Predict, and Teleprompt

These findings come from the follow-up manual review of `dspy/history`,
`dspy/serialization`, `dspy/clients`, `dspy/adapters`, `dspy/integrations`,
`dspy/persistence`, `dspy/predict`, and `dspy/teleprompt`.

### A.1 Replace recursive polling in GRPO

**Source:** Manual review.

`dspy/teleprompt/grpo.py` implements `_wait_until` with recursive async calls.
A long-running GRPO job can build an unbounded call stack while waiting for an
available training batch.

Refactor:

- Replace recursion with a simple `while not predicate(): await sleep(...)` loop.
- Keep the same poll interval and completion behavior.
- Consider adding a future timeout/cancellation hook only if the public optimizer
  contract needs one.

### A.2 Replace GRPO runtime `assert` validation with explicit errors

**Source:** Manual review.

`dspy/teleprompt/grpo.py` uses `assert` for constructor checks, compile-time
input validation, and runtime invariants. These checks disappear under optimized
Python, which would remove user-facing validation and internal safety checks.

Refactor:

- Convert user/config validation to `if ...: raise ValueError` or `TypeError`.
- Convert data-shape and job-state checks to explicit runtime exceptions with the
  existing messages.
- Reserve `assert` only for truly impossible local states that do not protect
  public behavior.

### A.3 Decompose the GRPO compile loop

**Source:** Manual review.

`dspy/teleprompt/grpo.py` has the largest structural concentration in this pass:
`compile()` mixes validation, training-set sampling, trace collection, adapter
resolution, finetune formatting, rollout grouping, pending-batch queue
management, and job stepping.

Refactor:

- Extract a helper for collecting and validating bootstrapped trace data.
- Extract a helper for turning traces into `GRPORolloutGroup` batches.
- Extract a helper for matching pending batch IDs with queued training groups.
- Keep behavior unchanged and test against small deterministic GRPO fixtures
  before and after extraction.

### A.4 Make optimizer trace capture explicit

**Source:** Manual review.

`dspy/teleprompt/bootstrap_trace.py` temporarily monkey-patches
`program._aforward_impl` to capture per-example optimization traces. The mutation
is hidden and fragile around concurrency, nested optimizer calls, and exceptions.

Refactor:

- Introduce a small wrapper/proxy module that delegates to the original program
  and captures traces without mutating the program object.
- Preserve failed-parse capture semantics and trace tuple shape.
- Keep the current `finally` restoration behavior until the wrapper replaces the
  monkey patch.

### A.5 Align JSON adapter parsing with the documented exact-field contract

**Source:** Manual review.

`dspy/adapters/utils/parse.py` documents that adapter parse results must match
the task spec's output field keys exactly. `dspy/adapters/json_adapter.py`,
however, filters unexpected fields before validation, so extra LM output keys are
silently discarded instead of reported as schema drift.

Refactor:

- Decide whether unexpected JSON fields are accepted or rejected.
- If rejected, validate before filtering and keep `validate_parsed_fields` as the
  single contract check.
- If accepted, update the parse helper documentation so adapter behavior is not
  contradictory.

### A.6 Narrow transient sampling exception handling

**Source:** Manual review.

`dspy/predict/sampling.py` catches `BaseException` and then filters retryability.
That catches cancellation and interpreter-control exceptions before re-raising
them.

Refactor:

- Catch `Exception` instead of `BaseException`.
- Keep `is_transient_sampling_error` as the retry classifier.
- Preserve `SamplingExhaustedError` chaining from the last transient exception.

### A.7 Isolate sync GEPA bridges

**Source:** Manual review.

`dspy/integrations/optimizers/gepa/adapter.py` calls async DSPy flows through
`asyncio.run` from sync GEPA methods. This fails if GEPA is invoked inside an
already-running event loop and makes the async boundary easy to miss.

Refactor:

- If GEPA supports async hooks, make the DSPy adapter async end-to-end.
- If GEPA must remain sync, isolate `asyncio.run` in a clearly named sync bridge
  helper and document that it requires no running event loop.
- Keep `RunContext` requirements explicit at the adapter boundary.

### A.8 Split local finetune process lifecycle from provider behavior

**Source:** Manual review.

`dspy/integrations/finetune/local.py` mixes provider behavior, subprocess launch,
server readiness polling, log buffering, tokenizer setup, and training. Its
`wait_for_server` timeout check only runs after an HTTP response, so repeated
connection failures can wait indefinitely.

Refactor:

- Move launch/process/log handling into a focused local-server helper.
- Make the timeout check unconditional in the polling loop.
- Keep provider methods as thin orchestration around launch, kill, and training
  helpers.

### A.9 Guarantee BetterTogether LM cleanup

**Source:** Manual review.

`dspy/teleprompt/bettertogether.py` launches LMs before baseline evaluation and
kills them after the strategy loop. If baseline evaluation or early setup fails,
launched local LMs may not be killed.

Refactor:

- Wrap the launch/evaluation/strategy block in `try/finally`.
- Keep the existing `flag_lms_launched` behavior, but ensure cleanup runs for
  baseline failures as well as later optimizer failures.

### A.10 Add a typed response boundary for Databricks retrieval

**Source:** Manual review.

`dspy/integrations/retrieval/databricks.py` has SDK and requests paths that both
return raw response dictionaries. The requests path manually reads JSON without
`raise_for_status`, and response validation is spread through the query method.

Refactor:

- Add a small parser/helper for the expected Databricks vector-search response
  shape.
- Call `raise_for_status()` in the requests path before parsing JSON.
- Share column validation and passage conversion between SDK and requests paths.

## Recommended Sequence

1. Harden `Unbatchify` and `run_bounded`.
2. Delete or intentionally shim empty `_meta` and `transparency` package shells.
3. Break the `core` -> `clients` dependency.
4. Break the `task_spec` -> `adapters` dependency.
5. Split transparency and rename LM config resolution responsibilities.
6. Relocate `PredictOptions`.
7. Centralize call-log fan-out and split run-log helpers.
8. Split `Module`, interpreter layout, and core type submodules.
9. Harden GRPO polling, validation, and compile-loop structure.
10. Replace monkey-patched trace capture with an explicit wrapper boundary.
11. Align adapter JSON parse validation with the documented field contract.
12. Fix local finetune and BetterTogether lifecycle cleanup.
13. Unify optional dependency imports and normalize spine re-exports.

The first five items provide the highest leverage for using this repository as a
foundational component: they remove hidden layer coupling, clarify ownership,
and reduce the chance that future adapter/client work destabilizes the spine.
