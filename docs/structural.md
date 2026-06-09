# Structural Review: Prioritized Change Chunks

## P0: Re-establish Layer Direction and Ownership

These are the highest-priority structural chunks because they determine where
future code should be added. Complete them before large local refactors so
subsequent work lands in the right package.

### P0.1 Core, client, and OpenAI wire-format boundary

**Status:** Done (2026-06).

**Sources:** Both reviews; External cross-package review.

Problem:

- `dspy/core/types/openai_compat.py` and
  `dspy/core/types/parts/openai.py` import client-layer helpers.
- OpenAI-compatible serialization is spread across core part serializers and
  `dspy/clients/openai_format`.
- Logging views can drift from provider request formatting because they maintain
  parallel OpenAI-compatible serializers.

Target shape:

- Keep core types provider-neutral.
- Move direction-neutral helpers such as media URI parsing into a core-owned
  module.
- Make `dspy.clients.openai_format` the single owner of OpenAI wire-format
  request and response conversion.
- Keep `openai_compat.py` as a thin compatibility/request-view layer that
  delegates instead of duplicating provider block conversion.

Details to preserve:

- Existing history/live serializer parity tests are the migration guardrail.
- Tool-call dict shaping should route through one canonical conversion helper.
- Observability should use the same serializer as provider request assembly.

### P0.2 TaskSpec, adapters, and call-validation boundary

**Status:** Done (2026-06).

**Sources:** Both reviews; External cross-package review.

Problem:

- `dspy/task_spec` imports adapter-specific types and prompt-formatting helpers.
- `AdapterCallPipeline` imports task-input validation from
  `dspy.predict.call_validation`.
- TaskSpec subclasses for framework and optimizer prompts are scattered across
  predict modules, teleprompt modules, integrations, and adapter code.

Target shape:

- Keep task specs responsible for field contracts and generic validation.
- Move adapter-aware prompt descriptions and field-type rendering to adapter
  modules.
- Move generic task-input validation to `dspy.task_spec` or a small spine module
  such as `dspy.call_validation`.
- Keep predict-only reserved-key and `PredictOptions` handling separate from
  adapter/spine validation.
- Establish a TaskSpec placement convention, such as `task_specs.py` per
  optimizer package or `dspy/task_spec/framework/` for shared framework specs.

Details to preserve:

- Adapter-owned specs such as boundary contracts can stay with adapters.
- Public field validation semantics should remain unchanged during the move.
- Field description strictness remains a TaskSpec contract, not adapter policy.

### P0.3 Predict, teleprompt, GEPA, and trace ownership

**Sources:** External cross-package review; Manual review.

Problem:

- `dspy/predict/sampling.py` and `dspy/predict/refine.py` import
  `run_program_with_trace` from `dspy.teleprompt.trace_helpers`.
- GEPA integration code imports teleprompt internals while GEPA's user-facing
  optimizer also lives in `dspy.teleprompt`.
- Trace capture has two paths: a lightweight run-context fork helper and a
  heavier bootstrap path that monkey-patches program internals.

Target shape:

- Move lightweight trace-running primitives below predict and teleprompt, likely
  into runtime or history.
- Extract shared compile/trace primitives used by GEPA and teleprompt into a
  documented owner such as `dspy.teleprompt.core`, `dspy.compile`, or runtime
  helpers.
- Keep external-package adaptation in `dspy.integrations`.
- Colocate or unify trace capture behind one explicit API with a flag such as
  `capture_parse_failures`.

Details to preserve:

- Failed-parse capture semantics remain available for bootstrap, GEPA, and GRPO.
- Teleprompt-specific trace-to-demo helpers can remain optimizer-owned.
- New optimizer code should not need to choose between similarly named trace
  helpers without a documented contract.

### P0.4 Clients, integrations, and finetune provider ownership

**Sources:** External cross-package review; Manual review.

Problem:

- `dspy/clients/finetune/lm.py` imports Databricks, local, and OpenAI finetune
  providers from `dspy.integrations`.
- The primary `LM` class carries inference behavior plus launch, kill, finetune,
  and reinforce lifecycle surface.
- Generic error extraction helpers are imported from LiteLLM-specific modules by
  dr-llm code.

Target shape:

- Keep `dspy.clients` free of hard imports from optional integration packages.
- Use lazy provider lookup or registration keyed by provider name.
- Move finetune behavior behind a facet, wrapper, or provider service object so
  the inference `LM` remains focused on request execution and state.
- Move generic LM error extraction to a shared `dspy/clients/errors.py` module.

Details to preserve:

- Backend-specific error translation stays in each backend package.
- Provider registration should preserve current provider names and inference.
- Existing public finetune methods can be kept as thin delegates only if
  compatibility is required.

## P1: Stabilize Runtime, Logging, and Execution Infrastructure

These chunks are next because they define the operational spine used by modules,
LMs, evaluators, and optimizers.

### P1.1 Split transparency and call-resolution responsibilities

**Sources:** Both reviews; External review.

Problem:

- `dspy/runtime/transparency.py` owns DTOs, call-site resolution, LM config
  resolution, validation, warning behavior, and raising/logging side effects.
- `resolve_lm_config` appears to mix "merge call config with LM defaults" and
  "trace provenance for audit" roles.
- `PredictOptions` lives in core LM wire types but carries predict/runtime
  contracts.

Target shape:

- Split transparency into focused modules:
  - `types.py`: `CompiledCall`, `TransparencyViolation`
  - `resolve.py`: call, adapter, LM config, and call-site resolution
  - `validate.py`: pure violation collection
  - `report.py`: warning, verbose logging, and raising behavior
- Rename config helpers by job, for example `merge_call_config(...)` and
  `trace_config_provenance(...)`.
- Move `ModuleCallOptions` and `PredictOptions` to runtime or predict, with
  package re-exports only where compatibility requires them.

Details to preserve:

- Existing strict-transparency semantics should not change.
- De-facto public helpers imported through submodule paths need explicit export
  decisions before callers move.

### P1.2 Centralize run logs, call logs, and inspection

**Sources:** Both reviews; External review.

Problem:

- Call logs are surfaced on `RunContext`, modules, and LMs.
- Fan-out and append behavior are tied to client code, while inspection helpers
  are duplicated across runtime, module, and LM surfaces.
- `dspy/runtime/run_log.py` mixes environment/path resolution, redaction, session
  creation, and JSONL append behavior.

Target shape:

- Add a runtime-owned call-log coordinator for bounded append and fan-out.
- Make `RunContext.inspect_call_log`, `Module.inspect_call_log`, and LM
  inspection delegate to one implementation.
- Split run-log helpers by responsibility:
  - `log_paths.py`: `DSPY_LOG_DIR`, `DSPY_RUN_ID`, slugging, path resolution
  - `log_redaction.py`: pure redaction helpers
  - `run_log_session.py`: session creation and JSONL append behavior

Details to preserve:

- Current call-log records and append timing should remain stable.
- Debug pretty-printing can be lazy-imported so ordinary `RunContext` imports do
  not pull terminal formatting helpers.

### P1.3 Make ambient run state ownership explicit

**Sources:** Both reviews; Spine parallel review.

Problem:

- `RunContext.create()`, `fork()`, and log-session handling mix config storage
  with log lifecycle policy and environment-derived path decisions.
- `ACTIVE_RUN` is defined in `dspy/runtime/callback.py` but written by
  `dspy/primitives/module.py`.
- `RunContext.caller_modules` and `usage_tracker` rely on strict LIFO,
  fork-before-concurrency discipline.

Target shape:

- Keep `RunContext` focused on runtime data.
- Move log-session resolution and fork policy behind runtime log helpers.
- Move `ACTIVE_RUN` to `dspy/runtime/run_context.py`; keep callback-specific
  `ACTIVE_CALL_ID` in `callback.py`.
- Either make per-call state task-local and immutable, or document and test the
  "fork before concurrent fan-out" precondition.

Details to preserve:

- `fork()` explicit override validation should remain.
- Existing evaluator and optimizer paths that already fork should stay
  semantically unchanged.

### P1.4 Simplify callback and batch execution infrastructure

**Sources:** Manual review; External cross-package review.

Problem:

- Callback dispatch has duplicated sync/async wrapper logic and branchy handler
  lookup.
- `Parallel` is exported beside predictor modules even though it is execution
  infrastructure, not a `Module`.
- `Module.batch` constructs `Parallel` internally, reinforcing that distinction.

Target shape:

- Factor shared callback wrapper state handling into one helper used by sync and
  async wrappers.
- Replace callback-kind branching with small start/end handler tables while
  keeping adapter `format` and `parse` special handling explicit.
- Move or re-export `Parallel` from a runtime/execution namespace such as
  `dspy.runtime.batch` or `dspy.execute`.

Details to preserve:

- Existing callback hook names and event order should remain unchanged.
- Compatibility imports for `Parallel` may be useful during migration.

## P2: Consolidate Serialization, Persistence, and Core Data Contracts

This priority reduces multiple sources of truth after the main layer boundaries
are clear.

### P2.1 Consolidate recursive JSON serialization

**Sources:** External cross-package review.

Problem:

- JSON-safe conversion is split across `dspy/serialization/json.py`,
  `dspy/task_spec/json_serialize.py`, history serializers, OpenAI media payload
  dumps, and scattered `model_dump(mode="json")` calls.
- Fallbacks and recursion semantics differ across paths.

Target shape:

- Make one recursive helper, likely `to_jsonable`, the canonical JSON-safe
  conversion boundary.
- Have task-spec and history serialization delegate to it unless a field
  intentionally emits a human-formatted string.
- Document or narrow any fallback that intentionally emits `str()` or `repr()`.

Details to preserve:

- Circular-reference handling should remain available.
- Human-readable formatted strings should be explicit exceptions, not accidental
  serializer drift.

### P2.2 Split core type modules by domain concept

**Sources:** External review; Spine parallel review.

Problem:

- `response.py`, `config.py`, `coercion.py`, and `stream.py` each mix several
  public concepts.
- Merge-overlay logic is triplicated across LM config, provider options, and
  embedder options.
- `LMRequestPatch` appears unused in production.
- `dspy/core/hashing.py` exposes an unused instance API.
- `coercion.py` returns an always-empty positional-tools value.

Target shape:

- Move `CallRecord`, tool specs/coercion, adaptation mode, and stream shared
  state into focused modules.
- Add a private `_merge_model_overlay(...)` helper and keep public merge
  functions as thin domain wrappers.
- Confirm whether `LMRequestPatch` has production consumers; delete it if not,
  rather than converting dead code.
- Reduce `Hasher` to the used classmethods or a module-level helper, then move
  it to internal or persistence ownership.
- Remove unused positional-tool plumbing unless the feature is actually planned.

Details to preserve:

- Package-barrel re-exports should keep public imports stable during migration.
- Hash output must remain unchanged if callers still depend on it.
- Merge helper consolidation needs focused tests to lock current edge cases.

### P2.3 Make TaskSpec serialization and formatting contracts explicit

**Sources:** External review; Manual review.

Problem:

- `parse.py` imports validation helpers from `task_spec.py`.
- `TaskSpec.from_dict()` and `field_spec_from_dict()` parse raw dictionaries via
  procedural helpers.
- `field_format.py`, `formatting.py`, `annotation_format.py`, and
  `json_serialize.py` have overlapping names and scopes.

Target shape:

- Move validation helpers to `dspy/task_spec/validation.py` or a focused
  invariants module.
- Add strict Pydantic ingest models for serialized task specs and fields.
- Keep `TaskSpec` and `FieldSpec` as the domain models.
- Rename or merge formatting helpers once adapter-aware pieces have moved out.

Details to preserve:

- Field default round trips and fingerprint behavior should stay unchanged.
- Avoid duplicating validation once ingest-model validators cover the same
  invariants.

### P2.4 Centralize persistence ownership

**Sources:** External cross-package review.

Problem:

- Whole-program pickle persistence lives under `dspy/persistence`.
- JSON state save/load behavior and some metadata handling live on
  `dspy/primitives/module.py`.
- Metadata assembly, pickle warnings, dependency-version drift handling, and
  broad load typing make the persistence boundary harder to reason about.

Target shape:

- Build one persistence facade with focused submodules such as `program.py`,
  `state.py`, and `metadata.py`.
- Make `Module.save` and `Module.load` thin delegates.
- Type the load boundary as narrowly as the actual contract allows, for example
  `load() -> Module`.

Details to preserve:

- Existing file formats and dependency-version warnings should remain stable.
- Behavior-changing schema hardening belongs in `docs/behavioral.md`.

## P3: Extract Shared Agent and Optimizer Control Flow

These chunks are high leverage but should come after boundaries and runtime
ownership are clearer, because they touch broad execution paths.

### P3.1 Unify history truncation and agent loop scaffolding

**Sources:** External cross-package review.

Problem:

- `call_with_turn_log_truncation` and `call_with_repl_history_truncation` are
  near-identical retry loops.
- `ReAct`, `ReActV2`, `CodeAct`, `Avatar`, and parts of `RLM` repeat tool
  normalization, synthetic finish/submit tools, truncation, termination
  branching, and task-spec instruction assembly.

Target shape:

- Add one generic `call_with_history_truncation(...)` helper over the existing
  history protocol shape.
- Put truncation capability on the shared history protocol if truncation remains
  protocol-owned.
- Export `REPLHistoryModule` alongside `TurnLogModule`.
- Add a non-opinionated `AgentLoopRunner` under `dspy/predict/agent_loop.py`.

Details to preserve:

- Preserve current truncation, retry, and termination semantics.
- Agent modules should supply step execution, tool wiring, and output extraction;
  the runner should own iteration and termination dispatch.
- Decide and document which ReAct implementation is canonical if both remain
  public.

### P3.2 Decompose GRPO and optimizer trace orchestration

**Sources:** Manual review; External cross-package review.

Problem:

- `dspy/teleprompt/grpo.py` `compile()` mixes validation, sampling, trace
  collection, adapter resolution, finetune formatting, rollout grouping, pending
  batch queue management, and job stepping.
- Bootstrap trace capture mutates program internals by monkey-patching
  `_aforward_impl`.
- GEPA sync integration calls async DSPy flows through `asyncio.run`.

Target shape:

- Extract GRPO helpers for collecting and validating bootstrapped trace data,
  turning traces into rollout groups, and matching pending batch IDs to training
  groups.
- Replace monkey-patching with a wrapper/proxy module that delegates and captures
  traces explicitly.
- If GEPA must remain sync, isolate `asyncio.run` in a named sync bridge with a
  clear "no running event loop" contract.

Details to preserve:

- Keep failed-parse trace capture semantics and trace tuple shape.
- Test GRPO extraction against small deterministic fixtures before and after the
  split.
- Keep `RunContext` requirements explicit at optimizer/integration boundaries.

### P3.3 Normalize optimizer shared helpers and candidate models

**Sources:** External cross-package review.

Problem:

- Optuna ask/tell orchestration is shared in one package and reimplemented in
  MIPRO search.
- `make_optimizer_evaluator` and `resolve_max_errors` exist, but some optimizers
  construct `Evaluate` directly or rebuild invariant evaluators in loops.
- COPRO uses raw dict candidate records while other optimizers use typed
  candidate models.
- Several optimizers stash compile-local values on `self`.

Target shape:

- Route MIPRO search through the shared ask/tell helper.
- Route GRPO through `make_optimizer_evaluator` and `resolve_max_errors`.
- Hoist invariant evaluator construction out of candidate loops.
- Migrate COPRO candidate records to `ProgramCandidate` or a COPRO-specific
  Pydantic model with the same semantics.
- Introduce compile-local session models passed through private helpers.

Details to preserve:

- Optimizer instances should hold configuration, not mutable per-compile state.
- Document which optimizers mutate the input student and which operate on
  copies.

## P4: Clarify Public API, Imports, and Package Organization

These chunks reduce confusion once the main owners and shared helpers exist.

### P4.1 Normalize package spines and empty barrels

**Sources:** External review; External cross-package review; Both reviews.

Problem:

- Production code imports deeply from submodules such as
  `dspy.core.types.config`, `dspy.task_spec.field_spec`, and
  `dspy.runtime.transparency`.
- Package barrels send mixed signals: some are empty, some expose user APIs, and
  some mix public optimizers with internal helpers.
- Empty `_meta` and `transparency` package shells look migrated but meaningful.

Target shape:

- Re-export stable public symbols from documented package spines.
- Migrate callers incrementally to documented imports.
- Keep direct submodule imports for private implementation details.
- Delete ghost package shells, or add intentional compatibility shims if the
  import path must survive.
- Split teleprompt public optimizer exports from internal composition helpers if
  those helpers are not intended user API.

Details to preserve:

- Import churn should be staged after owner modules are in their target homes.
- Public-vs-internal decisions should be documented in package placement and
  exports, not only by leading underscores.

### P4.2 Move adapter reuse from inheritance toward explicit composition

**Sources:** External cross-package review.

Problem:

- `JSONAdapter`, `ChatAdapter`, `XMLAdapter`, parse fallback behavior, and shared
  formatting mixins form an inheritance/composition web.
- `TwoStepAdapter.parse` intentionally raises despite the uniform adapter
  interface.
- `AdapterCallMixin._call_preprocess` is a switchboard for native function
  calling, tool stripping, parallel tool choice, reasoning/citation adaptation,
  and task-spec mutation.

Target shape:

- Extract shared formatting collaborators such as `ChatFieldFormatter`.
- Keep parse fallback as an explicit injectable strategy, building on the
  existing `call/policies` style.
- Split adapter protocols into direct-parse and pipeline-only capabilities, or
  add a typed marker/mixin for adapters that do not support standalone `parse`.
- Introduce a chain of small preprocessors keyed by adapter capabilities and
  field types.

Details to preserve:

- Existing adapter classes can remain as thin delegators during migration.
- Preserve current preprocessing order until tests prove equivalent behavior.

### P4.3 Model agent events, datasets, and integration shapes consistently

**Sources:** External cross-package review.

Problem:

- `TurnEvent` is a broad union-of-all-agents model with `extra="allow"`.
- Dataset integrations mix generic helpers, spine `Dataset` subclasses, and
  standalone classes with inline metrics.
- `DrLlmPoolLM` session identity is coupled to runtime logging state.

Target shape:

- Consider per-agent event models under a shared envelope or discriminated union.
- Move repeated terminal tool names and event-key literals into shared constants.
- Pick one dataset integration pattern, preferably spine `Dataset` subclasses
  plus shared metric registration where metrics are part of the contract.
- Inject an optional `session_id_resolver` or small protocol into
  `DrLlmPoolLM`, defaulting to current log-session behavior.

Details to preserve:

- Preserve wire JSON compatibility if event modeling becomes a migration.
- Keep lightweight dataset helpers only when they do not model dataset
  lifecycle.

### P4.4 Unify optional dependency import patterns

**Sources:** External review; External cross-package review.

Problem:

- The repo has both `_internal/lazy_import.py` and several bespoke eager
  try/except install-hint patterns.
- Dataset, Optuna, Databricks, OpenAI, Weaviate, SGLang, and inline optional
  import guards do not present one consistent failure style.

Target shape:

- Add a public internal helper such as
  `import_optional(top_level, *, extra, feature)` for eager entrypoints.
- Keep distribution-detection internals private.
- Migrate optional imports incrementally through the same helper.

Details to preserve:

- Feature-specific install hints are valuable and should stay.
- Optional dependencies should remain lazy where they are currently lazy.

## P5: Simplify Primitives, Test Helpers, and Test Suite Structure

These are lower priority than the architecture spine, but they improve
day-to-day modification once the main boundaries are stable.

### P5.1 Split primitive god modules and dynamic record helpers

**Sources:** Both reviews; Manual review.

Problem:

- `dspy/primitives/module.py` handles async invocation, callbacks, usage
  tracking, graph traversal, persistence, batching, LM propagation, and call-log
  inspection.
- `RecordBacked.__getattribute__` checks the backing store before normal public
  attributes, allowing keys like `keys`, `items`, or `to_dict` to mask methods.
- Interpreter code is split between root primitive files and
  `python_interpreter/`, with duplicated JSON-RPC response handling.

Target shape:

- Split module internals into graph, state, and execution helpers while keeping
  `Module` public methods as thin delegations.
- Reserve class attributes and public methods from dynamic field lookup, or make
  dynamic lookup a fallback after normal attribute lookup.
- Consolidate interpreter files under one subpackage and extract one JSON-RPC
  response pump.

Details to preserve:

- Keep `Module.batch()` as a thin `Parallel` wrapper unless the runtime move
  creates a better import path.
- Keep current primitive import paths re-exported while internals move.

### P5.2 Decide the status of `dspy.testing` and consolidate test doubles

**Sources:** External review; Manual test-suite review.

Problem:

- `dspy.testing` is packaged with `dspy`, but its contents read like internal
  test infrastructure.
- `DummyLM` mixes answer routing, adapter-specific formatting, dynamic
  `FieldSpec` synthesis, and output assembly.
- The test suite defines many local LM doubles and wrappers.

Target shape:

- Decide whether `dspy.testing` is public supported API or repo-local test
  support.
- If public, document `DummyLM` and `DummyVectorizer` and type their supported
  input shapes.
- If repo-local, move helpers under `tests/test_utils` and update imports.
- Split `DummyLM` into answer lookup, formatting, and output assembly helpers.
- Add a small catalog of LM doubles for sequential text responses, typed request
  recording, provider-capability wrapping, native tool-call responses, and
  failures.

Details to preserve:

- Adapter-specific capture helpers can stay near adapter tests when they depend
  on adapter internals.
- Avoid leaving helper behavior half-public and undocumented.

### P5.3 Make adapter tests reusable without weakening prompt contracts

**Sources:** Manual test-suite review.

Problem:

- Adapter tests contain large exact-message snapshots that mix scenario setup,
  domain model definitions, inputs, expected messages, and assertions.

Target shape:

- Extract shared scenario builders for QA, history, tool, image, document, and
  Pydantic-model cases.
- Keep exact full-string snapshots where prompt text is the behavior under
  review.
- Use normalized message structure or targeted field fragments for broad
  kitchen-sink coverage.

Details to preserve:

- Expected messages can stay as plain Python data unless snapshot tooling is
  introduced deliberately.
- Reuse should make new adapter scenarios smaller, not hide the contract being
  asserted.

### P5.4 Normalize pytest marker and live integration policy

**Sources:** Manual test-suite review.

Problem:

- Live-provider tests are not all guarded by the same opt-in marker.
- Marker definitions are split between `pyproject.toml` and dynamic registration
  in `tests/conftest.py`.
- Databricks tests are effectively live/manual integration tests but live among
  unit-style client tests.

Target shape:

- Mark every live-provider test with the same opt-in marker, including tests
  that already check credentials.
- Define all markers and descriptions in `pyproject.toml`.
- Keep `tests/conftest.py` responsible for CLI opt-in flags and skip behavior.
- Move or mark Databricks live integration tests explicitly.
- Add mocked Databricks unit tests for path validation, request construction, and
  deploy/fine-tune orchestration.

Details to preserve:

- Environment checks remain useful as secondary skip reasons after opt-in.
- `tests/README.md` should explain default-skipped categories, dependencies,
  credentials, and example commands.
