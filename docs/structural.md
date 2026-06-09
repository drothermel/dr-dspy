# Structural Review: Spine Submodules

Manual structural review of `dspy/_internal`, `dspy/_meta`, `dspy/core`,
`dspy/primitives`, `dspy/runtime`, `dspy/task_spec`, `dspy/testing`, and
`dspy/transparency`.

Additional manual pass covers `dspy/history`, `dspy/serialization`,
`dspy/clients`, `dspy/adapters`, `dspy/integrations`, `dspy/persistence`,
`dspy/predict`, and `dspy/teleprompt`.

Additional manual test-suite pass covers `dspy/testing` and all tracked Python
tests under `tests/`.

Additional external cross-package review covers `dspy/history`,
`dspy/serialization`, `dspy/clients`, `dspy/adapters`, `dspy/persistence`,
`dspy/predict`, `dspy/integrations`, and `dspy/teleprompt`.

Scope: behavior-preserving improvements to boundaries, flow, simplicity, and
ease of modification. This is not a feature-change list or a mechanical style
backlog. Behavior-changing notes live in `docs/behavioral.md`.

Source tags:

- **Both reviews**: independently identified in both manual passes.
- **Manual review**: identified in the Codex manual pass in this thread.
- **External review**: identified in the provided second-agent review.
- **Spine parallel review**: identified in the parallel-agent pass over the
  spine submodules; each item carrying this tag was re-verified against source
  before inclusion.
- **External cross-package review**: identified in the follow-up external
  findings covering the non-spine submodules and tests.

Path note: `dspy/_meta/` and `dspy/transparency/` have no tracked Python source
in this checkout. They are empty package shells with local `__pycache__` only.
The live transparency implementation is `dspy/runtime/transparency.py`; metadata
appears to live outside the reviewed paths.

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

### 1.7 Move adapter call validation out of `predict`

**Source:** External cross-package review.

`AdapterCallPipeline` imports `validate_task_inputs` from
`dspy.predict.call_validation`, but most of that logic is adapter/spine work:
reserved-key handling, `validate_task_inputs_from_spec`, and agent-history
coercion. This makes adapter code depend upward on predict code.

Refactor:

- Split predict-only reserved-key and `PredictOptions` handling from generic task
  input validation.
- Move generic validation to `dspy.task_spec` or a small spine module such as
  `dspy.call_validation`.
- Have predict and adapters import the shared validation module instead of
  adapters importing from predict.

### 1.8 Move trace-running primitives below `predict` and `teleprompt`

**Source:** External cross-package review.

`dspy/predict/sampling.py` and `dspy/predict/refine.py` import
`run_program_with_trace` from `dspy.teleprompt.trace_helpers`. That makes core
predict utilities depend on optimizer infrastructure.

Refactor:

- Move the lightweight `run_program_with_trace` helper to `dspy.runtime` or
  `dspy.history`, since it primarily forks `RunContext.optimization_trace`.
- Keep teleprompt trace-to-demo utilities in teleprompt if they remain
  optimizer-specific.
- Update predict and teleprompt to depend downward on the shared runtime/history
  helper.

### 1.9 Clarify GEPA ownership between integrations and teleprompt

**Source:** External cross-package review.

`dspy/integrations/optimizers/gepa/adapter.py` imports teleprompt internals such
as bootstrap tracing, task-spec context, and evaluator helpers, while GEPA's
user-facing teleprompter also lives under `dspy/teleprompt`. Ownership is split:
integration code is acting as both leaf integration and optimizer core.

Refactor:

- Extract shared compile/trace primitives into a documented module such as
  `dspy.teleprompt.core`, `dspy.compile`, or runtime-owned helpers.
- Keep external-package adaptation in `dspy.integrations`.
- Keep user-facing optimizer orchestration in `dspy.teleprompt`.

### 1.10 Decouple client finetune entrypoints from integration providers

**Source:** External cross-package review.

`dspy/clients/finetune/lm.py` imports Databricks, local, and OpenAI provider
implementations from `dspy.integrations.finetune`. That means client-layer import
paths can pull optional integration code and provider-specific dependencies.

Refactor:

- Use lazy provider lookup keyed by provider name, for example
  `infer_provider("openai")`.
- Let integrations register providers at import time or through entry points.
- Keep `dspy.clients` free of hard imports from optional integration packages.

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

### 2.5 Move `ACTIVE_RUN` ownership to the run-context module

**Source:** Spine parallel review.

`ACTIVE_RUN` is the ambient run `ContextVar`. It is defined in
`dspy/runtime/callback.py` (alongside `ACTIVE_CALL_ID`, which genuinely belongs to
callbacks) but is *written* in `dspy/primitives/module.py` `__call__`. Run-state
ownership is therefore split across modules, and `module.py` imports a "callback"
symbol to manage run scope.

Refactor:

- Move `ACTIVE_RUN` to `dspy/runtime/run_context.py` (the owner of the
  `RunContext` lifecycle) and have `callback.py` import it.
- Keep `ACTIVE_CALL_ID` in `callback.py`.

### 2.6 Make per-call run state safe-by-construction or document the fork precondition

**Source:** Spine parallel review.

`RunContext.caller_modules` is a mutable list, and `track_usage` swaps
`run.usage_tracker` in place. Both assume strict LIFO single-task nesting. This is
currently safe only because every concurrent fan-out forks first
(`run_program_with_trace` -> `RunContext.fork()` resets `caller_modules=[]`), so
the evaluator and optimizer paths are not corrupted. The invariant is enforced by
convention, not by the type system: a future caller that runs `module(run=...)`
concurrently on a shared, unforked `RunContext` would interleave the
append/pop stack and the tracker swap.

Refactor:

- Prefer task-local per-call state (a `ContextVar[tuple[Module, ...]]` set
  immutably) over mutating a list on the shared model, or
- Document the "fork before concurrent fan-out" precondition at the `RunContext`
  and `module.__call__` boundaries and add a test that asserts a forked run gets a
  fresh `caller_modules`.

## Tier 3: Core Type Flow and Serialization Drift

These changes reduce drift across provider compatibility, config merging, and
core object views.

### 3.1 Consolidate OpenAI serialization and parsing paths

**Source:** Both reviews; reinforced by external cross-package review.

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
- Have logging views delegate to the same OpenAI-format owner so observability
  does not maintain a parallel serializer.

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

### 3.4 Decide whether `LMRequestPatch` should exist before converting it

**Source:** External review; refined by spine parallel review.

`dspy/core/types/request.py` defines `LMRequestPatch` as a dataclass among
Pydantic core request models. The spine pass found it has **no production
consumers** anywhere in `dspy/` — the only references are in
`tests/core/types/test_config.py`. Its `*_parts`, `messages`, and `delete_*`
fields are never read. Converting dead code to Pydantic would polish an
abstraction that nothing uses.

Refactor:

- First confirm against the adapter/predict layers that `LMRequestPatch` and its
  `merge()` are genuinely unused in production. If so, delete it (and its test)
  rather than converting it.
- Only if a real consumer exists or is planned: convert it to a frozen Pydantic
  model with the same defaults and list/tuple `merge()`/config-overlay semantics.

### 3.5 Trim the dead instance API on `dspy/core/hashing.py`, then reconsider its home

**Source:** External review; refined by spine parallel review.

`dspy/core/hashing.py` sits at the core root but appears closer to persistence or
internal utility ownership. The spine pass also found that only the `Hasher`
classmethods (`hash`, `hash_bytes`) are used (in finetune and bootstrap); the
**instance API** (`__init__`, `update`, `hexdigest`, `self.m`) has no callers, and
`update` double-hashes (hashes the pickle, hex-encodes, then re-hashes the hex
string).

Refactor:

- Reduce `Hasher` to the two classmethods (or a module-level `hash_value`
  function) and drop the unused instance machinery.
- Then move it to `dspy/_internal/hashing.py` or `dspy/persistence/hashing.py`
  depending on its actual consumers, re-exporting or updating callers without
  changing hash output.

### 3.6 Remove the vestigial positional-tools return in `coercion.py`

**Source:** Spine parallel review.

`dspy/core/types/coercion.py` `_messages_from_items` is typed
`tuple[list[LMMessage], list[Any]]` (messages plus positional tools), but every
return path returns `[]` for the second element, and `request.py` `from_call`
extends `collected_tools` with this always-empty list. The "positional tools from
items" feature is unimplemented and the plumbing is misleading.

Refactor:

- Drop the always-empty second return value and the `positional_tools` plumbing in
  `from_call`, simplifying the signature to `list[LMMessage]`.
- If positional tools are genuinely planned, leave a `# TODO` naming the intended
  behavior instead of dead structure that looks wired up.

### 3.7 Consolidate recursive JSON serialization

**Source:** External cross-package review.

JSON-safe conversion is split across `dspy/serialization/json.py`,
`dspy/task_spec/json_serialize.py`, history model serializers, OpenAI media
payload dumps, and scattered `model_dump(mode="json")` calls. The fallbacks and
recursion semantics differ: some paths use `repr`, some use `str`, and circular
reference handling only exists in one helper.

Refactor:

- Make one recursive helper, likely `to_jsonable`, the canonical JSON-safe
  conversion boundary.
- Have task-spec and history serialization delegate to it unless a field
  intentionally emits a human-formatted string.
- Remove silent fallback differences or make them visible through a narrow log or
  explicit documented boundary.

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

**Source:** External review, consistent with manual test-suite review of test
helper coupling.

`DummyLM` handles answer routing, adapter-specific formatting, dynamic
`FieldSpec` synthesis, and output assembly while importing production adapter
details.

Refactor:

- `_dummy_answers.py`: answer lookup and follow-example behavior.
- `_dummy_format.py`: field spec synthesis and adapter formatting.
- `_dummy_output.py`: `LMOutput` and `LMResponse` assembly.
- Keep `DummyLM` as the thin orchestrator exported from `dspy.testing`.

### 5.5 Unify optional dependency import patterns

**Source:** External review and external cross-package review, supported by
manual review of `_internal/lazy_import.py`.

The repo has the lazy `require()` pattern and separate eager try/except install
hint patterns. Some integrations reportedly import private helpers.

Refactor:

- Add a public internal helper such as
  `import_optional(top_level, *, extra, feature)` for eager entrypoints.
- Keep `_detect_dspy_dist()` private.
- Migrate integrations incrementally.
- Route dataset, Optuna, Databricks, OpenAI, Weaviate, SGLang, and inline
  optional-import guards through the same helper instead of hand-rolled variants.

### 5.6 Unify history truncation helpers

**Source:** External cross-package review.

`dspy/history/truncation.py` has near-identical
`call_with_turn_log_truncation` and `call_with_repl_history_truncation` retry
loops. They differ mostly by history type, log messages, and result model names;
`REPLHistoryCallResult` also still names its history field `turn_log`.

Refactor:

- Add one generic `call_with_history_truncation(...)` helper over the existing
  history protocol shape.
- Put `truncate_oldest` or equivalent behavior on the shared history protocol if
  truncation remains protocol-owned.
- Export `REPLHistoryModule` from `dspy/history/__init__.py` alongside
  `TurnLogModule`.
- Preserve current truncation, retry, and error messages unless renaming fields
  is explicitly accepted as a cleanup.

### 5.7 Extract a shared agent loop skeleton

**Source:** External cross-package review.

`ReAct`, `ReActV2`, `CodeAct`, `Avatar`, and parts of `RLM` repeat the same
control-flow scaffolding: tool normalization, synthetic finish/submit tools,
turn-log truncation, termination-reason branching, and task-spec instruction
assembly. The duplicated loops have already drifted in robustness and
termination handling.

Refactor:

- Add a non-opinionated `AgentLoopRunner` or similar helper under
  `dspy/predict/agent_loop.py`.
- Let each agent supply step execution, tool wiring, and output extraction while
  the runner owns iteration, truncation, and termination dispatch.
- Decide and document which ReAct implementation is canonical if both remain
  public.

### 5.8 Unify trace-capture mechanisms

**Source:** External cross-package review; overlaps with A.4.

Teleprompt has a lightweight trace runner (`trace_helpers.run_program_with_trace`)
and a heavier bootstrap path that monkey-patches `_aforward_impl` to capture
parse failures. Bootstrap, GEPA, and GRPO use the heavy path; predict sampling
uses the light path.

Refactor:

- Colocate both implementations and document when each is required, or
  consolidate behind one trace runner with an option such as
  `capture_parse_failures`.
- Keep the current failed-parse capture semantics available for optimizers that
  depend on them.
- Avoid making new optimizer code choose between two similarly named trace paths
  without a contract.

### 5.9 Centralize persistence ownership

**Source:** External cross-package review.

Whole-program pickle persistence lives under `dspy/persistence`, while JSON
state save/load behavior and some metadata handling live on
`dspy/primitives/module.py`. Metadata assembly, pickle warnings, dependency
version drift handling, and `Any` return typing make the persistence boundary
harder to reason about.

Refactor:

- Build a single persistence facade with focused submodules such as
  `program.py`, `state.py`, and `metadata.py`.
- Make `Module.save` and `Module.load` thin delegates.
- Type the load boundary as narrowly as possible, for example `load() -> Module`
  where that is the actual contract.

### 5.10 Move `Parallel` toward runtime execution infrastructure

**Source:** External cross-package review.

`Parallel` is exported beside predictor modules, but it is not a `Module` and
does not participate in `named_predictors()` or `dump_state`. `Module.batch`
also constructs `Parallel` internally, which makes it look like execution
infrastructure rather than a predictor.

Refactor:

- Move or re-export `Parallel` from a runtime/execution namespace such as
  `dspy.runtime.batch` or `dspy.execute`.
- Keep compatibility imports during migration if needed.
- Document that `Parallel` is batch execution infrastructure, not a model
  program component.

## Tier 6: Public Spine Imports and Verification Gaps

These are cleanup and guardrail items to apply while touching the tiers above.

### 6.1 Normalize public imports through package spines

**Source:** External review and external cross-package review.

Production code frequently imports deeply from `dspy.core.types.config`,
`dspy.task_spec.field_spec`, and `dspy.runtime.transparency`, creating two
public APIs: package barrels and submodules.

Refactor:

- Re-export stable symbols from package `__init__` files.
- Migrate callers incrementally to documented spine imports.
- Keep direct submodule imports only for genuinely private internals.
- Decide which package barrels are intentionally empty, curated public surfaces,
  or internal-only. `adapters`, `clients`, `integrations`, `predict`, and
  `teleprompt` currently send mixed signals.
- Split teleprompt public optimizer exports from internal composition helpers
  such as trace/demo conversion utilities if those helpers are not meant as user
  API.

### 6.2 Prefer adapter composition over inheritance for shared formatting behavior

**Source:** External cross-package review.

`JSONAdapter`, `ChatAdapter`, `XMLAdapter`, parse fallback behavior, and shared
formatting mixins form an inheritance/composition web where format and parse
changes can ripple across adapters unexpectedly.

Refactor:

- Extract shared formatting collaborators such as `ChatFieldFormatter` where
  behavior is not fundamentally adapter-subtype inheritance.
- Keep parse fallback as an explicit injectable strategy, building on the
  existing `call/policies` shape.
- Preserve existing adapter classes as thin delegators during migration.

### 6.3 Make pipeline-only adapters explicit in the type story

**Source:** External cross-package review.

`TwoStepAdapter.parse` intentionally raises instead of implementing the same
parse contract as direct-parse adapters. The runtime behavior may be deliberate,
but the uniform `Adapter` interface makes the exception surprising.

Refactor:

- Split adapter protocols into direct-parse and pipeline-only capabilities, or
  add a typed marker/mixin for adapters that do not support standalone `parse`.
- Keep the runtime exception if needed, but make it part of the public contract.

### 6.4 Model agent history events by agent contract

**Source:** External cross-package review.

`TurnEvent` is a broad union-of-all-agents model with fields for ReAct, ReActV2,
CodeAct, Avatar, RLM, and `extra="allow"`. The docstring carries the per-agent
contract, but the type does not enforce it.

Refactor:

- Consider per-agent event models under a shared envelope or a discriminated
  union.
- Preserve wire JSON compatibility through serializers if this becomes a larger
  migration.
- At minimum, move repeated terminal tool names and event-key literals into
  constants shared by agent modules.

### 6.5 Standardize TaskSpec placement for framework and optimizer tasks

**Source:** External cross-package review.

TaskSpec subclasses and framework specs are spread across predict modules,
optimizer modules, GEPA integration files, and adapter base modules. That makes
it hard to discover which package owns framework-level prompts.

Refactor:

- Establish a placement convention, such as `task_specs.py` per optimizer
  subpackage or `dspy/task_spec/framework/` for shared framework specs.
- Keep adapter-owned task specs with adapters when they are truly adapter
  boundary contracts.
- Avoid mixing framework prompt specs into unrelated runtime code paths.

### 6.6 Normalize dataset integration shapes

**Source:** External cross-package review.

Dataset integrations use mixed patterns: some helpers are generic, some classes
subclass the spine `Dataset`, and some standalone dataset classes carry inline
metrics. That makes the canonical loader style unclear.

Refactor:

- Prefer one pattern, ideally spine `Dataset` subclasses plus shared metric
  registration where metrics are part of the dataset contract.
- Keep lightweight helper functions only when they do not model a dataset
  lifecycle.

### 6.7 Reduce compile-time mutation of optimizer instances

**Source:** External cross-package review.

Several optimizers stash compile-local values such as `trainset`, `valset`,
`run`, or `student` on `self` during `compile()`. That makes optimizer instances
not re-entrant and awkward to reason about in parallel or repeated compiles.

Refactor:

- Introduce compile-local session models passed through private helpers.
- Keep optimizer instances responsible for configuration, not per-compile
  mutable state.
- Document the input-program ownership contract: which optimizers mutate the
  student and which operate on copies.

### 6.8 Inject replaceable DrLlm pool session resolution

**Source:** External cross-package review.

`DrLlmPoolLM` session identity is coupled to runtime logging state:
`resolve_pool_session_id` reads `run.log_session` and run-log bucket helpers.
That makes session identity harder to test without a fully configured logging
`RunContext`.

Refactor:

- Accept an optional `session_id_resolver: Callable[[RunContext], str]` or small
  protocol at construction.
- Keep the current log-session-derived behavior as the default resolver.
- Unit-test pool session identity through the injected resolver rather than
  through disk logging setup.

### 6.9 Add focused tests when refactoring these areas

**Source:** External review plus manual review additions and manual test-suite
review additions, with external cross-package review additions.

Useful guardrails:

- `coerce_tool_spec` for OpenAI wrapper and flat dict inputs.
- Merge behavior for LM, provider, and embedder options.
- `LMRequest.from_call` exclusion of messages plus direct-call inputs.
- Task-spec serialized model drift and field default round trips.
- Dynamic record key collisions with public methods.
- Trace capture around concurrent optimizer calls.
- Live-provider tests remain skipped unless the relevant opt-in marker is passed.
- `DummyLM` response routing and output assembly remain stable after any helper
  split.

## Tier 7: Test Suite Structure and Test Helper Contracts

These findings come from the manual review of `dspy/testing` and the tracked
Python tests under `tests/`. They focus on making the test suite easier to
modify without weakening the behavior it protects.

### 7.1 Put every live-provider test behind the same opt-in marker

**Source:** Manual test-suite review.

Most live LLM tests use `@pytest.mark.llm_call`, which is skipped by default in
`tests/conftest.py`. `tests/primitives/test_module.py::test_usage_tracker_in_parallel`
is guarded only by `OPENAI_API_KEY`, so a developer with credentials can run a
paid live call in the default suite.

Refactor:

- Mark all live-provider tests with `llm_call`, including tests that already
  have credential checks.
- Keep environment checks as secondary skip reasons for opted-in live runs.
- Prefer shared live-provider fixtures for model names, credentials, and skip
  messages so the policy is visible in one place.

### 7.2 Decide whether `dspy.testing` is public API or repo-local test support

**Source:** Manual test-suite review.

`dspy.testing` is packaged with `dspy`, but its contents read like internal test
infrastructure. That creates a public-looking surface without a clear contract.

Refactor:

- If it is public, document `DummyLM` and `DummyVectorizer` as supported testing
  APIs and give their accepted input shapes explicit types.
- If it is repo-local, move the helpers under `tests/test_utils` and update
  imports.
- Avoid leaving helper behavior half-public, where tests and users can depend on
  undocumented internals.

### 7.3 Consolidate LM test doubles

**Source:** Manual test-suite review.

The test suite defines several LM doubles and wrappers across different files:
`DummyLM`, adapter `CapturingLM`, client typed-contract LMs, two-step recording
LMs, and many local `DummyLM` subclasses. This makes new tests copy local
patterns instead of choosing from a small, known set of doubles.

Refactor:

- Add a focused `tests/test_utils/lm_doubles.py` module, or formalize the same
  concepts under `dspy.testing` if they are meant to be package-level helpers.
- Provide explicit doubles for common roles: sequential text response, typed
  `LMRequest` recorder, provider-capability wrapper, native tool-call response,
  and failing LM.
- Keep adapter-specific capture helpers close to adapter tests only when they
  depend on adapter-private details.

### 7.4 Extract reusable adapter formatting scenarios

**Source:** Manual test-suite review.

The adapter tests contain very large exact-message snapshots, especially in
`tests/adapters/test_chat_adapter.py`, `tests/adapters/test_json_adapter.py`,
and `tests/adapters/test_baml_adapter.py`. These tests are valuable, but each
scenario currently mixes fixture construction, domain model definitions, input
data, expected messages, and assertion logic in one function.

Refactor:

- Extract shared scenario builders for recurring QA, history, tool, image,
  document, and Pydantic-model cases.
- Keep exact full-string snapshots for true prompt contracts.
- For broad "kitchen sink" coverage, assert normalized message structure or
  targeted field fragments where exact text is not the behavior under review.
- Keep generated expected messages in plain Python data, not external snapshot
  files, unless snapshot tooling is introduced deliberately.

### 7.5 Centralize pytest marker definitions and skip policy

**Source:** Manual test-suite review.

`integration` is defined in `pyproject.toml`, while `reliability`, `extra`,
`llm_call`, and `deno` are added dynamically in `tests/conftest.py`. The runtime
behavior works, but the available test categories are not fully discoverable
from configuration.

Refactor:

- Define all markers and short descriptions in `pyproject.toml`.
- Keep `tests/conftest.py` responsible for CLI opt-in flags and skip behavior.
- Update `tests/README.md` with one table of default-skipped categories, required
  dependencies or credentials, and example commands.

### 7.6 Split Databricks live integration tests from unit tests

**Source:** Manual test-suite review.

`tests/clients/test_databricks.py` is effectively a live/manual integration
module. It imports the Databricks SDK, instantiates a workspace client at import
time, and includes hard-coded workspace paths.

Refactor:

- Move live Databricks tests under an explicitly marked integration path or mark
  every live test with an opt-in marker.
- Replace hard-coded workspace paths with environment-configured values.
- Add mocked unit tests for path validation, provider request construction, and
  deploy/fine-tune orchestration so default tests still prove local behavior.

## Additional Manual Pass: Adapters, Clients, Integrations, Predict, and Teleprompt

These findings come from the follow-up manual review of `dspy/history`,
`dspy/serialization`, `dspy/clients`, `dspy/adapters`, `dspy/integrations`,
`dspy/persistence`, `dspy/predict`, and `dspy/teleprompt`.

### A.1 Decompose the GRPO compile loop

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

### A.2 Make optimizer trace capture explicit

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

### A.3 Isolate sync GEPA bridges

**Source:** Manual review.

`dspy/integrations/optimizers/gepa/adapter.py` calls async DSPy flows through
`asyncio.run` from sync GEPA methods. This fails if GEPA is invoked inside an
already-running event loop and makes the async boundary easy to miss.

Refactor:

- If GEPA supports async hooks, make the DSPy adapter async end-to-end.
- If GEPA must remain sync, isolate `asyncio.run` in a clearly named sync bridge
  helper and document that it requires no running event loop.
- Keep `RunContext` requirements explicit at the adapter boundary.

### A.4 Consolidate Optuna ask/tell orchestration

**Source:** External cross-package review.

`dspy/integrations/optimizers/optuna/study.py` exposes a shared
`run_ask_tell_loop`, while MIPRO search reimplements manual `study.ask()` /
`study.tell()` flow.

Refactor:

- Route MIPRO search through the shared ask/tell helper.
- Keep MIPRO-specific objective and callback logic local to the MIPRO package.

### A.5 Split BaseLM logging and state serialization

**Source:** External cross-package review.

`BaseLM` owns forward entry, capability properties, usage tracking, memory and
disk call logging, run-log record assembly, dynamic deserialization, copy, and
inspect helpers. This makes logging hard to test without involving the LM
runtime surface.

Refactor:

- Extract a call-log writer/coordinator for in-memory and disk append behavior.
- Extract an LM state serializer/loader for dynamic import and copy behavior.
- Keep `BaseLM` focused on forward execution and capability surface.

### A.6 Split adapter call preprocessing into small policies

**Source:** External cross-package review.

`AdapterCallMixin._call_preprocess` acts as a switchboard for native function
calling, tool stripping, parallel tool choice, reasoning/citation adaptation,
and task-spec mutation.

Refactor:

- Introduce a small chain of preprocessors keyed by adapter capabilities and
  field types, following the existing `call/policies` style.
- Keep each preprocessor independently testable.
- Preserve the current order of mutations until tests prove equivalent behavior.

### A.7 Separate LM inference from finetune lifecycle surface

**Source:** External cross-package review.

The primary `LM` class carries inference behavior alongside finetune lifecycle
methods such as launch, kill, finetune, and reinforce. Call sites that only infer
therefore inherit finetune API surface and optional-provider concerns.

Refactor:

- Move finetune behavior behind a `FinetuneFacet`, `FinetuneLM`, or provider
  service object.
- Keep the inference `LM` surface focused on request execution and state.
- Preserve existing public methods during migration only if compatibility is
  required.

### A.8 Normalize evaluator construction in optimizers

**Source:** External cross-package review.

`make_optimizer_evaluator` and `resolve_max_errors` exist as shared optimizer
helpers, but GRPO constructs `Evaluate` directly in multiple places and random
search rebuilds an evaluator inside a candidate loop where nothing per-candidate
changes.

Refactor:

- Route GRPO through `make_optimizer_evaluator` and `resolve_max_errors`.
- Hoist random-search evaluator construction out of the candidate loop when the
  evaluator inputs are invariant.

### A.9 Use typed optimizer candidate models consistently

**Source:** External cross-package review.

COPRO uses raw dict candidate records plus custom equality and duplicate
helpers, while other optimizers use typed candidate models such as
`ProgramCandidate`.

Refactor:

- Migrate COPRO candidate records to the shared typed candidate model or a
  COPRO-specific Pydantic model with the same semantics.
- Delete dict-specific duplicate/equality helpers after callers move.

### A.10 Move generic LM error extraction out of LiteLLM-specific modules

**Source:** External cross-package review.

`dr_llm/errors.py` imports private helpers from `clients/lm/errors.py` to
extract exception messages, statuses, and status-class mappings. Those helpers
are generic error-boundary utilities, not LiteLLM-only concepts.

Refactor:

- Move generic extraction helpers to a shared `dspy/clients/errors.py` module.
- Let LiteLLM and dr-llm backends import from the shared boundary.
- Keep backend-specific error translation in each backend package.

## Positive Patterns to Preserve

**Source:** External cross-package review.

Keep these shapes intact while refactoring:

- Immutable, copy-on-append history models (`TurnLog`, `REPLHistory`).
- Runtime-checkable history protocols that avoid inheritance coupling.
- `AdapterCallPipeline` / `PreparedAdapterCall` as a prepare -> invoke ->
  postprocess seam.
- Policy objects for response-format and parse-fallback behavior.
- Thin `Predict._aforward_impl` delegation to the adapter boundary.
- Teleprompter registration with typed compile params.
- Consistent `CompileResult` and `ProgramCandidate` return shapes.
- LM transport routing by model type.
- Lazy optional-dependency guards with feature-specific messages.
- MIPRO's multi-phase subpackage decomposition.
