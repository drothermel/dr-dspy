# Prompt: Make dr-dspy Ready for Updated dr-llm Experiments

You are a smart implementation agent working in the `dr-dspy` repository. Use this document as your running prompt and status tracker. Your goal is to make this repository fully ready to build and run experiments on top of the updated `dr-llm` backends, using either `DrLlmDirectLM` or `DrLlmPoolLM`, while also fixing the DSPy-facing defects that would block realistic optimizer, evaluation, and experiment workflows.

Read this entire document before editing code. Also read `docs/final_review_drllm.md` before making changes because several dr-dspy fixes must preserve compatibility with the reviewed dr-llm behavior and request contracts.

## How to Work

1. Update the status table in this document as you go.
   - Set a row to `In progress` before editing that area.
   - Set it to `Done` only after code, tests, docs, and verification are complete.
   - Set it to `Blocked` only when a concrete external dependency or product decision is required.
   - Add short dated notes in the `Notes` column when useful.
2. Keep changes surgical. Preserve public behavior, diagnostics, and local patterns unless the task explicitly asks for a behavior change.
3. Prefer Pydantic `BaseModel` over dataclasses for new structured data.
4. Use the async-only public API. Pass `run=` explicitly to modules, evaluation, optimizers, adapters, and task calls.
5. Define new DSPy tasks as `TaskSpec` subclasses or via `make_task_spec`, not legacy `Signature`.
6. When using dr-llm LMs, assume v1 text-only support unless you explicitly add and test broader contract support.
7. Before committing, run the required repo gates in order:

```bash
uv run ruff check --fix
uv run ty check --fix
uv run ruff format
uv run python scripts/check_lazy_imports.py
```

All four commands must exit 0. Re-run the full sequence after any fixes.

## Desired End State

At the end of this process, `dr-dspy` should be able to run a new experiment immediately with updated `dr-llm` by using:

- `DrLlmDirectLM` for text-only `Predict`, `ChainOfThought`, `Evaluate`, and optimizer task execution through explicit `RunContext.create(lm=..., adapter=...)`.
- `DrLlmPoolLM` for cache-first single completions and explicit-session acquisition where no-replacement sampling is intentional.
- Clear docs and tests for what is supported, what is rejected, and how provider-specific dr-llm controls affect exact experiment parity.

Do not claim exact reproduction of existing `nl_latents` pool curves through `Predict(TaskSpec)` unless a raw single-message request path and provider-control parity are implemented and verified. The `nl_latents` stack used raw dr-llm requests and a separate pool-grid harness.

## Status Tracker

| ID | Area | Status | Notes |
| --- | --- | --- | --- |
| P0-1 | Public import cycles | Not started | |
| P0-2 | SIMBA / `Parallel(access_examples=False)` | Not started | |
| P0-3 | Legacy agent truncation handling | Not started | |
| P0-4 | Bootstrap threshold semantics | Not started | |
| P0-5 | `collect_trace_data` async/module metrics | Not started | |
| P0-6 | GEPA multimodal/custom proposer `run=` threading | Not started | |
| P0-7 | Unique disk call-log sessions | Not started | |
| P0-8 | dr-llm safe state loading | Not started | |
| P0-9 | dr-llm pool session identity | Not started | |
| P0-10 | dr-llm provider-specific reasoning/config bridge | Not started | |
| P0-11 | dr-llm pool lifecycle | Not started | |
| P1-1 | OpenAI reasoning-model validation | Not started | |
| P1-2 | LiteLLM provider options behavior | Not started | |
| P1-3 | Strict transparency first-run ergonomics | Not started | |
| P1-4 | Generated-code execution timeout | Not started | |
| P1-5 | Finetuning poll and HTTP timeouts | Not started | |
| P1-6 | `run_with_trace` module spine observability | Not started | |
| P1-7 | `run_bounded` abort accounting | Not started | |
| P1-8 | Parallel and optimizer call-log isolation | Not started | |
| P1-9 | `Evaluate` empty devsets | Not started | |
| P1-10 | Empty answer-list metrics | Not started | |
| P1-11 | Sampling threshold miss metadata | Not started | |
| P1-12 | Sampling trace mutation | Not started | |
| P1-13 | JSON parsing strictness | Not started | |
| P1-14 | MCP mixed tool-result conversion | Not started | |
| P1-15 | Persistence failure modes | Not started | |
| P2-1 | Pool aggregate provenance | Not started | |
| P2-2 | Pool fingerprint and metadata docs | Not started | |
| P2-3 | dr-llm `n=1` contract | Not started | |
| P2-4 | dr-llm v1 scope docs/tests | Not started | |
| P2-5 | `nl-code` TaskSpec experiment scaffold | Not started | |
| P2-6 | Advanced provider option gaps | Not started | |
| P2-7 | Smaller DSPy footguns | Not started | |
| V-1 | Final verification matrix | Not started | |

## Cross-Repo Context from `docs/final_review_drllm.md`

Coordinate these assumptions with the updated `dr-llm` repo:

- `DrLlmPoolLM` currently maps DSPy `LMRequest` to dr-llm `BackendRequest` fingerprints. Existing `nl_latents` pools use a different grid-seeding system and will not produce cache hits through `DrLlmPoolLM`.
- Exact `nl_latents` compression reproduction should remain dr-llm-native until dr-dspy can carry provider-specific reasoning objects and raw single-message prompt requests.
- Updated dr-llm can express provider-native controls used by T1 compression configs:
  - OpenRouter reasoning disabled: `{"kind": "openrouter", "enabled": false}`.
  - OpenRouter provider-specific effort: `{"kind": "openrouter", "effort": "low"}`.
  - OpenAI minimal thinking: `{"kind": "openai", "thinking_level": "minimal"}`.
  - Google thinking off: provider-specific thinking control from the dr-llm Google request-control layer.
- DSPy `ReasoningEffort` currently maps to generic `BackendRequest.effort`; that is not equivalent to provider-specific `BackendRequest.reasoning`.
- Pool acquisition no-replacement semantics depend on a stable session identity. Metadata does not isolate cache keys or claim cells because dr-llm fingerprints exclude metadata and extensions.
- `PoolBackend.submit_batch` plus drain remains a dr-llm-native prefill workflow. `DrLlmPoolLM` should document or expose only the workflow it actually supports.
- dr-llm v1 is text-only. Rejecting tools, multimodal parts, unsupported roles, structured response formats, stop sequences, logprobs, prompt cache, and unsupported reasoning fields is expected unless the dr-llm contract changes.

## Implementation Workstreams

### P0-1: Fix Public Import Cycles

Problem: Fresh-process public imports fail:

```python
from dspy.primitives import Module
from dspy.evaluate.evaluator import Evaluate
```

Known cycle:

`dspy.primitives.__getattr__("Module")` -> `dspy.primitives.module` -> `dspy.primitives.module_graph` -> `dspy.predict.protocol`, which triggers eager imports in `dspy.predict.__init__` and re-enters `dspy.predict.predict` before `Module` is available.

References:

- `dspy/primitives/__init__.py`
- `dspy/primitives/module_graph.py`
- `dspy/predict/__init__.py`

Required outcome:

- Documented public imports work in a clean process.
- Lazy imports remain compatible with `scripts/check_lazy_imports.py`.
- Add clean-process tests that would have failed before the fix.

### P0-2: Fix SIMBA and `Parallel(access_examples=False)`

Problem: `Parallel(access_examples=False)` passes `run=` to callables that do not accept it. SIMBA wraps programs in a plain async function that accepts only `example`, then passes it through this path. For real modules, the path calls `module(example, run=...)`, but `Module.__call__` expects task inputs as keyword arguments.

References:

- `dspy/runtime/batch.py`
- `dspy/teleprompt/simba_utils.py`

Required outcome:

- SIMBA `compile()` does not fail with `TypeError: wrapped_program() got an unexpected keyword argument 'run'`.
- `access_examples=False` has explicit, tested semantics for plain callables and modules.
- Add a focused SIMBA or `Parallel` regression test.

### P0-3: Align Legacy Agent Truncation Handling

Problems:

- Legacy `ReAct` does not assign `turn_log = extracted.turn_log` after `call_with_history_truncation`, so it can append to a pre-truncation log.
- `TruncationExhaustedError` subclasses `ValueError`, so legacy `ReAct` can catch it as a parse error.
- Legacy `ReAct`, `CodeAct`, and `Avatar` do not consistently handle truncation exhaustion around post-loop extract calls.

References:

- `dspy/predict/react.py`
- `dspy/history/truncation.py`
- `dspy/predict/code_act.py`
- `dspy/predict/avatar/avatar.py`

Required outcome:

- Legacy agent truncation behavior matches the safer `ReActV2` pattern.
- Truncation exhaustion is not misclassified as a parse failure.
- Add regression tests for loop and post-loop extraction paths.

### P0-4: Fix Bootstrap Threshold Semantics

Problem: Bootstrap treats `metric_threshold=0.0` as no threshold because it uses truthiness:

```python
success = metric_val >= self.metric_threshold if self.metric_threshold else metric_val
```

Reference:

- `dspy/teleprompt/bootstrap.py`

Required outcome:

- Threshold checks use `is not None`.
- Scores of exactly `0.0` and thresholds of `0.0` are handled intentionally.
- Add tests for `metric_threshold=0.0`.

### P0-5: Make `collect_trace_data` Use Normal Metric Invocation

Problem: `collect_trace_data` wraps metrics synchronously and calls `metric(example, prediction, trace)` directly. This bypasses `invoke_metric` support for async metrics and `Module` metrics.

References:

- `dspy/teleprompt/core/trace_collection.py`
- `dspy/evaluate/metric_invoke.py`

Required outcome:

- `collect_trace_data` delegates through the normal metric invocation path.
- Async metrics and module metrics work consistently with `Evaluate` and direct bootstrap paths.
- Add regression tests for async and `Module` metrics.

### P0-6: Thread `run=` Through GEPA Proposal Paths

Problems:

- `SingleComponentMultiModalProposer` calls `Predict` without `run=`.
- `MultiModalInstructionProposer` calls a module wrapper without `run=`.
- The custom proposer branch creates `opt_run` but does not pass it to the proposer.

References:

- `dspy/integrations/optimizers/gepa/instruction_proposal.py`
- `dspy/integrations/optimizers/gepa/adapter.py`

Required outcome:

- Strict transparency works for multimodal GEPA instruction proposals and custom async proposers.
- `AsyncProposalFn` call sites receive the configured optimizer run.
- Add focused strict-transparency tests.

### P0-7: Make Disk Call-Log Sessions Unique

Problem: `create_run_log_session` uses second-resolution UTC timestamps and `exist_ok=True`, so independent runs created in the same second can share `run.json` and `calls.jsonl`. `DrLlmPoolLM` may derive session IDs from this timestamp.

References:

- `dspy/runtime/run_log_session.py`
- `dspy/clients/dr_llm/pool.py`

Required outcome:

- Multiple `RunContext.create(...)` calls in the same second create distinct log directories.
- Existing log path readability remains straightforward.
- dr-llm pool session derivation does not collide for independent disk-logged runs.
- Add a collision regression test.

### P0-8: Allow Safe State Loading for Builtin dr-llm LMs

Problem: `DrLlmDirectLM` and `DrLlmPoolLM` are listed as builtin LM classes and dump normal state under `dspy.clients.dr_llm.*`, but `BaseLM.load_state` rejects every class path except `dspy.clients.lm.LM` unless `allow_custom_lm_class=True`.

References:

- `dspy/clients/lm_registry.py`
- `dspy/clients/base_lm.py`
- `dspy/clients/dr_llm/base.py`

Required outcome:

- Saved programs using builtin dr-llm LMs reload through the normal safe path.
- Custom LM loading remains protected.
- Add direct and pool `dump_state`/`load_state` round-trip tests.

### P0-9: Make Pool Acquisition Session Identity Explicit and Safe

Problem: `resolve_pool_session_id` uses disk log session when present, then LM-level `session_id`, then a fresh `uuid4()` per call. This can either collide when disk sessions collide or reset no-replacement semantics across repeated calls when no stable session is configured.

References:

- `dspy/runtime/run_log_session.py`
- `dspy/clients/dr_llm/pool.py`
- `tests/clients/dr_llm/test_integration_pool.py`

Required outcome:

- No-session fallback behavior is explicit and tested.
- Repeated `acquire_samples(...)` calls have intentional no-replacement semantics only when a stable session is configured.
- Docs tell experiment authors to pass an explicit stable session identity unless disk logging is known-safe.
- Direct `aforward` remains cache-first and unaffected.

### P0-10: Bridge Provider-Specific dr-llm Reasoning and Config Controls

Problem: Last-week compression configs rely on provider-native dr-llm controls that current dr-dspy cannot represent. Current mapping places DSPy `reasoning.effort` on generic `BackendRequest.effort` and hard-codes `BackendRequest.reasoning=None`.

Required controls to support or explicitly reject with clear guidance:

- OpenRouter reasoning disabled.
- OpenRouter provider-specific effort.
- OpenAI minimal thinking.
- Google thinking off.
- dr-llm `EffortSpec.MAX` if the updated dr-llm contract exposes it for relevant providers.
- Explicit empty sampling controls when needed to suppress provider defaults.

References:

- `dspy/clients/dr_llm/base.py`
- `dspy/clients/dr_llm/mapping.py`
- `dspy/clients/dr_llm/contract.py`
- `../nl_latents/src/nl_latents/sampling/llm/catalog.py`
- `../dr-llm/src/dr_llm/backends/models.py`
- `../dr-llm/src/dr_llm/llm/config.py`
- `../dr-llm/src/dr_llm/llm/providers/concepts/reasoning.py`
- `../dr-llm/src/dr_llm/llm/providers/impls/openrouter/request_controls.py`
- `../dr-llm/src/dr_llm/llm/providers/impls/openai/request_controls.py`
- `../dr-llm/src/dr_llm/llm/providers/impls/google/request_controls.py`

Required outcome:

- Decide and implement a clear public surface for dr-llm-native provider controls, or document a deliberate non-goal and preserve typed rejection.
- Default T1 compression configs can be represented faithfully through dr-dspy only if provider-specific request parity is implemented.
- Add mapping and contract tests that compare expected `BackendRequest` fields for the default T1 model-control cases.
- Do not silently approximate provider-specific reasoning with generic `effort`.

### P0-11: Fix `DrLlmPoolLM` Lifecycle

Problems:

- `BaseLM.copy()` is shallow, so copied `DrLlmPoolLM` wrappers share `_backend` while `_closed` remains per wrapper.
- Closing a copy and then the original can close the same backend twice.
- `aforward` and `acquire_samples` do not check `_closed` before delegating to the backend.

References:

- `dspy/clients/base_lm.py`
- `dspy/clients/dr_llm/pool.py`
- `dspy/predict/sampling.py`
- `dspy/teleprompt/simba_utils.py`
- `../dr-llm/src/dr_llm/backends/pool.py`

Required outcome:

- Pool LM copies cannot accidentally double-close or use a torn-down backend.
- Calls after close fail clearly.
- If dr-llm `PoolBackend.close()` is made idempotent, still keep the wrapper state coherent.
- Add focused lifecycle tests.

### P1-1: Fix OpenAI Reasoning-Model Validation

Problem: The reasoning-model constructor guard uses truthiness, so `temperature=0.0` passes even though only `1.0` or `None` should be accepted. `LM.copy(temperature=0.0)` can bypass the reasoning-specific constructor rule.

References:

- `dspy/clients/lm/client.py`
- `dspy/clients/base_lm.py`

Required outcome:

- Validate `temperature is not None and temperature != 1.0`.
- Validate `max_tokens is not None and max_tokens < 16000` where that rule applies.
- `LM.copy(...)` preserves reasoning-model validation.
- Add tests for constructor and copy paths.

### P1-2: Wire or Clarify LiteLLM Provider Options

Problem: `LMProviderOptions.cache` and `max_retries` are currently no-ops on the LiteLLM path. The client always passes no-cache/no-store, and only `LM.num_retries` controls retries.

References:

- `dspy/clients/lm/client.py`
- `dspy/core/types/lm_provider.py`

Required outcome:

- Either wire these options through correctly or reject/document unsupported combinations clearly.
- Strict transparency audit data should not claim a setting that is ignored at execution.
- Add tests for cache and retry behavior or validation.

### P1-3: Improve Strict Transparency First-Run Ergonomics

Problem: `TelemetryConfig.transparency` defaults to strict. Bare `LM("openai/gpt-4o-mini")` leaves `temperature`, `max_tokens`, and `provider_options.cache` unset, so adapter calls can fail unless callers construct a fully explicit `RunContext`. dr-llm LMs cannot use `provider_options`, which makes cache-related checks awkward.

References:

- `dspy/runtime/config.py`
- `dspy/runtime/transparency/validate.py`
- `dspy/clients/lm/client.py`
- `dspy/clients/dr_llm/base.py`

Required outcome:

- First-run strict transparency errors are actionable.
- Examples and constructors make required configuration obvious.
- dr-llm strict transparency checks align with the dr-llm configuration surface.

### P1-4: Add Timeout Protection to Generated-Code Execution

Problem: `PythonInterpreter.execute` synchronously waits on Deno/Pyodide response loops. `read_until_response` caps skipped non-JSON lines but has no wall-clock timeout. `CodeAct`, `ProgramOfThought`, and RLM call this path directly.

References:

- `dspy/primitives/python_interpreter/interpreter.py`
- `dspy/primitives/python_interpreter/pump.py`
- `dspy/predict/code_act.py`
- `dspy/predict/program_of_thought.py`
- `dspy/predict/rlm/execution.py`

Required outcome:

- Generated code cannot hang an async DSPy run indefinitely.
- Timeout behavior is explicit and tested.

### P1-5: Add Bounded Finetuning Waits and HTTP Timeouts

Problems:

- Databricks finetuning poll loop exits only on `"Completed"` or `"Failed"` and has no training timeout.
- Several Databricks deployment HTTP requests omit request timeouts.
- OpenAI finetune `wait_for_job` treats unknown statuses as pending with no max duration.

References:

- `dspy/integrations/finetune/databricks.py`
- `dspy/integrations/finetune/openai.py`

Required outcome:

- Poll loops are bounded and report unknown/stuck states clearly.
- External HTTP requests have explicit timeouts.
- Add tests with fake provider clients.

### P1-6: Route `run_with_trace` Through the Normal Module Spine

Problem: `run_with_trace` calls `program.aforward(...)` directly instead of `await program(...)`, skipping module call-scope setup, callbacks, and top-level usage tracking.

Reference:

- `dspy/runtime/optimization_trace.py`

Required outcome:

- Tracing preserves normal module observability.
- Bootstrap, SIMBA, Refine, and trace collection still work.
- Add a callback or usage-tracking regression test.

### P1-7: Improve `run_bounded` Abort Accounting

Problem: When `max_errors` triggers cancellation, never-started items remain pending and are finalized as `None`. They are absent from `stats.failed_indices` and `exceptions_map`, so `Parallel` does not record them as failures.

References:

- `dspy/runtime/async_parallel.py`
- `dspy/runtime/batch.py`

Required outcome:

- Aborted, never-started, failed, and successful items are represented distinctly.
- `Parallel` exposes structured partial state without hiding missing work.
- Add cancellation/accounting tests.

### P1-8: Isolate or Document Concurrent Call Logs

Problems:

- `fork_worker_run` isolates worker `RunContext` state, but workers share LM and program objects.
- `record_call` appends to shared bounded ring buffers on `lm.call_log` and module `call_log` without per-worker isolation.
- Optimizer forks preserve shared call-log state rather than isolating teacher or candidate run logs.

References:

- `dspy/runtime/run_fork.py`
- `dspy/runtime/call_log/coordinator.py`
- `dspy/runtime/batch.py`
- `dspy/runtime/run_context.py`
- `dspy/runtime/run_log_policy.py`

Required outcome:

- Concurrent call-log behavior is either isolated or clearly documented as shared best-effort.
- `run.inspect_call_log()` remains the recommended reliable view under concurrency.
- Add tests if behavior changes.

### P1-9: Handle Empty `Evaluate` Devsets

Problem: `Evaluate` computes a defensive `mean_pct` for empty devsets, then returns `EvaluationResult(score=round(100 * score_sum / ntotal, 2), ...)`, causing `ZeroDivisionError`.

Reference:

- `dspy/evaluate/evaluator.py`

Required outcome:

- `Evaluate(devset=[])` returns a clear validation error or an explicit empty-result score by design.
- Add a regression test.

### P1-10: Handle Empty Answer Lists in Metrics

Problem: `max_em_score` and `max_token_f1_score` call `max()` over `answers_list` without guarding empty lists.

Reference:

- `dspy/evaluate/metrics.py`

Required outcome:

- Empty reference answer lists produce a clear validation error or a defined zero score.
- Add tests.

### P1-11: Surface Sampling Threshold Misses

Problem: `BestOfN` and `Refine` return the highest-reward prediction when no sample meets the threshold, without a failure flag, exception, or metadata indicating the miss.

References:

- `dspy/predict/sampling.py`
- `dspy/predict/best_of_n.py`
- `dspy/predict/refine.py`

Required outcome:

- Callers can tell whether the threshold was met.
- Existing early-stop behavior remains intact.
- Add tests for threshold misses.

### P1-12: Avoid Accidental Parent Trace Mutation During Sampling

Problem: `sample_with_reward` extends the parent run's `optimization_trace` with the winning attempt trace. Repeated `Refine` or `BestOfN` calls can accumulate attempt traces and evict older entries.

Reference:

- `dspy/predict/sampling.py`

Required outcome:

- Decide whether this mutation is intentional. If not, isolate or record it explicitly.
- Add tests for trace contents after repeated sampling.

### P1-13: Revisit JSON Parsing Strictness

Problem: `JSONAdapter` defaults `allow_json_repair=True`, and top-level parse failures can extract the first `{...}` substring from larger completions.

References:

- `dspy/adapters/json_adapter.py`
- `dspy/adapters/utils/json_loads.py`

Required outcome:

- Decide whether strict parsing should be the default under strict transparency.
- If permissive repair remains default, document it and add tests for multiple JSON blobs or prose-wrapped JSON.

### P1-14: Preserve Mixed MCP Tool Results

Problem: Mixed MCP tool results with text plus non-text content can collapse to a string and drop non-text content. Error-only non-text payloads can produce effectively empty `RuntimeError` messages.

Reference:

- `dspy/integrations/mcp.py`

Required outcome:

- Non-text MCP result content and useful error diagnostics are preserved or rejected clearly.
- Add conversion tests.

### P1-15: Improve Persistence Failure Modes

Problems:

- `apply_module_state` raises a bare `KeyError` for missing predictors and silently ignores extra predictor state.
- `save_program` writes pickle before metadata, so metadata failure leaves an unloadable directory.

References:

- `dspy/persistence/state.py`
- `dspy/persistence/program.py`

Required outcome:

- State/schema mismatch errors are actionable.
- Program save is atomic enough to avoid partial unloadable directories, or cleanup is explicit.
- Add tests for missing/extra predictor state and metadata-write failure.

### P2-1: Surface Pool Acquire Aggregate Provenance

Problem: dr-llm returns `AcquireResult(responses, claimed_from_cache, generated)`, but `DrLlmPoolLM.acquire_samples` returns only `list[LMResponse]`.

Reference:

- `dspy/clients/dr_llm/pool.py`

Required outcome:

- Decide whether to expose aggregate provenance directly, record it in telemetry, or document how to reconstruct it from per-response provider data.
- Preserve per-response provenance in `provider_data`.

### P2-2: Document Pool Fingerprint and Metadata Behavior

Problem: DSPy forwards `LMRequest.metadata` into `BackendRequest.metadata`, while dr-llm fingerprints exclude metadata and extensions. Users may incorrectly expect metadata to isolate cache keys or acquisition cells.

References:

- `dspy/clients/dr_llm/mapping.py`
- `../dr-llm/src/dr_llm/backends/fingerprint.py`

Required outcome:

- Docs state that metadata is not cache or claim isolation.
- Experiments that need isolation should change generation-relevant request fields, pool namespace/config, or session identity as appropriate.

### P2-3: Handle dr-llm `n=1` Contract for Optimizer Proposal Paths

Problem: The dr-llm contract rejects any non-`None` `config.n`, including `LMConfig(n=1)`, while the error says `n>1` is unsupported. MIPRO grounded proposer and dataset-summary paths use `n=1`; COPRO proposal calls use `n=breadth-1`.

References:

- `dspy/clients/dr_llm/contract.py`
- `dspy/teleprompt/copro_optimizer.py`
- `dspy/propose/grounded_proposer.py`
- `dspy/propose/dataset_summary_generator.py`

Required outcome:

- Allow `n=1` if dr-llm semantics are identical to unset `n`, or reject it with an exact error and document optimizer limitations.
- COPRO breadth should either use a supported alternate LM, emulated loop, native multi-completion support, or a clear typed rejection.
- Add contract and optimizer-path tests.

### P2-4: Document and Test dr-llm v1 Supported Scope

Problem: The mapping layer correctly rejects unsupported features, but users need clear boundaries.

Expected supported fit:

- Text-only `Predict`.
- Text-only `ChainOfThought`.
- `Evaluate` with text-only programs.
- Direct `aforward` and pool cache/acquire flows.

Expected unsupported v1 scope:

- `ReAct`, `ReActV2`, `CodeAct`, and tool agents.
- Multimodal programs.
- Tool-call history.
- Native structured-output paths.
- Stop sequences, logprobs, prompt cache, unsupported roles, unsupported reasoning fields.

References:

- `dspy/clients/dr_llm/mapping.py`
- `dspy/clients/dr_llm/contract.py`

Required outcome:

- Docs and tests make the support matrix obvious.
- Rejections are typed and actionable.

### P2-5: Add or Document the `nl-code` TaskSpec Experiment Path

Problem: `nl-code` is the realistic DSPy reproduction target because it already used DSPy-style programs and optimizers, but it still needs a port from legacy `Signature` and global `dspy.configure(lm=...)` to `TaskSpec` plus explicit `RunContext`.

Required outcome:

- Provide a minimal experiment scaffold or documentation showing:
  - HumanEval/code-spec TaskSpecs.
  - Generator modules.
  - Metrics.
  - Split wiring.
  - Optimizer configuration.
  - `DrLlmDirectLM` plus adapter setup.
- Make clear whether pool use is an intentional new experiment condition rather than a reproduction path.
- Do not claim bit-exact parity until LiteLLM-vs-dr-llm wire differences and provider reasoning controls are matched.

### P2-6: Track Advanced Provider Option Gaps

Known gaps:

- DSPy `ReasoningEffort` stops at `high`; dr-llm may expose `EffortSpec.MAX`.
- Custom `registry=` is accepted at construction but is not serialized in `dump_state`, so restored programs rebuild the default registry.
- OpenRouter reasoning-off, GPT-5 minimal thinking, OpenRouter provider-specific effort, and explicit empty sampling controls are required for exact experiment parity.

References:

- `dspy/core/types/lm_config.py`
- `dspy/clients/dr_llm/base.py`
- `dspy/clients/dr_llm/mapping.py`
- `../dr-llm/src/dr_llm/llm/names.py`

Required outcome:

- Either implement these controls or document them as non-goals for the current release.
- Add tests for any implemented serialization or mapping behavior.

### P2-7: Address Smaller DSPy Footguns Where In Scope

Known issues:

- `PredictOptions.trace` defaults to `True`, so every `Predict` appends to `optimization_trace` unless callers pass `trace=False`.
- Legacy `ReAct` ignores caller `turn_log` and always starts from `TurnLog.empty()`, unlike `ReActV2`.
- `Refine` and `BestOfN` default `fail_count=num_samples`, so transient LM/parse failures can consume the sample budget.
- `module.deepcopy()` falls back to shallow/reference copies with a warning only; sampling isolation can break when deep copy fails.
- Callback handler exceptions are swallowed and logged as warnings, so observability hooks can fail silently.
- `to_jsonable` in non-strict mode stringifies unknown types in audit logs without error.
- `collect_trace_data(..., raise_on_error=False)` can drop failed examples and return a shorter list than the input.
- GEPA sync bridge helpers intentionally raise from an active event loop, making sync adapter entry points awkward in async notebooks and apps.

Required outcome:

- Fix items that materially block the dr-llm experiment target or optimizer reliability.
- Otherwise document the behavior and add targeted tests where a future regression would be costly.

## Verification Matrix

At minimum, add or update tests for these gaps as the corresponding workstreams are completed:

| Area | Required verification |
| --- | --- |
| Public imports | Clean-process import tests for `from dspy.primitives import Module` and `from dspy.evaluate.evaluator import Evaluate`. |
| SIMBA | Compile or focused `Parallel(access_examples=False)` regression test. |
| Legacy agents | Truncation-exhaustion tests for `ReAct`, `CodeAct`, and `Avatar`. |
| Bootstrap | `metric_threshold=0.0` and metric result exactly `0.0`. |
| Trace collection | Async metric and `Module` metric through `collect_trace_data`. |
| GEPA | Strict `run=` tests for multimodal and custom proposer paths. |
| Runtime logging | Multiple disk log sessions created in one second produce distinct directories. |
| Parallel concurrency | Call-log behavior under concurrent workers is tested or documented. |
| Evaluation | Empty devset behavior. |
| Metrics | Empty reference answer-list behavior. |
| dr-llm direct | `Predict` + `JSONAdapter` + `DrLlmDirectLM` + `RunContext` happy-path test. |
| dr-llm state | Direct and pool `dump_state`/`load_state` round trips. |
| dr-llm pool acquire | No-session fallback and explicit-session no-replacement behavior. |
| dr-llm mapping | Provider-specific reasoning/config cases for the default T1 controls, if implemented. |
| dr-llm lifecycle | Pool copy/close/use-after-close behavior. |
| dr-llm contract | `LMConfig(n=1)` behavior and optimizer proposal paths. |

Useful existing focused checks from the review:

- `from dspy.primitives import Module` failed in a fresh process.
- Multiple disk `RunContext.create(...)` calls in the same second produced one unique log directory.
- `LM("openai/gpt-5-mini", temperature=0.0, max_tokens=16000)` was accepted.
- `Evaluate(devset=[])` raised `ZeroDivisionError`.
- dr-llm client tests passed:

```bash
uv run pytest tests/clients/dr_llm/test_contract.py tests/clients/dr_llm/test_mapping.py tests/clients/dr_llm/test_direct_lm.py tests/clients/dr_llm/test_pool_lm.py tests/clients/dr_llm/test_dr_llm_errors.py -q
```

Postgres integration checks were skipped during review because no integration DSN was configured. Do not require a live provider call or Postgres integration for ordinary unit verification, but document any integration command that would strengthen confidence when credentials are available.

## Experiment Readiness Checklist

Before marking `V-1` done, verify the repository supports or clearly rejects these immediate experiment flows:

- Direct backend text-only run:
  - Create `DrLlmDirectLM("openai/gpt-4.1-mini", temperature=0.0, max_tokens=4000)`.
  - Create an explicit `RunContext` with `JSONAdapter` or `XMLAdapter`.
  - Run a `TaskSpec`-based `Predict` or `ChainOfThought`.
  - Run `Evaluate` over a non-empty devset.
- Pool backend cache-first run:
  - Create `DrLlmPoolLM(..., pool_config=...)`.
  - Use `aforward` for cache-first single completions.
  - Close the pool cleanly or use it as a context manager.
- Pool acquisition run:
  - Pass an explicit stable `session_id` or resolver.
  - Use `acquire_samples` only when no-replacement sampling is intended.
  - Confirm generated/cache provenance is available or documented.
- Optimizer run:
  - Use `DrLlmDirectLM` for task calls where supported.
  - Use a separate supported proposal LM when optimizer proposal paths require unsupported `n>1` or provider features, unless those paths have been fixed.
  - Pass `run=` through all module, optimizer, and metric calls.
- Exact `nl_latents` reproduction:
  - Keep on raw dr-llm/nl_latents infrastructure unless raw prompt and provider-control parity are explicitly implemented.
  - Do not use `Predict(TaskSpec)` for bit-exact encoder/decoder pool curves.

## Final Handoff Requirements

When you finish the implementation:

1. Update every relevant status row and note any intentionally deferred work.
2. Run the required commit gates.
3. Run focused tests for changed areas.
4. Summarize the final supported dr-llm experiment path in docs or examples.
5. Commit only the intended changes.
