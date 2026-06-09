# Final Repository Review

Manual findings-first review for behavioral bugs, runtime footguns, and major code smells. Scope covered public API imports, runtime execution, call logging, predict/agent modules, teleprompt optimizers, clients, adapters, integrations, persistence, and tests. No CodeRabbit.

## High-Impact Bugs

### Public imports are broken in a fresh process

`from dspy.primitives import Module` and `from dspy.evaluate.evaluator import Evaluate` fail with an import cycle. The cycle is:

`dspy.primitives.__getattr__("Module")` -> `dspy.primitives.module` -> `dspy.primitives.module_graph` -> `dspy.predict.protocol`, which executes eager imports in `dspy.predict.__init__` and re-enters `dspy.predict.predict` before `Module` is available.

References:
- `dspy/primitives/__init__.py`
- `dspy/primitives/module_graph.py`
- `dspy/predict/__init__.py`

Impact: users cannot reliably import documented public APIs in clean processes. This can also hide in tests when another import has already materialized `Module`.

### SIMBA parallel execution is broken

`Parallel(access_examples=False)` passes `run=` to callables that do not accept it. SIMBA wraps programs in a plain async function that only accepts `example`, then passes it through `Parallel(..., access_examples=False)`.

References:
- `dspy/runtime/batch.py`
- `dspy/teleprompt/simba_utils.py`

Impact: `SIMBA.compile()` should fail with `TypeError: wrapped_program() got an unexpected keyword argument 'run'` on the first parallel batch. Even for a real `Module`, the `access_examples=False` path calls `module(example, run=...)`, but `Module.__call__` is keyword-only for task inputs.

### ReAct loses truncation state and misclassifies truncation exhaustion

Legacy `ReAct` does not assign `turn_log = extracted.turn_log` after `call_with_history_truncation`, unlike `ReActV2`, `CodeAct`, and `Avatar`. If truncation shortens the log inside the helper, the loop keeps appending to the pre-truncation log.

`TruncationExhaustedError` subclasses `ValueError`, so legacy `ReAct` catches it as a parse error. Legacy `ReAct`, `CodeAct`, and `Avatar` also do not handle truncation exhaustion around post-loop extract calls, so exhaustion can abort a forward after a graceful loop exit.

References:
- `dspy/predict/react.py`
- `dspy/history/truncation.py`
- `dspy/predict/code_act.py`
- `dspy/predict/avatar/avatar.py`

Impact: context-window failures can repeat or spiral in legacy agents. Prefer `ReActV2` for new tool-calling agents until these paths are aligned.

### OpenAI reasoning-model validation accepts `temperature=0.0`

The reasoning-model constructor guard uses truthiness:

```python
if (temperature and temperature != 1.0) or (max_tokens and max_tokens < 16000):
```

That lets `temperature=0.0` pass even though the error message and tests say reasoning models only accept `1.0` or `None`. `LM.copy(temperature=0.0)` also rebuilds kwargs through generic validation and bypasses the reasoning-specific constructor rule.

References:
- `dspy/clients/lm/client.py`
- `dspy/clients/base_lm.py`

Impact: a deterministic configuration error becomes a provider-facing runtime failure.

### Bootstrap treats `metric_threshold=0.0` as no threshold

Bootstrap uses truthiness when deciding whether to apply the threshold:

```python
success = metric_val >= self.metric_threshold if self.metric_threshold else metric_val
```

References:
- `dspy/teleprompt/bootstrap.py`

Impact: `metric_threshold=0.0` is ignored as a threshold. Scores like `0.3` are accepted through truthiness, while exactly `0.0` is rejected. The threshold check should use `is not None`.

### `collect_trace_data` breaks async and module metrics

`collect_trace_data` wraps the metric in a synchronous function that calls `metric(example, prediction, trace)` directly. That bypasses the normal `invoke_metric` handling for async metrics and `Module` metrics.

References:
- `dspy/teleprompt/core/trace_collection.py`
- `dspy/evaluate/metric_invoke.py`

Impact: GEPA trace collection, GRPO trace grids, and bootstrap-finetune paths only work reliably with synchronous callables using `(example, prediction, trace)`. Direct bootstrap and `Evaluate` paths support async and `Module` metrics, so this is an inconsistent optimizer bug.

### GEPA multimodal and custom proposers omit `run=`

Several GEPA instruction-proposal paths create or need a configured optimizer run but do not pass it through:

- `SingleComponentMultiModalProposer` calls `Predict` without `run=`.
- `MultiModalInstructionProposer` calls the module wrapper without `run=`.
- The custom proposer branch creates `opt_run` but never passes it to the proposer.

References:
- `dspy/integrations/optimizers/gepa/instruction_proposal.py`
- `dspy/integrations/optimizers/gepa/adapter.py`

Impact: multimodal GEPA instruction proposal and custom async proposers fail under strict transparency. The built-in non-custom path correctly passes `run=opt_run`.

## Runtime, Observability, and Concurrency Footguns

### Disk call-log sessions collide for runs created in the same second

`create_run_log_session` uses only second-resolution UTC timestamps for the log directory and creates the directory with `exist_ok=True`. Independent strict-transparency runs created within the same second share the same `run.json` and `calls.jsonl`. The dr-llm pool session ID also uses the log session timestamp; see the dr-llm addendum below for pool-specific fallout.

References:
- `dspy/runtime/run_log_session.py`
- `dspy/clients/dr_llm/pool.py`

Impact: audit logs can merge unrelated runs, and pool-backed sampling can unintentionally share a session identity.

### `run_with_trace` bypasses the normal module spine

Tracing calls `program.aforward(...)` directly instead of `await program(...)`.

References:
- `dspy/runtime/optimization_trace.py`

Impact: this skips `invoke_module` call-scope setup, `@with_callbacks`, and top-level usage tracking. Inner `Predict` calls still go through `__call__`, but outer program observability is inconsistent. This path is used by bootstrap, SIMBA, Refine, and trace collection.

### `run_bounded` abort semantics hide never-started work

When `max_errors` triggers cancellation, items that never start stay as `RUN_BOUNDED_PENDING` and are finalized as `None`. They are absent from `stats.failed_indices` and `exceptions_map`, so `Parallel` does not record them as failures. `BoundedRunAbortedError` is raised even when some items succeeded.

References:
- `dspy/runtime/async_parallel.py`
- `dspy/runtime/batch.py`

Impact: callers must catch `BoundedRunAbortedError` and inspect partial state, but never-started work is not represented as structured failures.

### Parallel workers share LM and module call logs

`fork_worker_run` isolates worker `RunContext` state, but workers share the same LM and program objects. `record_call` appends to shared bounded ring buffers on `lm.call_log` and each caller `module.call_log` without per-worker isolation.

References:
- `dspy/runtime/run_fork.py`
- `dspy/runtime/call_log/coordinator.py`
- `dspy/runtime/batch.py`

Impact: concurrent `Parallel` runs can interleave and drop entries in shared call logs. `run.inspect_call_log()` is safer than shared LM/module logs under concurrency, but even there disk-session collision remains a separate risk.

### Optimizer forks share parent call logs

Optimizer `fork()` behavior preserves shared call-log state rather than isolating the teacher or candidate run logs the way worker forks do.

References:
- `dspy/runtime/run_context.py`
- `dspy/runtime/run_log_policy.py`

Impact: teacher bootstrap and optimizer attempts can mix into parent call-log views.

## Client and LM Configuration Footguns

### `LMProviderOptions.cache` and `max_retries` are no-ops on the LiteLLM path

The LiteLLM client always passes `{"no-cache": True, "no-store": True}` regardless of `provider_options.cache`. `LMProviderOptions.max_retries` is defined but not read by `LM`; only `LM.num_retries` controls LiteLLM retry count.

References:
- `dspy/clients/lm/client.py`
- `dspy/core/types/lm_provider.py`

Impact: strict transparency may audit `cache=True` while every LiteLLM call is uncached, and users can set `max_retries` without effect.

### Strict transparency is an easy first-run failure mode

`TelemetryConfig.transparency` defaults to strict. Bare `LM("openai/gpt-4o-mini")` leaves `temperature`, `max_tokens`, and `provider_options.cache` unset, so adapter calls can raise transparency violations unless callers build a fully explicit `RunContext`. dr-llm LMs cannot set `provider_options`, which makes cache-related strict checks especially awkward for those backends.

References:
- `dspy/runtime/config.py`
- `dspy/runtime/transparency/validate.py`
- `dspy/clients/lm/client.py`
- `dspy/clients/dr_llm/base.py`

Impact: the intended strict-audit mode is useful, but the default experience is brittle unless examples and constructors make required configuration obvious.

## Predict, Sampling, and Evaluation Issues

### `BestOfN` and `Refine` silently return a sub-threshold best sample

`sample_with_reward` stops early when a threshold is met, but if no sample meets the threshold it still returns the highest-reward prediction without a failure flag, exception, or metadata indicating threshold miss.

References:
- `dspy/predict/sampling.py`
- `dspy/predict/best_of_n.py`
- `dspy/predict/refine.py`

Impact: callers may believe a threshold was enforced when it only controlled early stopping.

### `sample_with_reward` mutates the caller's optimization trace

When a best trace exists, `sample_with_reward` extends the parent run's `optimization_trace` with the winning attempt trace.

References:
- `dspy/predict/sampling.py`

Impact: repeated `Refine` or `BestOfN` calls accumulate attempt traces on the shared run and can evict older entries under bounded trace policy.

### `Evaluate` divides by zero on empty devsets

`Evaluate` computes `mean_pct` defensively for empty devsets, but returns `EvaluationResult(score=round(100 * score_sum / ntotal, 2), ...)` without guarding `ntotal == 0`.

Reference:
- `dspy/evaluate/evaluator.py`

Impact: `Evaluate(devset=[])` raises `ZeroDivisionError` instead of returning a clear validation error or an explicit empty-result score.

### `Evaluate` binarizes `Module` judge metrics when a trace exists

`invoke_metric` sets `use_threshold = trace is not None` for `Module` metrics. `Evaluate` always passes a trace, so metrics such as `SemanticF1(threshold=0.66)` return pass/fail booleans rather than continuous scores.

References:
- `dspy/evaluate/metric_invoke.py`
- `dspy/evaluate/auto_evaluation.py`

Impact: optimizer selection can ignore how far above or below threshold each example is. Tests document this behavior, but it is surprising in production.

### Empty answer lists break max-score helpers

`max_em_score` and `max_token_f1_score` call `max()` over `answers_list` without guarding empty lists.

Reference:
- `dspy/evaluate/metrics.py`

Impact: an empty reference answer list raises `ValueError` instead of returning a clear validation error or a defined zero score.

## Generated-Code Execution

### Generated-code execution has no timeout and blocks async call paths

`PythonInterpreter.execute` synchronously waits on the Deno/Pyodide response loop. `read_until_response` caps skipped non-JSON lines but has no wall-clock timeout. `CodeAct`, `ProgramOfThought`, and RLM call this path directly, so an infinite loop or hung generated program can hang the whole async DSPy operation despite `max_iters`.

References:
- `dspy/primitives/python_interpreter/interpreter.py`
- `dspy/primitives/python_interpreter/pump.py`
- `dspy/predict/code_act.py`
- `dspy/predict/program_of_thought.py`
- `dspy/predict/rlm/execution.py`

Impact: bad generated code can wedge an otherwise bounded agent/program run.

## Adapter, Integration, and Persistence Risks

### JSON parsing is permissive by default

`JSONAdapter` defaults `allow_json_repair=True`, and when top-level parsing fails it can extract the first `{...}` substring from a larger completion.

References:
- `dspy/adapters/json_adapter.py`
- `dspy/adapters/utils/json_loads.py`

Impact: malformed JSON or prose with multiple JSON blobs can be coerced into a successful parse, potentially scoring the wrong structure as success instead of surfacing `AdapterParseError`.

### MCP tool-result conversion can drop non-text content

Mixed MCP tool results with one text part and non-text parts can collapse to string text and drop non-text content. Error-only non-text payloads can produce effectively empty `RuntimeError` messages.

Reference:
- `dspy/integrations/mcp.py`

Impact: multimodal tool outputs and error diagnostics can be lost at the integration boundary.

### Finetuning poll loops can wait forever

The Databricks finetuning poll loop only exits on `"Completed"` or `"Failed"` and has no overall timeout. Deployment has a timeout later, but the training wait before deployment does not. Several deployment HTTP requests also omit request timeouts.

OpenAI finetune `wait_for_job` has a similar shape: unknown statuses map to pending, with no max duration.

References:
- `dspy/integrations/finetune/databricks.py`
- `dspy/integrations/finetune/openai.py`

Impact: stuck or unknown provider states can block indefinitely.

### Persistence and state loading have rough failure modes

`apply_module_state` validates on a deepcopy before applying, which is good for all-or-nothing mutation, but topology mismatch still raises a bare `KeyError` for missing predictors and silently ignores extra predictor state. `save_program` writes the pickle before metadata; metadata failure leaves an unloadable directory.

References:
- `dspy/persistence/state.py`
- `dspy/persistence/program.py`

Impact: state/schema mismatch and partial program saves are harder to diagnose and recover from than they need to be.

### GEPA sync bridge fails from running event loops

GEPA sync bridge helpers intentionally raise from an active event loop.

Reference:
- `dspy/integrations/optimizers/gepa/sync_bridge.py`

Impact: synchronous GEPA adapter entry points such as `propose_new_texts` and `evaluate` are awkward in async notebooks and async app environments.

## Smaller Footguns

- `PredictOptions.trace` defaults to `True`, so every `Predict` appends to `optimization_trace` unless callers pass `trace=False`.
- Legacy `ReAct` ignores caller `turn_log` and always starts from `TurnLog.empty()`, unlike `ReActV2`.
- `Refine` and `BestOfN` default `fail_count=num_samples`, so transient LM/parse failures can consume the entire sample budget.
- `module.deepcopy()` falls back to shallow/reference copies with a warning only; sampling isolation can break when deep copy fails.
- Callback handler exceptions are swallowed and logged as warnings, so observability hooks can fail silently.
- `to_jsonable` in non-strict mode stringifies unknown types in audit logs without error.
- `collect_trace_data(..., raise_on_error=False)` can drop failed examples and return a shorter list than the input.

## Test and Verification Gaps

| Area | Gap |
| --- | --- |
| Public imports | No clean-process import test for `from dspy.primitives import Module` or direct evaluator imports. |
| SIMBA | No compile/integration test; `Parallel(access_examples=False)` is untested. |
| Legacy agents | No truncation-exhaustion regression tests for `ReAct`, `CodeAct`, or `Avatar`. |
| Bootstrap | No test for `metric_threshold=0.0` or metric returning exactly `0.0`. |
| Trace collection | No async or `Module` metric test through `collect_trace_data`. |
| GEPA multimodal | No strict `run=` test for `MultiModalInstructionProposer` or custom proposer paths. |
| Runtime logging | No collision test for multiple disk log sessions created in one second. |
| Parallel concurrency | No assertion that call-log views remain reliable under concurrent workers. |
| Evaluation | No empty-devset test and no empty-answer-list metric tests. |

## Recommended Fix Priority

1. Fix public import cycles around `dspy.primitives.Module` and direct evaluator imports.
2. Fix SIMBA / `Parallel._run_pair` by passing `run=` only to supported call targets, or by avoiding `access_examples=False` for plain callables.
3. Align legacy `ReAct` truncation behavior with `ReActV2`, including post-loop extract handling.
4. Fix Bootstrap threshold logic with `is not None`.
5. Make `collect_trace_data` delegate to `invoke_metric`.
6. Thread `run=` through GEPA multimodal and custom proposer paths.
7. Make disk log session IDs unique below one-second granularity.
8. Wire or reject/document `LMProviderOptions.cache` and `max_retries`.
9. Add timeouts for generated-code execution and finetune polling.

## What Looks Solid

- Adapter input validation at pipeline entry (`validate_task_inputs`) is consistently applied.
- `fork_worker_run` isolates worker `optimization_trace` and per-worker in-memory `RunContext.call_log`.
- Transactional state load via deepcopy-then-apply protects against partial mutation on failure.
- dr-llm v1 mapping rejects unsupported features with typed errors before opaque backend failures.
- Recent branch work includes useful behavioral hardening around Refine predictor naming, COPRO empty scores, and test alignment.

## Verification Notes

The original manual pass verified the top issues with focused local commands:

- `from dspy.primitives import Module` fails in a fresh process.
- Multiple disk `RunContext.create(...)` calls in the same second produced one unique log directory.
- `LM("openai/gpt-5-mini", temperature=0.0, max_tokens=16000)` was accepted.
- `Evaluate(devset=[])` raises `ZeroDivisionError`.

Existing targeted tests for reasoning requirements and run-log session creation passed, but they do not cover the failing cases above. The merged review findings from the second manual pass were source-verified by that reviewer and have been grouped here with deduplication.

## dr-llm Focused Review Addendum

Manual review focused on the dr-llm integration seams between this repository and `../dr-llm`: DSPy request/response mapping, direct backend usage, pool-backed cache/acquire behavior, constructor and state contracts, lifecycle, and targeted tests on both sides.

### High: pool acquisition session identity has unsafe defaults

`resolve_pool_session_id` uses the disk log session when present, falls back to an LM-level `session_id` when provided, and otherwise returns a fresh `uuid4()` for every call. That creates two related pool-acquire footguns:

- With disk call logging enabled, the session id is based on `{DSPY_RUN_ID}:{log_session.timestamp}`. Because log sessions currently use second-resolution timestamps, independent runs created in the same second can collide.
- With disk call logging disabled and no explicit `session_id`, repeated `acquire_samples(...)` calls get different random sessions, so no-replacement semantics do not hold across calls.

References:
- `dspy/runtime/run_log_session.py`
- `dspy/clients/dr_llm/pool.py`
- `tests/clients/dr_llm/test_integration_pool.py`

Impact: pool-backed experiments can either unintentionally share acquisition state or unintentionally reset it. Direct `aforward` is unaffected because it is cache-first and does not claim samples.

### High: dr-llm LMs cannot be loaded through the safe state path

`DrLlmDirectLM` and `DrLlmPoolLM` are listed as builtin LM classes, and they dump normal LM state with class paths under `dspy.clients.dr_llm.*`. However, `BaseLM.load_state` still rejects every class path except `dspy.clients.lm.LM` unless `allow_custom_lm_class=True`.

References:
- `dspy/clients/lm_registry.py`
- `dspy/clients/base_lm.py`
- `dspy/clients/dr_llm/base.py`

Impact: saved DSPy programs using either dr-llm approach cannot reload through the normal safe path. Users have to opt into unsafe custom LM loading for a class that this repo already treats as builtin.

### High: exact experiment reproduction requires choosing the experiment family

The reviewed experiment docs describe two distinct stacks. The `nl_latents` pool compression curves run outside DSPy through dr-llm `LlmConfig` and pool grids, with single-user-message prompts. The `nl-code` DSPy optimization and full-5x eval path runs through legacy `dspy.LM` / LiteLLM / OpenRouter with `Signature` plus `ChatAdapter` scaffolding.

Impact: treating dr-dspy as a drop-in reproduction path for both families will produce non-comparable numbers. Pool experiments never went through DSPy, and DSPy optimization runs did not use the dr-llm pool grid.

### Context: compression experiment surface reviewed

The last-week optimization-shift baseline used `mimo-v2-flash`, `llama-3.3-nemotron-super-49b-v1.5`, `gpt-oss-20b`, and `gpt-5-nano` over prompted character budgets `32`, `64`, `128`, `256`, `512`, and `1024`. The encoder prompt was raw code plus "Provide a concise natural language description of the code using at most {{BUDGET}} characters." The decoder prompt was the generated description plus "Write functional code in Python according to the description." The main curve plotted representation compression ratio against decoder pass rate, with fixed lossless compression/minification baselines beside the raw representations.

The two compression planning docs prioritize an encoder-only or representation-policy-only optimizer pass, with compressed-output length visible in the metric:

| Plan | Optimizer | Target | Key setup |
| --- | --- | --- | --- |
| Optimizer ranking | MIPROv2 | encoder-only | `auto=medium`, demos disabled, `seed=9`, `init_temperature=1.0`, valset around `50` to `69`. |
| Optimizer ranking | COPRO | encoder-only | `breadth=6`, `depth=2`, compression-aware metric on the compile set. |
| Optimizer ranking | GEPA | encoder-only | `max_metric_calls=64` to `128`, `reflection_minibatch_size=3`, compression feedback. |
| Optimizer ranking | SIMBA | encoder rules | `max_demos=0`, `bsize=16` to `32`, low candidate temperatures. |
| Optimizer ranking | InferRules | encoder rules | `num_rules=3` to `5`, small demo budget. |
| Shift plot | MIPROv2 | representation-policy section only | `auto=light` smoke, then `medium`, with heldout final paired eval. |
| Shift plot fallback | GEPA | representation-policy section only | constrained `max_metric_calls=64` to `128`; inspect prompt bloat. |

Impact: the optimizer surface mostly exists in dr-dspy, but the experiment harness still has to provide the mutable prompt section, compression-aware metric, exact split discipline, decoder scaffold accounting, and final paired curve evaluation.

### High: last-week compression model controls are only partially expressible through DrLlmLM

The observed `nl_latents` compression curves used these request controls:

| nl_latents config | Model string | Key request controls | dr-dspy status today |
| --- | --- | --- | --- |
| `openrouter/xiaomi/mimo-v2-flash/off/v1` | `openrouter/xiaomi/mimo-v2-flash` | OpenRouter `reasoning.enabled=false`, `temperature=0.7`, `top_p=0.95` | Sampling approximable; reasoning toggle missing. |
| `openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5/off/v1` | `openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5` | OpenRouter `reasoning.enabled=false`, `temperature=0.7`, `top_p=0.95` | Sampling approximable; reasoning toggle missing. |
| `openrouter/openai/gpt-oss-20b/low/v1` | `openrouter/openai/gpt-oss-20b` | OpenRouter `reasoning.effort=low`, `BackendRequest.effort=na` | Current mapping puts effort on the wrong field. |
| `openrouter/openai/gpt-5-nano/low/v1` | `openrouter/openai/gpt-5-nano` | OpenRouter `reasoning.effort=low`, `BackendRequest.effort=na` | Current mapping puts effort on the wrong field. |
| `openai/gpt-5-nano/minimal/v1` | `openai/gpt-5-nano` | OpenAI `thinking_level=minimal`, no sampling override | Minimal thinking is not representable. |
| `google/gemini-2.5-flash-lite/off/v1` | `google/gemini-2.5-flash-lite` | Google `thinking_level=off`, `temperature=0.7`, `top_p=0.95` | Sampling approximable; Google thinking control missing. |

References:
- `../nl_latents/src/nl_latents/sampling/llm/catalog.py`
- `dspy/clients/dr_llm/base.py`
- `dspy/clients/dr_llm/mapping.py`
- `../dr-llm/src/dr_llm/llm/providers/impls/openrouter/request_controls.py`
- `../dr-llm/src/dr_llm/llm/providers/impls/openai/request_controls.py`
- `../dr-llm/src/dr_llm/llm/providers/impls/google/request_controls.py`

Impact: even when the model string matches, provider payloads and pool fingerprints are not exact unless dr-dspy can carry provider-specific dr-llm reasoning objects. Per-call `LMConfig(top_p=0.95)` can match ordinary sampling, but constructor defaults currently only preserve `temperature` and `max_tokens`.

### High: default nl_latents T1 configs are rejected through current dr-dspy mapping

The default T1 compression setup in `../nl_latents/scripts/code_comp_t1/shared_config.sh` uses `humaneval-plus`, budgets `64,128,256`, one encoder sample and one decoder sample per config, and same-as-encoder decoder LLM mode. Its default LLM config IDs are:

- `openrouter/xiaomi/mimo-v2-flash/off/v1`
- `openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5/off/v1`
- `openrouter/openai/gpt-5-nano/low/v1`
- `openrouter/openai/gpt-oss-20b/low/v1`
- `openai/gpt-5-nano/minimal/v1`

Those catalog entries use dr-llm-native `BackendRequest.reasoning` shapes: OpenRouter disabled reasoning (`{"kind": "openrouter", "enabled": false}`), OpenRouter effort (`{"kind": "openrouter", "effort": "low"}`), and OpenAI minimal thinking (`{"kind": "openai", "thinking_level": "minimal"}`). Current dr-dspy maps only `LMReasoningConfig.effort` to `BackendRequest.effort` and hard-codes `BackendRequest.reasoning=None`.

Focused validation through the dr-llm provider registry showed all five default T1 configs fail when represented through current dr-dspy mapping:

- MiMo and Nemotron: `reasoning is required for provider='openrouter'`.
- OpenRouter GPT-5 nano and GPT-OSS 20B: `effort is not supported for provider='openrouter'`.
- Direct OpenAI GPT-5 nano: `effort is not supported for provider='openai'`.

References:
- `dspy/clients/dr_llm/mapping.py`
- `dspy/clients/dr_llm/contract.py`
- `../nl_latents/scripts/code_comp_t1/shared_config.sh`
- `../nl_latents/src/nl_latents/sampling/llm/catalog.py`
- `../dr-llm/src/dr_llm/backends/models.py`
- `../dr-llm/src/dr_llm/llm/config.py`
- `../dr-llm/src/dr_llm/llm/providers/concepts/reasoning.py`

Impact: the current dr-dspy setup cannot carry exact default T1 requests through either `DrLlmDirectLM` or `DrLlmPoolLM`. The dr-llm direct and pool backends can carry the exact requests, but dr-dspy needs an explicit dr-llm config/request path or provider-specific reasoning support before it can be used for this experiment family.

### High: nl_latents pool curves cannot be exactly reproduced through dr-dspy `Predict`

The nl_latents encoder/decoder pool prompts are raw single-user-message templates. The budgeted encoder prompt is exactly `{{CODE}}` followed by a character-budget instruction, and the decoder prompt is exactly `{{DESCRIPTION}}` followed by "Write functional code in Python according to the description." The encoder and decoder request builders both render one `role="user"` message and no system message.

Routing those calls through dr-dspy `Predict` changes the wire prompt by adding adapter/system scaffolding, field descriptions, field structure, output requirements, and field wrappers. `MessageAssembler` always appends a system message for adapter-formatted calls, and `ChatAdapter`/`JSONAdapter` add task output-format requirements. This scaffold is useful for DSPy programs, but it is not part of the T1 raw-prompt experiment.

References:
- `../nl_latents/prompt_spaces/t1-budgeted-encoder__code.json`
- `../nl_latents/prompt_spaces/t1-description-decoder__code.json`
- `../nl_latents/src/nl_latents/sampling/encoder/request.py`
- `../nl_latents/src/nl_latents/sampling/decoder/request.py`
- `dspy/adapters/base/adapter.py`
- `dspy/adapters/format/message_assembler.py`

Impact: exact nl_latents pool replication should stay on nl_latents plus dr-llm pool infrastructure, call dr-llm `build_request_from_config()` directly, or use a deliberately raw DSPy LM request path. Using `Predict(TaskSpec)` will change prompts and invalidate bit-exact comparisons.

### High: nl_latents pools and DrLlmPoolLM pools are different systems

The existing nl_latents experiments use `dr_llm.pool.LlmPoolBackend` through a grid seeding workflow. Each row stores key axes such as prompt template, data sample, and LLM config ID, plus serialized `LlmConfig` and messages. `DrLlmPoolLM` uses `dr_llm.backends.PoolBackend`, where the primary key is `request_fingerprint` over a canonical `BackendRequest`.

References:
- `../nl_latents/src/nl_latents/sampling/encoder/pool.py`
- `../nl_latents/src/nl_latents/sampling/decoder/pool.py`
- `dspy/clients/dr_llm/pool.py`
- `../dr-llm/src/dr_llm/backends/pool.py`
- `../dr-llm/src/dr_llm/backends/fingerprint.py`

Impact: pointing `DrLlmPoolLM` at an nl_latents-seeded encoder or decoder pool will not produce cache hits. Exact curve replication should stay on the nl_latents pool harness, or the requests must be re-seeded through `PoolBackend` after the dr-dspy request-shape gaps are fixed.

### Medium: nl-code DSPy reproduction is feasible only after an explicit port

The nl-code optimization family is the realistic target for dr-dspy because it already used DSPy-style programs and optimizers. But it still needs porting from legacy `Signature` and global `dspy.configure(lm=...)` to TaskSpec plus explicit `RunContext.create(lm=..., adapter=...)`. The HumanEval/code-spec TaskSpecs, generator modules, metrics, split wiring, and optimizer scripts are not present in this repo today.

Impact: dr-dspy can approximate nl-code after porting programs and metrics, likely with `DrLlmDirectLM` rather than pool. It should not be claimed as bit-exact until LiteLLM-vs-dr-llm wire differences and reasoning controls are matched.

### Medium: pool LM lifecycle is easy to misuse

`BaseLM.copy()` is shallow. For `DrLlmPoolLM`, the copied wrapper shares the same `_backend`, while `_closed` remains per wrapper. Closing a copy and then closing the original calls `PoolBackend.close()` twice on the same backend. Separately, `_closed` is only checked by `close()`; `aforward` and `acquire_samples` still delegate to the torn-down backend after context-manager exit or manual close.

References:
- `dspy/clients/base_lm.py`
- `dspy/clients/dr_llm/pool.py`
- `dspy/predict/sampling.py`
- `dspy/teleprompt/simba_utils.py`

Impact: optimizers and sampling utilities that copy LMs can accidentally tear down or double-teardown a shared Postgres-backed pool backend while another wrapper still appears usable. Long-running scripts can also accidentally use an LM handle after the pool consumer has been torn down.

### Medium: dr-llm v1 scope blocks major DSPy modules

The mapping layer correctly rejects tools, multimodal parts, unsupported roles, structured response formats, stop sequences, logprobs, prompt cache, and unsupported reasoning fields before requests reach dr-llm. That aligns with dr-llm v1, but it means the dr-llm LM classes are not drop-in replacements for every DSPy program.

References:
- `dspy/clients/dr_llm/mapping.py`
- `dspy/clients/dr_llm/contract.py`

Impact: text-only `Predict`, `ChainOfThought`, and `Evaluate` paths are the expected fit. ReAct/ReActV2/CodeAct, tool agents, multimodal programs, tool-call history, and native structured-output paths are not supported through these v1 backends.

### Medium: batch-fill pool workflow is not surfaced on `DrLlmPoolLM`

dr-llm's `PoolBackend` supports a batch flow with `submit_batch`, `await_drain`/`adrain`, and then acquire. `DrLlmPoolLM` exposes `aforward` and `acquire_samples`, but not `submit_batch` or drain methods. It also exposes pool mechanics, not the nl_latents experiment harness around cross-product grids, encoder-to-decoder lineage, budget axis bindings, compression baselines, and curve aggregation.

References:
- `dspy/clients/dr_llm/pool.py`
- `../dr-llm/src/dr_llm/backends/pool.py`

Impact: users who want to pre-seed a grid and drain workers must still use dr-llm's `PoolBackend` directly. `aforward` on a miss does generate and insert one sample, but that is not the same workflow as worker-backed batch fill or nl_latents-style curve orchestration.

### Medium: dr-llm prompt-model use breaks proposal paths that set `n`

The dr-llm contract rejects any non-`None` `config.n` but reports that `n>1` is unsupported. I confirmed `LMConfig(n=1)` raises `LMUnsupportedFeatureError`. This matters for optimizers, not just direct user calls: MIPRO's grounded proposer and dataset-summary flows use single-completion `n=1`, while COPRO proposal calls use `n=breadth-1`.

References:
- `dspy/clients/dr_llm/contract.py`
- `dspy/teleprompt/copro_optimizer.py`
- `dspy/propose/grounded_proposer.py`
- `dspy/propose/dataset_summary_generator.py`

Impact: the compression docs' MIPRO and COPRO optimizer runs can use `DrLlmDirectLM` as the task LM only if proposal calls use another LM or this contract changes. Allowing `n=1` would fix some MIPRO paths; COPRO breadth still needs either multi-completion support or an emulated loop of single completions.

### Low/Medium: pool acquire aggregate provenance is dropped

dr-llm returns `AcquireResult(responses, claimed_from_cache, generated)`, but `DrLlmPoolLM.acquire_samples` returns only `list[LMResponse]`. Per-response provenance such as `provider_data["source"]` is preserved, but aggregate generated-vs-cache counts are not available to DSPy callers.

Reference:
- `dspy/clients/dr_llm/pool.py`

Impact: experiment telemetry and cost/debugging summaries have to reconstruct counts from per-response metadata or use `PoolBackend` directly.

### Low/Medium: pool fingerprint and metadata behavior is under-documented on the DSPy side

DSPy forwards `LMRequest.metadata` into `BackendRequest.metadata`, while dr-llm fingerprints exclude metadata and extensions. That is useful because run-specific metadata does not fragment the cache, but the DSPy-side docs do not make it clear that metadata is not cache or claim isolation.

References:
- `dspy/clients/dr_llm/mapping.py`
- `../dr-llm/src/dr_llm/backends/fingerprint.py`

Impact: users may tag requests with experiment IDs in metadata and expect separate pool cache keys or acquisition cells. They will instead share the same fingerprint when the generation-relevant fields are identical.

### Minor: small contract gaps remain around advanced options

`EffortSpec.MAX` exists in dr-llm, but DSPy's `ReasoningEffort` currently stops at `high`, so `max` cannot be requested through `LMConfig`. Custom `registry=` is accepted at construction but is not serialized in `dump_state`, so restored programs always rebuild the default registry.

The experiment-control deltas are larger than that for exact parity:

- OpenRouter reasoning-off controls such as `reasoning_enabled=False` have no dr-dspy v1 equivalent; `EffortSpec.NA` is not the same as an explicit disabled toggle.
- GPT-5 minimal thinking through dr-llm thinking-level controls is not representable by DSPy's `ReasoningEffort`.
- OpenRouter effort controls are not equivalent to DSPy's generic `ReasoningEffort`; current mapping sends them as `BackendRequest.effort`, which OpenRouter rejects for the T1 GPT-5 nano and GPT-OSS configs.
- Suppressing provider-default sampling with explicit empty sampling controls is not exposed on the dr-dspy constructor surface.

References:
- `dspy/core/types/lm_config.py`
- `dspy/clients/dr_llm/base.py`
- `dspy/clients/dr_llm/mapping.py`
- `../dr-llm/src/dr_llm/llm/names.py`

Impact: these are not correctness bugs for the default text-only path, but they should be explicit for experiments that rely on custom provider registries, dr-llm's maximum-effort mode, or exact parity with nl_latents/nl-code catalog controls.

### Alignment Notes

The core request/response boundary mostly aligns with `../dr-llm`: text-only messages are converted to `BackendRequest`, provider/model splitting maps `openai/gpt-4.1-mini` to `ProviderName.OPENAI` plus `gpt-4.1-mini`, unsupported tools/multimodal/structured-output fields are rejected, response provenance is preserved in `provider_data`, error translation maps dr-llm backend/provider errors into the DSPy `LMError` hierarchy, and pool miss-to-hit plus session acquire semantics are covered by tests. DSPy reasoning effort maps to `BackendRequest.effort` for providers that actually use dr-llm `EffortSpec`, but it does not cover provider-specific `BackendRequest.reasoning`. Capabilities probing through a dedicated `DirectBackend` for pool LMs also matches dr-llm's current design because `PoolBackend` has no public `.capabilities()` API.

Direct path guidance: `DrLlmDirectLM` is ready for text-only programs with `JSONAdapter` or `XMLAdapter`. Configure auth and routing through the dr-llm registry/environment, not `LMProviderOptions`.

Pool path guidance: use `aforward` for cache-first single completions, and use `acquire_samples` only with an explicit stable session identity unless disk logging provides a known-safe session. Use dr-llm `PoolBackend.submit_batch` plus `await_drain` directly for batch pre-fill workflows today.

Experiment parity guidance: keep nl_latents pool curves on the raw dr-llm/nl_latents infrastructure for exact replication until dr-dspy exposes provider-specific dr-llm reasoning/config controls and a raw single-message request path. For nl-code reproduction, port the code-spec programs and metrics to TaskSpec/Predict, run with `DrLlmDirectLM` plus `ChatAdapter`, match optimizer compile settings and splits, and disclose remaining LiteLLM-vs-dr-llm wire differences. Do not use the pool backend for optimizer runs unless cached sampling is an intentional new experiment condition.

Compression optimizer guidance: the documented optimizer knobs are present in dr-dspy for MIPROv2, COPRO, GEPA, SIMBA, and InferRules, but the experiment-specific pieces are not framework-native. "Optimize only the representation-policy section" needs experiment-layer TaskSpec composition, compression-aware scoring needs a custom metric over pass rate and representation length, and decoder scaffold accounting needs run metadata. SIMBA also remains blocked by the parallel `run=` forwarding issue noted above if it is included in the full compression ranking.

### dr-llm Verification Notes

I did not modify implementation code during this focused review. I ran:

- `uv run pytest tests/clients/dr_llm/test_contract.py tests/clients/dr_llm/test_mapping.py tests/clients/dr_llm/test_direct_lm.py tests/clients/dr_llm/test_pool_lm.py tests/clients/dr_llm/test_dr_llm_errors.py -q` in this repo: 43 passed.
- `uv run pytest tests/backends/test_direct_backend.py tests/backends/test_pool_backend.py tests/backends/test_converters.py tests/backends/test_fingerprint.py tests/backends/test_validation.py tests/backends/test_async_bridge.py -q` in `../dr-llm`: 37 passed.
- Postgres integration checks in both repos: skipped because no integration DSN was configured.

I also used focused local scripts to confirm the safe state-load failure, shallow pool copy/double-close behavior, and `LMConfig(n=1)` rejection. Remaining gaps worth covering with durable tests: a full `Predict` + `JSONAdapter` + `DrLlmDirectLM` + `RunContext` happy path, direct and pool `dump_state`/`load_state` round trips, and a pool acquire test that makes the no-session fallback behavior explicit.

For the compression parity pass, I used source inspection and focused local scripts to compare T1 dr-llm catalog payloads against dr-dspy `LMRequest` to `BackendRequest` mapping. The no-provider-call validator check confirmed that all five default T1 configs are rejected through current dr-dspy mapping before any live API call.

Experiment-parity checks not run: live provider calls, nl-code session replay, or nl_latents curve replay. Useful next checks would be a wire-parity probe comparing dr-llm `build_request_from_config()` payloads to `DrLlmDirectLM` backend requests for the same messages, a prompt-parity diff between nl-code ChatAdapter rendering and nl_latents raw templates, and a one-task HumanEval smoke optimizer after porting the TaskSpecs and metric.
