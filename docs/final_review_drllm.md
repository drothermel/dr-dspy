# Final Review: dr-llm Updates

Source: `docs/final_review.md`. This split keeps findings relevant when updating `../dr-llm`, its pool/backend behavior, provider-control surface, or experiment harnesses that bypass DSPy.

## Cross-Boundary Issues to Coordinate with dr-dspy

Several findings are primarily implemented in dr-dspy but affect dr-llm-backed experiments. Track them alongside any dr-llm changes so the two repositories do not drift:

- dr-dspy disk call-log sessions use second-resolution timestamps; `DrLlmPoolLM` can derive pool acquisition session IDs from those timestamps.
- dr-dspy cannot currently carry provider-specific dr-llm reasoning objects such as OpenRouter reasoning-off, OpenRouter provider-specific effort, or OpenAI minimal thinking.
- dr-dspy `Predict(TaskSpec)` changes the prompt shape compared with raw single-message `nl_latents` encoder/decoder prompts.
- dr-dspy `DrLlmPoolLM` uses `PoolBackend` request fingerprints, while existing `nl_latents` pools use a different grid-seeding system.

## High-Priority dr-llm-Relevant Findings

### Exact experiment reproduction requires choosing the experiment family

The reviewed experiment docs describe two distinct stacks. The `nl_latents` pool compression curves run outside DSPy through dr-llm `LlmConfig` and pool grids, with single-user-message prompts. The `nl-code` DSPy optimization and full-5x eval path runs through legacy `dspy.LM` / LiteLLM / OpenRouter with `Signature` plus `ChatAdapter` scaffolding.

Impact: treating dr-dspy as a drop-in reproduction path for both families will produce non-comparable numbers. Pool experiments never went through DSPy, and DSPy optimization runs did not use the dr-llm pool grid.

### Compression experiment surface reviewed

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

### Provider-specific model controls need a clear bridge

The observed `nl_latents` compression curves used these request controls:

| nl_latents config | Model string | Key request controls |
| --- | --- | --- |
| `openrouter/xiaomi/mimo-v2-flash/off/v1` | `openrouter/xiaomi/mimo-v2-flash` | OpenRouter `reasoning.enabled=false`, `temperature=0.7`, `top_p=0.95` |
| `openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5/off/v1` | `openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5` | OpenRouter `reasoning.enabled=false`, `temperature=0.7`, `top_p=0.95` |
| `openrouter/openai/gpt-oss-20b/low/v1` | `openrouter/openai/gpt-oss-20b` | OpenRouter `reasoning.effort=low`, `BackendRequest.effort=na` |
| `openrouter/openai/gpt-5-nano/low/v1` | `openrouter/openai/gpt-5-nano` | OpenRouter `reasoning.effort=low`, `BackendRequest.effort=na` |
| `openai/gpt-5-nano/minimal/v1` | `openai/gpt-5-nano` | OpenAI `thinking_level=minimal`, no sampling override |
| `google/gemini-2.5-flash-lite/off/v1` | `google/gemini-2.5-flash-lite` | Google `thinking_level=off`, `temperature=0.7`, `top_p=0.95` |

References:
- `../nl_latents/src/nl_latents/sampling/llm/catalog.py`
- `../dr-llm/src/dr_llm/llm/providers/impls/openrouter/request_controls.py`
- `../dr-llm/src/dr_llm/llm/providers/impls/openai/request_controls.py`
- `../dr-llm/src/dr_llm/llm/providers/impls/google/request_controls.py`
- `dspy/clients/dr_llm/base.py`
- `dspy/clients/dr_llm/mapping.py`

Impact: dr-llm can express the native provider controls, but dr-dspy cannot yet pass them through exactly. If dr-llm changes these provider-control shapes, update the dr-dspy bridge and parity checks in the same review.

### Default `nl_latents` T1 configs should remain dr-llm-native until the bridge is fixed

The default T1 compression setup in `../nl_latents/scripts/code_comp_t1/shared_config.sh` uses `humaneval-plus`, budgets `64,128,256`, one encoder sample and one decoder sample per config, and same-as-encoder decoder LLM mode. Its default LLM config IDs are:

- `openrouter/xiaomi/mimo-v2-flash/off/v1`
- `openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5/off/v1`
- `openrouter/openai/gpt-5-nano/low/v1`
- `openrouter/openai/gpt-oss-20b/low/v1`
- `openai/gpt-5-nano/minimal/v1`

Those catalog entries use dr-llm-native `BackendRequest.reasoning` shapes: OpenRouter disabled reasoning (`{"kind": "openrouter", "enabled": false}`), OpenRouter effort (`{"kind": "openrouter", "effort": "low"}`), and OpenAI minimal thinking (`{"kind": "openai", "thinking_level": "minimal"}`).

References:
- `../nl_latents/scripts/code_comp_t1/shared_config.sh`
- `../nl_latents/src/nl_latents/sampling/llm/catalog.py`
- `../dr-llm/src/dr_llm/backends/models.py`
- `../dr-llm/src/dr_llm/llm/config.py`
- `../dr-llm/src/dr_llm/llm/providers/concepts/reasoning.py`
- `dspy/clients/dr_llm/mapping.py`
- `dspy/clients/dr_llm/contract.py`

Impact: the dr-llm direct and pool backends can carry the exact requests, but dr-dspy currently rejects or misroutes them. Keep exact T1 reproduction on dr-llm/nl_latents until dr-dspy has an explicit dr-llm config/request path or provider-specific reasoning support.

### Raw `nl_latents` prompts are not DSPy `Predict` prompts

The `nl_latents` encoder/decoder pool prompts are raw single-user-message templates. The budgeted encoder prompt is exactly `{{CODE}}` followed by a character-budget instruction, and the decoder prompt is exactly `{{DESCRIPTION}}` followed by "Write functional code in Python according to the description." The encoder and decoder request builders both render one `role="user"` message and no system message.

References:
- `../nl_latents/prompt_spaces/t1-budgeted-encoder__code.json`
- `../nl_latents/prompt_spaces/t1-description-decoder__code.json`
- `../nl_latents/src/nl_latents/sampling/encoder/request.py`
- `../nl_latents/src/nl_latents/sampling/decoder/request.py`
- `dspy/adapters/base/adapter.py`
- `dspy/adapters/format/message_assembler.py`

Impact: exact `nl_latents` pool replication should stay on `nl_latents` plus dr-llm pool infrastructure, call dr-llm `build_request_from_config()` directly, or use a deliberately raw DSPy LM request path. Using `Predict(TaskSpec)` will change prompts and invalidate bit-exact comparisons.

### `nl_latents` pools and `DrLlmPoolLM` pools are different systems

The existing `nl_latents` experiments use `dr_llm.pool.LlmPoolBackend` through a grid seeding workflow. Each row stores key axes such as prompt template, data sample, and LLM config ID, plus serialized `LlmConfig` and messages. `DrLlmPoolLM` uses `dr_llm.backends.PoolBackend`, where the primary key is `request_fingerprint` over a canonical `BackendRequest`.

References:
- `../nl_latents/src/nl_latents/sampling/encoder/pool.py`
- `../nl_latents/src/nl_latents/sampling/decoder/pool.py`
- `../dr-llm/src/dr_llm/backends/pool.py`
- `../dr-llm/src/dr_llm/backends/fingerprint.py`
- `dspy/clients/dr_llm/pool.py`

Impact: pointing `DrLlmPoolLM` at an `nl_latents`-seeded encoder or decoder pool will not produce cache hits. Exact curve replication should stay on the `nl_latents` pool harness, or the requests must be re-seeded through `PoolBackend` after the dr-dspy request-shape gaps are fixed.

## Pool and Backend Behavior

### Pool acquisition session identity has unsafe cross-repo defaults

From the dr-dspy wrapper, `resolve_pool_session_id` can derive session IDs from disk log sessions or generate a fresh random session per call. The dr-llm-side concern is that no-replacement acquisition semantics depend on stable session identity, regardless of which caller generated it.

References:
- `../dr-llm/src/dr_llm/backends/pool.py`
- `dspy/runtime/run_log_session.py`
- `dspy/clients/dr_llm/pool.py`
- `tests/clients/dr_llm/test_integration_pool.py`

Impact: pool-backed experiments can unintentionally share acquisition state or unintentionally reset it. dr-llm tests and docs should make acquisition session identity explicit for callers outside DSPy too.

### Batch-fill workflow remains a dr-llm-native workflow

dr-llm's `PoolBackend` supports a batch flow with `submit_batch`, `await_drain`/`adrain`, and then acquire. `DrLlmPoolLM` exposes only `aforward` and `acquire_samples`, and it does not expose the `nl_latents` experiment harness around cross-product grids, encoder-to-decoder lineage, budget axis bindings, compression baselines, and curve aggregation.

References:
- `../dr-llm/src/dr_llm/backends/pool.py`
- `dspy/clients/dr_llm/pool.py`

Impact: users who want to pre-seed a grid and drain workers must still use dr-llm's `PoolBackend` directly. `aforward` on a miss does generate and insert one sample, but that is not the same workflow as worker-backed batch fill or `nl_latents`-style curve orchestration.

### Pool fingerprint and metadata behavior should be documented

DSPy forwards `LMRequest.metadata` into `BackendRequest.metadata`, while dr-llm fingerprints exclude metadata and extensions. That is useful because run-specific metadata does not fragment the cache, but docs should make clear that metadata is not cache or claim isolation.

References:
- `../dr-llm/src/dr_llm/backends/fingerprint.py`
- `dspy/clients/dr_llm/mapping.py`

Impact: users may tag requests with experiment IDs in metadata and expect separate pool cache keys or acquisition cells. They will instead share the same fingerprint when generation-relevant fields are identical.

### Pool wrapper lifecycle issue may benefit from backend idempotency

The immediate bug is in dr-dspy: `BaseLM.copy()` is shallow, so copied `DrLlmPoolLM` wrappers can close the same `_backend` twice, and calls after close still delegate to the torn-down backend. If `PoolBackend.close()` is not already idempotent, making that true in dr-llm would reduce blast radius for wrappers and direct backend users.

References:
- `../dr-llm/src/dr_llm/backends/pool.py`
- `dspy/clients/base_lm.py`
- `dspy/clients/dr_llm/pool.py`

Impact: optimizers and sampling utilities that copy LMs can accidentally tear down or double-teardown a shared Postgres-backed pool backend while another wrapper still appears usable.

### Aggregate acquire provenance is not surfaced through DSPy

dr-llm returns `AcquireResult(responses, claimed_from_cache, generated)`, but `DrLlmPoolLM.acquire_samples` returns only `list[LMResponse]`.

Reference:
- `dspy/clients/dr_llm/pool.py`

Impact: the dr-llm backend already has the aggregate data. DSPy callers lose it unless the wrapper returns or records it somewhere. Keep this in mind if changing `AcquireResult` or provenance fields.

## Provider and Contract Scope

### dr-llm v1 scope intentionally blocks major DSPy modules

The dr-dspy mapping layer correctly rejects tools, multimodal parts, unsupported roles, structured response formats, stop sequences, logprobs, prompt cache, and unsupported reasoning fields before requests reach dr-llm. That aligns with dr-llm v1, but it means the dr-llm LM classes are not drop-in replacements for every DSPy program.

References:
- `../dr-llm/src/dr_llm/backends/models.py`
- `dspy/clients/dr_llm/mapping.py`
- `dspy/clients/dr_llm/contract.py`

Impact: text-only `Predict`, `ChainOfThought`, and `Evaluate` paths are the expected fit. `ReAct`, `ReActV2`, `CodeAct`, tool agents, multimodal programs, tool-call history, and native structured-output paths are not supported through these v1 backends.

### `n=1` proposal calls are rejected by the dr-dspy contract

The dr-dspy contract rejects any non-`None` `config.n` but reports that `n>1` is unsupported. Focused validation confirmed `LMConfig(n=1)` raises `LMUnsupportedFeatureError`. This matters for optimizers: MIPRO's grounded proposer and dataset-summary flows use single-completion `n=1`, while COPRO proposal calls use `n=breadth-1`.

References:
- `dspy/clients/dr_llm/contract.py`
- `dspy/teleprompt/copro_optimizer.py`
- `dspy/propose/grounded_proposer.py`
- `dspy/propose/dataset_summary_generator.py`

Impact: this is a dr-dspy contract issue unless dr-llm also needs native or emulated multi-completion support. Allowing `n=1` would fix some MIPRO paths; COPRO breadth still needs either multi-completion support or an emulated loop of single completions.

### Advanced option gaps around provider controls

`EffortSpec.MAX` exists in dr-llm, but DSPy's `ReasoningEffort` currently stops at `high`, so `max` cannot be requested through `LMConfig`. More importantly for experiment parity:

- OpenRouter reasoning-off controls such as `reasoning_enabled=False` have no dr-dspy v1 equivalent; `EffortSpec.NA` is not the same as an explicit disabled toggle.
- GPT-5 minimal thinking through dr-llm thinking-level controls is not representable by DSPy's `ReasoningEffort`.
- OpenRouter effort controls are not equivalent to DSPy's generic `ReasoningEffort`; current mapping sends them as `BackendRequest.effort`, which OpenRouter rejects for the T1 GPT-5 nano and GPT-OSS configs.
- Suppressing provider-default sampling with explicit empty sampling controls is not exposed on the dr-dspy constructor surface.

References:
- `../dr-llm/src/dr_llm/llm/names.py`
- `../dr-llm/src/dr_llm/llm/providers/concepts/reasoning.py`
- `dspy/core/types/lm_config.py`
- `dspy/clients/dr_llm/base.py`
- `dspy/clients/dr_llm/mapping.py`

Impact: these are not correctness bugs for the default text-only path, but they should be explicit for experiments that rely on custom provider registries, dr-llm's maximum-effort mode, or exact parity with `nl_latents`/`nl-code` catalog controls.

## Alignment Notes

The core request/response boundary mostly aligns with `../dr-llm`: text-only messages are converted to `BackendRequest`, provider/model splitting maps `openai/gpt-4.1-mini` to `ProviderName.OPENAI` plus `gpt-4.1-mini`, unsupported tools/multimodal/structured-output fields are rejected, response provenance is preserved in `provider_data`, error translation maps dr-llm backend/provider errors into the DSPy `LMError` hierarchy, and pool miss-to-hit plus session acquire semantics are covered by tests.

DSPy reasoning effort maps to `BackendRequest.effort` for providers that actually use dr-llm `EffortSpec`, but it does not cover provider-specific `BackendRequest.reasoning`. Capabilities probing through a dedicated `DirectBackend` for pool LMs also matches dr-llm's current design because `PoolBackend` has no public `.capabilities()` API.

Direct path guidance: `DrLlmDirectLM` is ready for text-only programs with `JSONAdapter` or `XMLAdapter`. Configure auth and routing through the dr-llm registry/environment, not `LMProviderOptions`.

Pool path guidance: use `aforward` for cache-first single completions, and use `acquire_samples` only with an explicit stable session identity unless disk logging provides a known-safe session. Use dr-llm `PoolBackend.submit_batch` plus `await_drain` directly for batch pre-fill workflows today.

Experiment parity guidance: keep `nl_latents` pool curves on the raw dr-llm/nl_latents infrastructure for exact replication until dr-dspy exposes provider-specific dr-llm reasoning/config controls and a raw single-message request path. For `nl-code` reproduction, port the code-spec programs and metrics to `TaskSpec`/`Predict`, run with `DrLlmDirectLM` plus `ChatAdapter`, match optimizer compile settings and splits, and disclose remaining LiteLLM-vs-dr-llm wire differences. Do not use the pool backend for optimizer runs unless cached sampling is an intentional new experiment condition.

Compression optimizer guidance: the documented optimizer knobs are present in dr-dspy for MIPROv2, COPRO, GEPA, SIMBA, and InferRules, but the experiment-specific pieces are not framework-native. "Optimize only the representation-policy section" needs experiment-layer TaskSpec composition, compression-aware scoring needs a custom metric over pass rate and representation length, and decoder scaffold accounting needs run metadata.

## dr-llm Verification Notes

The focused review ran:

- `uv run pytest tests/backends/test_direct_backend.py tests/backends/test_pool_backend.py tests/backends/test_converters.py tests/backends/test_fingerprint.py tests/backends/test_validation.py tests/backends/test_async_bridge.py -q` in `../dr-llm`: 37 passed.
- Postgres integration checks in both repos: skipped because no integration DSN was configured.
- Source inspection and focused local scripts compared T1 dr-llm catalog payloads against dr-dspy `LMRequest` to `BackendRequest` mapping.
- The no-provider-call validator check confirmed that all five default T1 configs are rejected through current dr-dspy mapping before any live API call.

Experiment-parity checks not run: live provider calls, `nl-code` session replay, or `nl_latents` curve replay. Useful next checks would be a wire-parity probe comparing dr-llm `build_request_from_config()` payloads to `DrLlmDirectLM` backend requests for the same messages, a prompt-parity diff between `nl-code` `ChatAdapter` rendering and `nl_latents` raw templates, and a one-task HumanEval smoke optimizer after porting the TaskSpecs and metric.
