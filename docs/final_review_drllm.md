# Prompt: Make dr-llm Ready for dr-dspy Direct and Pool Experiments

You are a smart coding agent working in the `dr-llm` repository, currently located at `../dr-llm` relative to this `dr-dspy` checkout. Use this document as your working prompt and status tracker. Update the status tables in this file as you work so a reviewer can tell what is done, what changed, what remains blocked, and which verification commands passed.

The paired DSPy-side review is in `../dr-dspy/docs/final_review_dspy.md`. Read it before editing code. Your goal is not to reproduce every DSPy bugfix inside `dr-llm`; your goal is to make `dr-llm` expose clear, stable backend behavior and provider controls so the updated `dr-dspy` integration can immediately run text-only experiments through either `DrLlmDirectLM` or `DrLlmPoolLM`.

## Current Status

Update these rows as you go. Use one of: `todo`, `in_progress`, `blocked`, `done`, or `not_applicable`.

| Area | Status | Notes |
| --- | --- | --- |
| Read `../dr-dspy/docs/final_review_dspy.md` and this prompt | todo |  |
| Inspect current `dr-llm` backend, pool, fingerprint, provider-control, and docs code | todo |  |
| Provider-specific reasoning/config bridge supports dr-dspy experiment parity needs | todo |  |
| Pool acquisition session identity is explicit and documented for direct `dr-llm` users | todo |  |
| Pool batch-fill workflow docs are clear and still dr-llm-native | todo |  |
| Pool fingerprint and metadata behavior are tested or documented | todo |  |
| `PoolBackend.close()` is idempotent, or current idempotency is verified and documented | todo |  |
| Acquire provenance remains stable for wrappers and direct backend users | todo |  |
| dr-llm v1 unsupported feature scope is documented for DSPy callers | todo |  |
| Exact `nl_latents` reproduction path remains dr-llm/native unless prompt parity is added elsewhere | todo |  |
| Direct backend smoke checks pass | todo |  |
| Pool backend smoke/integration checks pass or are blocked only by missing DSN | todo |  |
| Live OpenRouter `gpt-5-nano` endpoint test passes | todo | Required for the plan to be considered fully tested; use the local OpenRouter API key environment variable. |
| Cross-repo readiness note for dr-dspy is written | todo |  |

## Target Outcome

At the end of this work:

- `dr-llm` can express the provider-specific controls required by the reviewed compression experiments, especially OpenRouter reasoning-off, OpenRouter provider-specific reasoning effort, OpenAI minimal thinking, Google thinking-off, ordinary sampling controls, and an explicit "no sampling override" case.
- `dr-dspy` can map its updated direct and pool wrappers onto those `dr-llm` controls without guessing, falling back to lossy generic `effort`, or changing cache fingerprints unexpectedly.
- Direct experiments can run through `DrLlmDirectLM` with explicit `RunContext`, adapter, model controls, and audit logs.
- Pool experiments can run through `DrLlmPoolLM` with stable acquisition session identity, clear cache/fingerprint behavior, and predictable lifecycle behavior.
- Exact `nl_latents` compression-curve reproduction remains on the raw `nl_latents` plus `dr-llm` harness until a raw single-message request path exists in `dr-dspy`; do not claim DSPy `Predict(TaskSpec)` prompts are bit-equivalent to those pools.
- The repository contains tests or docs that prevent future drift on these boundaries.

## Required Context

Read these before implementing:

- `../dr-dspy/docs/final_review_dspy.md`
- `../dr-dspy/docs/final_review_drllm.md` after you start editing this prompt as your tracker
- `src/dr_llm/backends/models.py`
- `src/dr_llm/backends/pool.py`
- `src/dr_llm/backends/fingerprint.py`
- `src/dr_llm/llm/config.py`
- `src/dr_llm/llm/names.py`
- `src/dr_llm/llm/providers/concepts/reasoning.py`
- `src/dr_llm/llm/providers/impls/openrouter/request_controls.py`
- `src/dr_llm/llm/providers/impls/openai/request_controls.py`
- `src/dr_llm/llm/providers/impls/google/request_controls.py`

Also inspect these cross-repo callers and fixtures when validating compatibility:

- `../dr-dspy/dspy/clients/dr_llm/base.py`
- `../dr-dspy/dspy/clients/dr_llm/mapping.py`
- `../dr-dspy/dspy/clients/dr_llm/contract.py`
- `../dr-dspy/dspy/clients/dr_llm/pool.py`
- `../nl_latents/src/nl_latents/sampling/llm/catalog.py`
- `../nl_latents/scripts/code_comp_t1/shared_config.sh`
- `../nl_latents/src/nl_latents/sampling/encoder/request.py`
- `../nl_latents/src/nl_latents/sampling/decoder/request.py`

## Working Rules

- Make the smallest coherent `dr-llm` changes that let `dr-dspy` integrate cleanly. Do not redesign experiment harnesses unless the existing surface cannot support the required request shapes.
- Preserve `dr-llm` as the source of truth for provider-native controls. `dr-dspy` should adapt to this API, not duplicate provider-specific payload construction.
- Keep raw `nl_latents` pool reproduction separate from DSPy prompt generation unless you add and test a raw request path explicitly.
- Add regression tests for behavior changes. If a check needs Postgres or live provider credentials, add a skipped integration test or document the exact command and blocker.
- Update the status table and the "Implementation Log" as each phase completes.

## Phase 1: Establish the Cross-Repo Contract

Status: `todo`

Write down, in code comments, docs, or tests as appropriate, the contract between `dr-llm` and `dr-dspy`:

- `dr-llm` owns provider/model routing, provider-native reasoning controls, provider-native sampling controls, backend request validation, fingerprinting, pool acquisition semantics, and aggregate acquire provenance.
- `dr-dspy` owns `TaskSpec`/adapter prompt rendering, DSPy `LMRequest` to `BackendRequest` mapping, transparency/audit logging, `RunContext`, and DSPy optimizer call behavior.
- Metadata may flow from DSPy into `BackendRequest.metadata`, but metadata and extensions must not become cache-key or claim-isolation fields unless a deliberate breaking change is made.
- Exact `nl_latents` pool replication uses raw single-user-message prompts and the existing `dr-llm`/`nl_latents` pool harness; DSPy `Predict(TaskSpec)` prompt shape is a different experiment condition.

Acceptance checks:

- There is a discoverable doc or test assertion that states metadata is not included in the backend fingerprint.
- There is a discoverable doc or test assertion that states session identity, not metadata, controls no-replacement acquisition grouping.
- There is a note for `dr-dspy` maintainers listing the exact `dr-llm` request fields they should map for provider-specific reasoning and sampling.

## Phase 2: Preserve and Expose Provider-Specific Controls

Status: `todo`

The reviewed compression experiments used these request controls:

| Config family | Model string | Required provider-native controls |
| --- | --- | --- |
| MiMo off | `openrouter/xiaomi/mimo-v2-flash` | OpenRouter `reasoning.enabled=false`, `temperature=0.7`, `top_p=0.95` |
| Nemotron off | `openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5` | OpenRouter `reasoning.enabled=false`, `temperature=0.7`, `top_p=0.95` |
| GPT-OSS low | `openrouter/openai/gpt-oss-20b` | OpenRouter provider-specific `reasoning.effort=low`, not generic backend `effort` |
| GPT-5 nano low through OpenRouter | `openrouter/openai/gpt-5-nano` | OpenRouter provider-specific `reasoning.effort=low`, not generic backend `effort` |
| GPT-5 nano minimal through OpenAI | `openai/gpt-5-nano` | OpenAI `thinking_level=minimal`, no sampling override |
| Gemini flash-lite off | `google/gemini-2.5-flash-lite` | Google `thinking_level=off`, `temperature=0.7`, `top_p=0.95` |

Implementation guidance:

- Verify that current `BackendRequest.reasoning` and provider request-control models can represent every row above without lossy conversion.
- If the existing public model is too awkward for `dr-dspy`, add a small explicit construction or parsing helper in `dr-llm` rather than making `dr-dspy` know provider internals.
- Make sure "provider-specific reasoning effort" is distinct from generic `BackendRequest.effort`. The reviewed OpenRouter GPT-5 nano and GPT-OSS requests must not be represented as generic backend effort.
- Preserve an explicit way to suppress provider-default sampling overrides when a catalog entry means "no sampling override".
- Confirm whether `EffortSpec.MAX` is intentionally supported and document how callers should request it. DSPy's current generic `ReasoningEffort` stops at `high`, so a provider-native bridge may be needed for `max`.

Acceptance checks:

- Add or update tests that build the six request families above and verify provider payload controls without live API calls.
- Add a small parity test or fixture comparing `nl_latents` catalog request construction with direct `dr-llm` request construction for the five default T1 configs.
- If you cannot add the parity test inside `dr-llm` because it would depend on `nl_latents`, document the exact external parity command and expected payload fields.

## Phase 3: Make Pool Acquisition Semantics Hard to Misuse

Status: `todo`

`PoolBackend.aacquire(request, session_id, n)` implements no-replacement sampling by session. The unsafe behavior happens when callers accidentally reuse a session across independent experiments or accidentally generate a new session for repeated calls in one experiment.

Implementation guidance:

- Ensure docs and tests make `session_id` mandatory in spirit, even if the API can accept a generated value through a wrapper.
- Explain that `session_id` controls acquisition state and metadata does not.
- Confirm whether `PoolBackend` should reject empty session IDs or normalize them consistently.
- Add examples for stable experiment session IDs, for example `experiment-name:split:seed` or a caller-provided run ID.
- Coordinate with the DSPy-side fix: `dr-dspy` should stop relying on second-resolution log timestamps for unique pool acquisition sessions.

Acceptance checks:

- A test or doc example demonstrates two calls with the same `session_id` claiming without replacement.
- A test or doc example demonstrates that different `session_id` values do not share claim state.
- A doc note warns against deriving session IDs from low-resolution timestamps.

## Phase 4: Clarify Pool Batch Fill Versus Cache-First Calls

Status: `todo`

`PoolBackend` supports a native batch-fill workflow with `submit_batch`, drain, and then acquire. `DrLlmPoolLM.aforward` is cache-first single-completion behavior; on a miss it generates and inserts one response. These are different experiment workflows.

Implementation guidance:

- Keep batch pre-fill as a direct `dr-llm` workflow unless you intentionally add a higher-level wrapper.
- Document the lifecycle: seed or submit a request grid, drain workers, acquire with a stable session ID, aggregate results.
- Document that `DrLlmPoolLM` does not reproduce `nl_latents` grid axes, encoder-to-decoder lineage, budget bindings, compression baselines, or curve aggregation.

Acceptance checks:

- There is a minimal docs example for direct `PoolBackend` batch-fill and acquire.
- There is a clear contrast between `PoolBackend` batch-fill and wrapper-level cache-first `aforward`.

## Phase 5: Stabilize Pool Lifecycle and Provenance

Status: `todo`

The DSPy review found that shallow-copied `DrLlmPoolLM` wrappers can share a backend while each wrapper tracks its own `_closed` flag. `dr-dspy` should guard against use-after-close, but `dr-llm` can reduce blast radius by making backend teardown robust.

Implementation guidance:

- Verify whether `PoolBackend.close()` is idempotent. If not, make it idempotent.
- Verify whether async close/drain paths are safe to call multiple times or after partial initialization.
- Preserve `AcquireResult(responses, claimed_from_cache, generated)` semantics. `dr-dspy` may expose or log aggregate provenance later, so avoid changing field names without a coordinated update.
- Ensure per-response provenance such as cache/provider source remains available in response metadata or provider data.

Acceptance checks:

- Add or update a unit test for calling close twice.
- Add or update a unit test that acquire provenance counts distinguish cache claims from generated responses.
- If close-after-use behavior depends on external resources, document expected exceptions.

## Phase 6: Fingerprint and Metadata Contract

Status: `todo`

The current desired behavior is that generation-relevant request fields define `request_fingerprint`, while metadata and extensions do not. This lets run-specific tags avoid fragmenting the cache.

Implementation guidance:

- Confirm the fingerprint excludes `metadata` and `extensions`.
- Confirm the fingerprint includes provider/model, messages, generation controls, provider-native reasoning controls, and any other fields that can affect model output.
- If provider-specific reasoning controls are added or re-shaped, make sure they are included in the fingerprint.

Acceptance checks:

- A test proves two requests differing only by metadata share a fingerprint.
- A test proves two requests differing by provider-specific reasoning controls do not share a fingerprint.
- Docs warn that metadata is not claim isolation.

## Phase 7: Document dr-llm v1 Scope for DSPy Callers

Status: `todo`

The first stable `dr-dspy` bridge should target text-only programs:

- Expected fit: `Predict`, `ChainOfThought`, `Evaluate`, text-only metrics, and explicit `RunContext` with `JSONAdapter` or `XMLAdapter`.
- Not supported through v1 backends: tools, tool-call history, ReAct/ReActV2 tool agents, CodeAct execution loops, multimodal parts, native structured-output response formats, stop sequences if still unsupported, logprobs, and prompt-cache controls.

Implementation guidance:

- Keep unsupported feature errors typed and early.
- Do not make `dr-llm` silently ignore unsupported request fields.
- Make docs clear enough that a `dr-dspy` user can choose direct, pool, or plain LiteLLM without reading provider internals.

Acceptance checks:

- Docs list the supported and unsupported DSPy-facing feature surface.
- Tests still prove unsupported fields fail before opaque provider errors.

## Phase 8: Experiment Parity Guidance

Status: `todo`

Do not collapse the two reviewed experiment families:

- `nl_latents` compression curves: raw `dr-llm`/pool workflow, raw single-user-message encoder and decoder prompts, pool grids, compression baselines, representation compression ratio against decoder pass rate.
- `nl-code` DSPy optimization and full-5x eval: legacy DSPy-style programs and optimizers, now needing a `TaskSpec`, explicit `RunContext`, and adapter-based port in `dr-dspy`.

Implementation guidance:

- For exact `nl_latents` replay, keep using `nl_latents` request builders or direct `dr-llm` `build_request_from_config()`-style construction.
- For new `dr-dspy` experiments, disclose that `Predict(TaskSpec)` prompt rendering is a new prompt condition unless a raw LM request path is used.
- For optimizer experiments, prefer `DrLlmDirectLM` until pool-backed cached sampling is an intentional condition. Optimizers can copy LMs and issue proposal calls with `n`; coordinate with `dr-dspy` on `n=1` and multi-completion behavior.

Acceptance checks:

- A doc section explains which path to use for exact replay versus new DSPy experiments.
- A final readiness note states which experiment family is ready to run immediately and which one remains native-only.

## Phase 9: Verification Commands

Status: `todo`

Run the relevant checks in `dr-llm`. Update the command, result, and notes columns.

| Command | Result | Notes |
| --- | --- | --- |
| `uv run pytest tests/backends/test_direct_backend.py tests/backends/test_pool_backend.py tests/backends/test_converters.py tests/backends/test_fingerprint.py tests/backends/test_validation.py tests/backends/test_async_bridge.py -q` | todo |  |
| Provider-control unit tests you added or updated | todo |  |
| Pool lifecycle/provenance tests you added or updated | todo |  |
| Postgres pool integration test with `DR_LLM_TEST_DATABASE_URL` or `DR_LLM_DATABASE_URL` | todo |  |
| Live OpenRouter smoke test against `openrouter/openai/gpt-5-nano` | todo | Required before marking the plan fully tested. Use the repo's local OpenRouter API key environment variable, expected to be `OPENROUTER_API_KEY`, and verify the call actually reaches OpenRouter, returns nonempty text, and records OpenRouter/provider provenance. |

If Postgres credentials are unavailable, mark only the Postgres row `blocked` and include the missing environment variable or service. The OpenRouter `gpt-5-nano` live test is part of the full acceptance bar; if `OPENROUTER_API_KEY` is unexpectedly unavailable, mark the row `blocked`, explain the missing environment variable, and do not describe the plan as fully tested.

For the live OpenRouter test, add or run a checked-in live test when possible. A CLI smoke command is acceptable only if it verifies the response and provenance, for example an equivalent of `uv run dr-llm query --provider openrouter --model openai/gpt-5-nano --message "Return exactly: dr-llm live ok"` plus an assertion that the returned content is nonempty and came from the OpenRouter provider path.

## Cross-Repo Readiness Checklist for dr-dspy

Status: `todo`

Before declaring this done, write a short note for the `dr-dspy` implementer that answers:

- What exact `BackendRequest` fields should `dr-dspy` set for OpenRouter reasoning-off?
- What exact fields should it set for OpenRouter provider-specific effort?
- What exact fields should it set for OpenAI minimal thinking and Google thinking-off?
- How should it represent "no sampling override" versus explicit `temperature`/`top_p`?
- Which fields affect the pool fingerprint?
- What does `dr-dspy` need to pass as stable `session_id` for no-replacement pool acquisition?
- Is `PoolBackend.close()` idempotent?
- Are aggregate acquire provenance fields stable?
- Which backend features remain unsupported and should be rejected in `dr-dspy` before provider calls?

## Implementation Log

Append short entries as you work:

| Date | Agent | Change | Verification |
| --- | --- | --- | --- |
|  |  |  |  |

## Original Review Findings Incorporated

This prompt incorporates the following review findings from the prior `dr-llm` final review:

- Cross-boundary coordination with `dr-dspy` disk log session identity, provider-specific reasoning gaps, DSPy prompt-shape differences, and pool fingerprint differences.
- Exact compression reproduction requires choosing between the raw `nl_latents`/`dr-llm` family and the DSPy optimizer family.
- The reviewed compression surface used MiMo, Nemotron, GPT-OSS 20B, GPT-5 nano, and Gemini Flash Lite with budgets `32`, `64`, `128`, `256`, `512`, and `1024`.
- The default T1 configs use `humaneval-plus`, budgets `64,128,256`, one encoder sample and one decoder sample per config, same-as-encoder decoder mode, and five default LLM config IDs:
  - `openrouter/xiaomi/mimo-v2-flash/off/v1`
  - `openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5/off/v1`
  - `openrouter/openai/gpt-5-nano/low/v1`
  - `openrouter/openai/gpt-oss-20b/low/v1`
  - `openai/gpt-5-nano/minimal/v1`
- Raw `nl_latents` encoder and decoder prompts are single user messages, not DSPy `Predict(TaskSpec)` prompts.
- `nl_latents` pools and `DrLlmPoolLM` pools use different seeding and keying systems.
- Pool acquisition depends on explicit stable session identity.
- Batch-fill remains a native `dr-llm` workflow.
- Pool fingerprints exclude metadata and extensions by design, but provider-output-affecting controls must be included.
- `dr-llm` v1 is intentionally text-only for the DSPy bridge.
- `LMConfig(n=1)` and multi-completion proposal behavior are primarily `dr-dspy` contract issues unless `dr-llm` adds native or emulated multi-completion support.
- `AcquireResult` aggregate provenance is already present in `dr-llm`; preserve it for future wrapper exposure.
