# dr-llm Text-Only DSPy Pilot

This note describes the supported `dr-dspy` path for post-review experiments
that use `DrLlmDirectLM` or `DrLlmPoolLM`.

## Supported v1 Scope

Use dr-llm backends for text-only DSPy programs:

- `Predict` and `ChainOfThought` with `TaskSpec` inputs/outputs.
- `Evaluate` over non-empty text-only devsets.
- Optimizer task calls that use single completions, including `LMConfig(n=1)`.
- `DrLlmDirectLM.aforward` for direct provider calls.
- `DrLlmPoolLM.aforward` for cache-first single completions.
- `DrLlmPoolLM.acquire_samples_result` for explicit-session no-replacement pool acquisition.

Unsupported v1 features fail early: tools and tool-call history, ReAct-style
tool agents, multimodal parts, native structured `response_format`, stop
sequences, logprobs, prompt-cache controls, unsupported roles, arbitrary
`LMConfig.extensions`, and reasoning fields not represented by the dr-llm
provider-control bridge.

## Provider Controls

Generic DSPy `LMConfig(reasoning={"effort": ...})` maps to generic
`BackendRequest.effort`. Do not use it for provider-native OpenRouter/OpenAI/
Google controls.

Use `DrLlmProviderControls` or `LMConfig.extensions["dr_llm"]` for provider
controls that must affect dr-llm request fingerprints:

```python
from dspy.clients.dr_llm import DR_LLM_EXTENSION_KEY, DrLlmDirectLM
from dspy.core.types import LMConfig
from dspy.predict.call_options import PredictOptions

lm = DrLlmDirectLM("openrouter/openai/gpt-5-nano", max_tokens=512)

options = PredictOptions(
    config=LMConfig(
        extensions={
            DR_LLM_EXTENSION_KEY: {
                "reasoning": {"kind": "openrouter", "effort": "low"},
                "sampling": {"temperature": 0.7, "top_p": 0.95},
            }
        }
    )
)
```

Default T1 controls:

- OpenRouter reasoning off: `{"reasoning": {"kind": "openrouter", "enabled": false}}`.
- OpenRouter provider effort: `{"reasoning": {"kind": "openrouter", "effort": "low"}}`.
- OpenAI minimal thinking: `{"reasoning": {"kind": "openai", "thinking_level": "minimal"}}`.
- Google thinking off: `{"reasoning": {"kind": "google", "thinking_level": "off"}}`.
- Explicit sampling: `{"sampling": {"temperature": 0.7, "top_p": 0.95}}`.
- No sampling override: `{"sampling": {"temperature": null, "top_p": null}}`, which resolves to `BackendRequest.sampling=None` before dr-llm fingerprinting.

`metadata` is forwarded to `BackendRequest.metadata`, but dr-llm fingerprints
exclude metadata and extensions. Use generation-relevant fields, a pool
namespace/config change, or an acquisition `session_id` for isolation.

## Pool Sessions

`DrLlmPoolLM.aforward` is cache-first single-completion behavior and does not
claim no-replacement samples. Use acquisition only when no-replacement sampling
is intentional:

```python
result = await pool.acquire_samples_result(request, n=8, run=run, session_id="exp:split:seed")
responses = result.responses
print(result.claimed_from_cache, result.generated)
```

Pass a stable explicit `session_id` such as `experiment-name:split:seed`.
Do not derive acquisition sessions from low-resolution timestamps. Metadata
does not isolate claims.

Acquisition helpers are not normal `BaseLM.__call__` calls, so DSPy memory and
disk call logs should not be treated as the source of truth for acquisition
provenance. Experiment runners that call `acquire_samples_result(...)` should
persist `claimed_from_cache`, `generated`, and each response's `source`,
`sample_id`, and `request_fingerprint` provider data in their own artifacts.

## Minimal `nl-code` TaskSpec Pilot

The fastest post-review DSPy pilot should use direct dr-llm calls, not pools:

- Define encoder and decoder `TaskSpec` classes with explicit field descriptions.
- Create `RunContext.create(lm=DrLlmDirectLM(...), adapter=JSONAdapter())`.
- Run one manual AE/minify policy and one optimized policy over the same task IDs.
- Use `MIPROv2` first with no demos and a small train/val split.
- Keep the decoder task, adapter, budgets, lossless layer, and evaluator fixed.

`JSONAdapter` keeps its existing permissive parsing default
(`allow_json_repair=True`) under strict transparency. Prose-wrapped JSON and
multiple JSON objects may be repaired into a valid output object and are still
recorded in the call log for audit. Use `JSONAdapter(allow_json_repair=False)`
only when the experiment should reject repaired completions instead of salvaging
them.

## Experiment Footguns

For optimizer and sampling experiments, keep these defaults explicit in run
notes:

- `PredictOptions.trace` defaults to enabled, so ordinary `Predict` calls append
  to the run optimization trace unless `trace=False` is passed.
- `BestOfN` and `Refine` return the best prediction even when no sample meets
  `threshold`; inspect `get_sampling_metadata(prediction).threshold_met` when
  threshold success matters.
- `BestOfN` and `Refine` default `fail_count` to `num_samples`, so parse or LM
  failures consume the same sampling budget unless a smaller `fail_count` is set.
- Sampling modules are copied per attempt; if a module cannot be deep-copied,
  DSPy warns and isolation may be weaker.
- Callback exceptions are logged as warnings. Treat callback-dependent metrics or
  telemetry as best-effort unless the callback path has its own assertions.
- `collect_trace_data(..., raise_on_error=False)` may return fewer records than
  the input set when examples fail.

This is a new DSPy prompt condition. Exact `nl_latents` compression-curve
replay remains on the raw `nl_latents`/`dr-llm` pool harness unless a raw
single-message DSPy adapter path is added and verified.
