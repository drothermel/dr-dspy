# Optimize Decoder Format Plan

## Purpose

This experiment is the first optimization stage before optimizing the full code
compression pathway. The goal is to freeze a validated decoder prompt/template
before changing the encoder prompt.

The larger pipeline is:

```text
code sample
  -> encoder LLM
  -> description
  -> lossless compression
  -> lossless decompression
  -> decoder LLM
  -> reconstructed code
  -> tests
```

This stage isolates the decoder:

```text
ground-truth docstring/description + decoder template
  -> decoder LLM
  -> generated Python code
  -> formatting/functional-recovery evaluation
```

The immediate objective is not compression. The objective is to learn how to run
prompt optimization reliably and to select decoder formatting that produces
valid, testable Python before the encoder is introduced.

## Starting Assumptions

- DSPy's default prompt formatting is bypassed for this work. Optimizer
  candidates are rendered into explicit prompt/template text and evaluated by
  the sibling experiment runtime.
- LLM generations route through `../dr-bottleneck/` and `../dr-providers/`
  rather than the normal DSPy adapter formatting path.
- Coordination between `dr-dspy` and `dr-bottleneck` is queue-based, not
  filesystem-based. Filesystem exports may remain useful for inspection, but
  they are not the control plane.
- Initial runs should use one model: `openrouter/openai/gpt-oss-20b/low/v1`.
  This keeps cost and variance down while the optimization setup is being
  validated.
- The first optimizer target should be easy and interpretable: AST-parseable
  Python output.
- After parseability is reliable, the target can move to HumanEval/HumanEval+
  test pass rate.
- The final decoder prompt should be frozen before encoder-only optimization
  starts, so later gains or regressions can be attributed to the encoder prompt.

## Relevant Context

General project context lives in `docs/starting_state.md`.

Local DSPy optimizer surface:

- `dspy/teleprompt/` - optimizers and compile flows to inspect or adapt.
- `dspy/adapters/` - likely area to bypass or customize prompt rendering and
  output parsing.
- `dspy/task_spec/` - task contracts if this experiment is expressed as DSPy
  TaskSpecs.
- `dspy/runtime/` - run context, logging, tracing, and async execution support.

Provider/client surface:

- `../dr-providers/README.md` - typed OpenRouter client usage.
- `../dr-providers/src/dr_providers/query/from_prompt.py` - simple prompt-based
  query helper.
- `../dr-providers/src/dr_providers/query/request.py` - typed request model.
- `../dr-providers/src/dr_providers/query/response.py` - typed response model.

Current black-box pipeline surface:

- `../dr-bottleneck/configs/workflows/humaneval_encode_decode.yaml` - current
  encode/decode workflow.
- `../dr-bottleneck/src/dr_bottleneck/experiments/humaneval.py` - HumanEval+
  loading, source construction, job expansion, and conversion into `dr-code`
  attempt records.
- `../dr-bottleneck/src/dr_bottleneck/candidate_eval/` - queue-facing
  candidate-evaluation request/result schema, decoder-only evaluator, and
  candidate worker.
- `../dr-bottleneck/scripts/preview_humaneval_prompts.py` - prompt rendering
  helper for inspecting workflow prompts without LLM calls.
- `../dr-bottleneck/scripts/run_humaneval_demo.py` - current HumanEval sweep
  entry point.
- `../dr-code/src/dr_code/pipeline/export.py` - exposes persisted parse/test
  outcomes for candidate-result aggregation.
- `dspy/integrations/bottleneck/` - `dr-dspy` queue client for submitting
  candidates and collecting optimizer-facing results.

## Current Implementation State

The first integration slice is implemented and committed across the sibling
repos:

- `dr-dspy` can submit candidate-evaluation requests and collect matching
  results through `BottleneckQueueEvaluator`.
- `dr-bottleneck` owns the candidate request/result models and a
  `dr_bottleneck.candidate_eval.worker` process that consumes request jobs,
  evaluates decoder-only candidates, and publishes result jobs.
- `dr-bottleneck` derives decoder-only `{signature}` and
  `{encoded_description}` values from HumanEval prompts, then runs a one-step
  decode workflow through the normal `dr-providers` call path.
- `dr-code` parse/test outcomes are reconstructed from persisted queue events
  and summarized into candidate-level metrics and failure buckets.

Supported now:

- `decoder_format` and `decoder_correctness` candidate phases.
- Signature-side-channel and description-only variants, selected by template
  text usage.
- Baseline/manual decoder templates and slot-addendum templates.
- AST parse rate, test pass rate, failure buckets, per-example feedback, and
  decoder-input byte/compression summaries.
- Scaling through queue workers, task slices, lane lists, budgets, repeats, and
  bottleneck/code-eval worker counts.

Not supported yet:

- Full `encoder_full_path` optimization.
- Candidate-specific two-step encode/decode workflows.
- Using encoder-generated descriptions as decoder input.
- Meaningful correctness-then-compression scoring over encoder output.
- Freezing a selected decoder prompt and evaluating encoder candidates against
  it.

## Required Integration Capabilities

The `dr-dspy` <=> `dr-bottleneck` integration should support these black-box
evaluation shapes:

1. `Blackbox_Dec_Only_Eval(sample)`: fixed decoder prompt evaluation, with
   batch evaluation over predefined samples and inputs.
2. `Blackbox_Dec_Only_Eval(sample, dec_config)`: templated decoder prompt
   evaluation, with batch evaluation over predefined samples and inputs.
3. `Blackbox_Enc_Dec_Eval(sample, enc_config, dec_config)`: templated encoder
   prompt plus fixed decoder prompt evaluation, with batch evaluation over
   predefined samples and inputs.
4. `Blackbox_Dec_Only_Iterative(sample)` and
   `Blackbox_Dec_Only_Iterative(sample, dec_config)`: fixed or templated
   decoder prompt evaluation with continuous queuing of new configuration,
   input, and sample combinations.
5. `Blackbox_Enc_Dec_Iterative(sample, enc_config, dec_config)`: templated
   encoder prompt plus fixed decoder prompt evaluation with continuous queuing
   of new configuration, input, and sample combinations.

## Clean-Break Integration Direction

For the next implementation pass, prefer a clean break over compatibility
shims. `dr-dspy` should own experiment expansion and emit concrete executable
jobs. `dr-bottleneck` should own only the executable job kinds and runtime
execution. `dr-code` should own code parsing/testing behavior and the
HumanEval+ suite contract.

This means:

- `dr-dspy` expands datasets, samples, repeats, optimizer candidates, prompt
  templates, slot values, and provenance metadata.
- `dr-dspy` submits concrete workflow jobs to `dr-bottleneck`, rather than a
  high-level candidate-evaluation request.
- `dr-bottleneck` executes generic job kinds and preserves opaque,
  JSON-serializable metadata.
- `dr-bottleneck` does not know about optimizer slots, decoder variants,
  candidate phases, grid search, COPRO, or HumanEval split logic.
- `dr-code` evaluates functional recovery by trying arity-compatible top-level
  functions, not by requiring exact entry-point-name recovery.

The queue-facing workflow job contract should live in:

```text
dr_bottleneck.workflow_jobs.spec
```

This module should contain only stable payload/config/output schemas for
workflow job execution. Runtime handlers, provider clients, persistence models,
and experiment-specific models belong elsewhere. `dr-dspy` should import these
schemas directly from `dr-bottleneck` instead of mirroring them.

The initial job kinds are:

```python
class JobKind(StrEnum):
    LLM_QUERY_STATIC = "llm_query_static"
    LLM_QUERY_FROM_PREVIOUS = "llm_query_from_previous"
    EVAL_FROM_PREVIOUS = "eval_from_previous"


class LLMQueryStaticConfig(BaseModel):
    metadata: dict[str, Any]
    model_id: str
    prompt: str
```

`llm_query_static`: send `prompt` to the configured model and emit
`{"output_text": model_response_text, "metadata": metadata}`.

```python
class LLMQueryFromPreviousConfig(BaseModel):
    metadata: dict[str, Any]
    model_id: str
    prompt_template: str
    placeholder: str
```

`llm_query_from_previous`: read `previous_step["output_text"]`, replace exactly
the declared `{placeholder}` in `prompt_template`, send the prompt to the
configured model, and emit
`{"output_text": model_response_text, "metadata": metadata}`.

```python
class EvalFromPreviousConfig(BaseModel):
    metadata: dict[str, Any]
    suite: Literal["humaneval_plus"]
    task_id: str
    decoder_input: str
```

`eval_from_previous`: read `previous_step["output_text"]` as generated code,
evaluate it against `suite` / `task_id`, and emit structured eval results plus
metadata.

Workflow topology should be explicit:

```python
class WorkflowStepSpec(BaseModel):
    name: str
    job_kind: JobKind
    input_queue: str
    output_queue: str | None = None


class WorkflowJobPayload(BaseModel):
    workflow_id: str
    steps: tuple[WorkflowStepSpec, ...]
    step_configs: dict[
        str,
        LLMQueryStaticConfig | LLMQueryFromPreviousConfig | EvalFromPreviousConfig,
    ]
    metadata: dict[str, Any]
```

Runtime resolution:

```text
current_step = steps[job.step_index]
config = step_configs[current_step.name]
previous_output = job.step_outputs[steps[job.step_index - 1].name]
```

For first steps, previous output must be absent or ignored. For previous-output
job kinds, missing previous output is `infra_non_retryable`.

Standard step outputs should be named objects stored in
`job.step_outputs[step_name]` as JSON dictionaries:

```python
class LLMQueryOutput(BaseModel):
    output_text: str
    metadata: dict[str, Any]


class CandidateFunction(BaseModel):
    name: str
    positional_arity: int


class EvalFromPreviousOutput(BaseModel):
    metadata: dict[str, Any]
    parse_success: bool
    test_pass_rate: float
    all_tests_passed: bool
    selected_function_name: str | None
    candidate_functions: tuple[CandidateFunction, ...]
    expected_entry_point_present: bool
    failure_bucket: str | None
```

LLM jobs may expose `output_text` for convenience, but `LLMQueryOutput` is the
canonical output shape.

`SamplingConfigId` should be defined in `dr-bottleneck` for these LLM job
kinds. For the first pass, make the model setup string the canonical config and
parse it with a fixed versioned regexp into a typed request config. Always log
both the original id and the parsed request settings. A candidate format is:

```text
openrouter.openai__gpt-oss-20b.reasoning-low.temp-0p7.top-p-0p95.v0
openrouter.google__gemini-3.1-flash-lite.reasoning-off.temp-0p7.top-p-0p95.v0
```

Grammar:

```text
<provider>.<model_escaped>.reasoning-<reasoning>.temp-<temp>.top-p-<top_p>.v<version>
```

Use `__` to escape provider-model slashes in `model_escaped`.

Regex:

```python
SAMPLING_CONFIG_ID_RE = re.compile(
    r"^(?P<provider>[a-z][a-z0-9_-]*)\."
    r"(?P<model>[a-zA-Z0-9_.-]+(?:__[a-zA-Z0-9_.-]+)*)\."
    r"reasoning-(?P<reasoning>off|low|medium|high)\."
    r"temp-(?P<temperature>[0-9]+p[0-9]+)\."
    r"top-p-(?P<top_p>[0-9]+p[0-9]+)\."
    r"v(?P<version>[0-9]+)$"
)
```

Parsing rules:

- `model = model_escaped.replace("__", "/")`
- `temp-0p7 -> 0.7`
- `top-p-0p95 -> 0.95`
- `reasoning-off -> ReasoningSpec(enabled=False)`
- `reasoning-low -> ReasoningSpec(effort=LOW)`

Allowed values for the first pass:

- `provider`: `openrouter`
- `reasoning`: `off`, `low`, `medium`, `high`
- `version`: `0`

Keep these ids as payload values and provenance values, not queue names.

Reference limits:

- RabbitMQ queue names are limited to 255 bytes of UTF-8:
  <https://www.rabbitmq.com/docs/queues>
- RabbitMQ 4.3 defaults `max_message_size` to 16 MiB, with a maximum allowed
  value of 512 MiB: <https://www.rabbitmq.com/docs/limits>
- MongoDB BSON documents are limited to 16 MiB. Database names and namespace
  names have separate shorter limits, but long ids are fine as ordinary string
  values as long as the full document stays under the BSON limit:
  <https://www.mongodb.com/docs/manual/reference/limits/>

## First-Pass Metrics

Define the first-pass optimizer metrics in `dr-dspy` and require each optimizer
run to declare one metric id. `dr-bottleneck` should return structured outputs;
`dr-dspy` should compute optimizer-facing scores.

Initial metric ids:

- `parse_binary`: `1.0` if code parsing succeeds, else `0.0`.
- `test_pass_binary`: `1.0` if all selected-function tests pass, else `0.0`.
- `test_pass_rate`: fraction of tests passed by the selected arity-compatible
  function.
- `compression_ratio_vs_ground_truth`: compressed encoder-output bytes divided
  by ground-truth code bytes.
- `pass_rate_with_bounded_compression_penalty`: bounded reward in `[0, 1]`
  derived from test pass rate and a clamped compression ratio.

For the bounded compression metric, treat `test_pass_rate` as `[0, 1]` and clamp
`compression_ratio` to `[0.01, 4.0]`, where lower is better. Normalize with:

```text
compression_score = (4.0 - clamped_compression_ratio) / (4.0 - 0.01)
reward = test_pass_rate * ((1.0 - weight) + weight * compression_score)
```

With `weight` in `[0, 1]`, the final reward remains in `[0, 1]` and failed
functional recovery keeps the reward at `0.0`.

Keep bounded compression constants in metric config rather than hardcoding them:

```python
class BoundedCompressionMetricConfig(BaseModel):
    metric_id: Literal["pass_rate_with_bounded_compression_penalty"]
    weight: float = Field(default=0.10, ge=0.0, le=1.0)
    min_compression_ratio: float = 0.01
    max_compression_ratio: float = 4.0
```

Correctness should dominate compression in the first pass. Start with
`weight = 0.10`; use `0.05` if compression should be nearly invisible during
early prompt selection. With `weight = 0.10`, compression can move a perfect
solution between `0.90` and `1.00`, but cannot rescue a candidate with no
functional recovery.

Compression for this phase should happen in `dr-dspy` analysis/scoring after a
workflow completes. Reuse `dr-code`'s zstd22 helper rather than reimplementing
compression. The workflow output should include the relevant text artifacts
such as encoder output, decoder output, eval result, and metadata.

## Failure Semantics

Use three failure classes across workflow execution and scoring:

```text
infra_non_retryable
infra_retryable
model_or_eval_outcome
```

Failure classification:

- Missing previous step `output_text`: `infra_non_retryable`. This means the
  workflow/job contract was broken. Log it, fail the job, and do not retry.
- Malformed placeholder config: `infra_non_retryable`. Bad job config should
  fail fast with no retry.
- Provider transport, rate-limit, timeout, or server failure:
  `infra_retryable`. Retry with bounded attempts and backoff.
- Provider request succeeds but returns no usable text:
  `model_or_eval_outcome`. This is distinct from provider failure and should
  score as `0.0` parse/test performance.
- Parse failure: `model_or_eval_outcome`; `parse_binary = 0.0` and
  `test_pass_rate = 0.0`.
- No top-level functions: `model_or_eval_outcome`; `test_pass_rate = 0.0`.
- No arity-matching functions: `model_or_eval_outcome`;
  `test_pass_rate = 0.0`.

Reports should distinguish provider-call failures from successful provider calls
that return empty or unusable text.

## Correlation Metadata

Every expanded workflow job should include these required metadata keys:

- `experiment_id`
- `optimizer_run_id`
- `candidate_id`
- `task_id`
- `split`
- `round_index`
- `seed_index`
- `metric_id`
- `candidate_surface`
- `evaluator_model_id`

Recommended optional keys:

- `workflow_id`
- `candidate_kind`
- `proposal_phase`
- `proposal_model_id`

For COPRO-style optimization, `proposal_phase` should use values such as
`baseline`, `grid`, `initial`, and `refine`. `candidate_surface` should use
values such as `bounded_slots`, `full_prompt`, and `encoder_prompt`.

Do not require `sample_id` unless it is separately defined as a deterministic
correlation id. The old `AttemptRecord.sample_id` is a hash of raw output and is
only known after generation, so it is not suitable as required pre-generation
metadata.

## Dataset Split

Use a fixed, model-conditioned HumanEval+ difficulty split for the first
experiments:

```text
humanevalplus_gpt5nano_nonperfect_stratified_v0
```

The source ranking is:

```text
../nl-code/data/humaneval-dspy-sample-performance/humaneval_gpt5nano_worst_all_settings_sample_sets.json
```

That file ranks HumanEval+ tasks by `openrouter/openai/gpt-5-nano` historical
average performance across all settings. The `worst_sample_ids_by_n` sets are
nested (`25` subset of `50`, `50` subset of `100`). The worst 100 contain 99
non-perfect tasks plus one perfect task (`HumanEval/0` at `1.0`). For this first
pass, exclude perfect tasks and use all 99 non-perfect tasks from the worst-100
set.

Create three difficulty buckets:

- `worst_25`: the 25 worst task ids.
- `worst_50_not_25`: worst 50 excluding worst 25.
- `nonperfect_not_50`: worst 100 excluding worst 50 and excluding tasks with
  `average_perf == 1.0`.

Sample without replacement from those buckets with a fixed RNG seed:

- `test`: 10 from `worst_25`, 10 from `worst_50_not_25`, 5 from
  `nonperfect_not_50` for 25 total.
- `dev`: 5 from `worst_25`, 5 from `worst_50_not_25`, 5 from
  `nonperfect_not_50` for 15 total.
- `train`: remaining 10 from `worst_25`, remaining 10 from
  `worst_50_not_25`, and remaining non-perfect tasks from
  `nonperfect_not_50` for up to 59 total.

Split roles:

- `train`: candidate evaluation during COPRO-style proposal rounds.
- `dev`: candidate-surface and hparam comparison, and decoder prompt selection.
- `test`: final held-out report only.

Persist the exact split as JSON and include:

- split name and version;
- RNG seed;
- source ranking path and source metadata;
- task ids by split;
- task ids by difficulty bucket;
- each task's source average performance;
- explicit list of excluded perfect tasks.

This is intentionally a model-conditioned hardness split, not a model-neutral
HumanEval+ difficulty split. Reports should name it as such.

## Repository TODOs

### `dr-code`

- Remove `entry_point` from `AttemptRecord`.
- Add cached/derived `expected_arity` to `HumanEvalPlusTask` or associated
  HumanEval+ metadata.
- Update the HumanEval+ snapshot build/load path to compute and store
  `expected_arity`.
- Add a one-job lower-level eval API for generated code so `dr-bottleneck` does
  not need to call `run_eval_once`.
- Update test execution to parse extracted code, find top-level functions,
  filter by positional arity, run matching candidates, and score by best test
  pass rate.
- Do not allow `*args` initially.
- Update `TestOutcome` or a related result model to include selected function
  name, candidate function names and arities, and whether the expected entry
  point was present.
- Keep `HumanEvalPlusTask.entry_point` only as suite metadata for diagnostics.
- Update exporters, reports, and analysis paths that currently assume
  `AttemptRecord.entry_point`.
- Replace entry-point failure terminology with arity-selection terminology,
  such as "no arity-matching functions."

### `dr-bottleneck`

- Create `dr_bottleneck.workflow_jobs.spec` as the public queue-facing contract
  module for workflow job schemas only.
- Define `JobKind` values for `llm_query_static`,
  `llm_query_from_previous`, and `eval_from_previous`.
- Define `LLMQueryStaticConfig`, `LLMQueryFromPreviousConfig`, and
  `EvalFromPreviousConfig` as the canonical job-kind schemas.
- Define `WorkflowStepSpec`, `WorkflowJobPayload`, `LLMQueryOutput`,
  `CandidateFunction`, and `EvalFromPreviousOutput` in the same spec module.
- Define `SamplingConfigId` and parse model setup strings into typed
  `dr-providers` request settings.
- Replace profile-YAML lookup for the new job kinds with `SamplingConfigId`
  parsing.
- Define the workflow/job envelope model that lets `dr-dspy` specify
  `in_queue -> job_kind -> out_queue`.
- Implement strict missing-output behavior for previous-output job kinds.
- Validate placeholder replacement strictly for `llm_query_from_previous`.
- Implement workflow failure classes `infra_non_retryable`,
  `infra_retryable`, and `model_or_eval_outcome`.
- Remove `entry_point` from candidate/eval plumbing and any `AttemptRecord`
  construction.
- Implement `eval_from_previous` by calling lower-level one-job eval functions,
  attaching structured results to the current `JobEnvelope`, and publishing the
  job to the configured output queue.
- Stop computing/reporting entry-point and signature failure buckets in
  `dr-bottleneck`; those are `dr-code` diagnostics.
- Preserve opaque metadata and pass through structured eval results.
- Preserve the original `model_id` / sampling config id and parsed request
  config in LLM step records.

### `dr-dspy`

- Update experiment docs/configs to defer signature-side-channel decoder
  variants to future work and use one description-only variant for the first
  experiments.
- Create the fixed dataset split
  `humanevalplus_gpt5nano_nonperfect_stratified_v0` from
  `../nl-code/data/humaneval-dspy-sample-performance/humaneval_gpt5nano_worst_all_settings_sample_sets.json`.
- Exclude perfect tasks from the worst-100 source set, including
  `HumanEval/0`.
- Build split buckets `worst_25`, `worst_50_not_25`, and
  `nonperfect_not_50`, then sample without replacement using a fixed RNG seed.
- Persist the split JSON with split name/version, RNG seed, source metadata,
  task ids by split, task ids by difficulty bucket, per-task average
  performance, and excluded perfect task ids.
- Use `train` for COPRO-style candidate evaluation rounds, `dev` for prompt
  selection and hparam/candidate-surface comparisons, and `test` only for the
  final held-out report.
- Stop sending `entry_point` in new eval job configs.
- Build concrete expanded workflow jobs using the `dr-bottleneck` job-kind
  schemas imported directly from `dr_bottleneck.workflow_jobs.spec`.
- Populate and validate required correlation metadata for every expanded
  workflow job.
- Submit jobs to workflow input queues and consume workflow output queues
  directly.
- Define the first-pass metric enum and scorer.
- Require every optimizer run to declare one metric id.
- Use functional-recovery metrics instead of exact-entry-point metrics for the
  first experiments.
- Compute compression metrics in analysis/scoring by reusing `dr-code`'s
  zstd22 helper.
- Aggregate workflow outputs into optimizer-facing metrics and report tables.

## Settled Design Decisions

- `dr-dspy` is the optimizer brain: it proposes prompt candidates, expands
  experiments into concrete workflow jobs, consumes results, and updates
  optimizer state.
- `dr-bottleneck` is the evaluator/runtime: it owns provider calls, queue
  execution, generic workflow job kinds, and pass-through result emission.
- The first integration slice used this queue message contract:

  ```text
  CandidateEvalRequest -> dr-bottleneck evaluation -> CandidateEvalResult
  ```

- The clean-break implementation should replace that high-level candidate eval
  contract with concrete workflow jobs built from `llm_query_static`,
  `llm_query_from_previous`, and `eval_from_previous`.
- Requests and results remain wrapped in `dr_queues.JobEnvelope`s. Durable
  records should keep full prompts, raw outputs, LLM call records, queue events,
  and eval artifacts inspectable.
- Full encoder optimization should reuse the concrete workflow-job contract
  rather than introduce a second integration path.

Initial prompt templates live in this repo under
`configs/prompts/templates/`. They are plain Markdown templates so the first
iteration can focus on exact prompt text before adding YAML sweep config or slot
metadata.

- `baseline_enc.md` - current encoder prompt from the HumanEval workflow.
- `baseline_dec_variantA.md` - current decoder prompt adapted to include
  `{signature}` side-channel context. This variant is deferred for the first
  clean-break experiment pass.
- `baseline_dec_variantB.md` - current decoder prompt, using only
  `{encoded_description}`.
- `manual_dec_v0_variantA.md` / `manual_dec_v0_variantB.md` - minimal manual
  decoder baseline with explicit code-only and parseability constraints.
  Variant A is deferred for the first clean-break experiment pass.
- `manual_dec_v1_variantA.md` / `manual_dec_v1_variantB.md` - slightly stronger
  manual decoder baseline with compact guidance about interface preservation,
  standard-library Python, edge cases, and avoiding tests/placeholders.
  Variant A is deferred for the first clean-break experiment pass.
- `optim_dec_slot_addendum_variantA.md` /
  `optim_dec_slot_addendum_variantB.md` - first optimizable decoder templates.
  These keep the baseline decoder structure and add three bounded instruction
  slots.

Restricted optimization configs live under `configs/optim/`. The first one is
`configs/optim/decoder_slot_addendum_v0.yaml`, which defines the reusable
slot-addendum contract for learned prompt optimizers. The first curated grid is
`configs/optim/decoder_slot_grid_v0.yaml`, which uses the same templates and
slot cap but enumerates three hand-written candidates per slot.

## Decoder Template Options

The main design question is which information is fixed decoder context and
which information must be recovered from the encoder output.

For this decoder-only formatting probe, `{encoded_description}` is the
HumanEval/HumanEval+ docstring text. It is not an encoder-generated
description yet. The variable name is retained because the selected decoder
template will later receive encoder output in the full compression pathway.

### Option A: Signature Side Channel

Option A gives the decoder the expected signature outside the encoded
description:

````text
{text_a}

```python
{signature}
```

{text_b}

```markdown
{encoded_description}
```

{text_c}
````

Option A is deferred for the first clean-break experiment pass. The initial
metric targets functional recovery from the description, and exact public
interface recovery is a future diagnostic/metric rather than a gating condition.

### Option B: Description Only

Option B gives the decoder only the encoded description:

````text
{text_a}

```markdown
{encoded_description}
```

{text_c}
````

This is more faithful to whole-program compression because the encoder must
communicate both interface and behavior. If it works cleanly, it is the stronger
formulation. The risk is that failures become harder to diagnose: the generated
code may be invalid, may omit the expected entry point, may expose an
incompatible signature, or may implement the wrong behavior.

Option B is the initial clean-break experiment variant. The first pass evaluates
whether generated code contains an arity-compatible top-level function that can
pass tests, without requiring exact entry-point-name recovery.

## Baselines And Target Optimizers

Use explicit baselines before claiming any optimizer movement. Then use a small
suite of optimization methods ranging from constrained slot search through
reflective optimization. The intent is not to force every method through the
full experiment; some methods may be dropped if they are brittle, expensive, or
hard to interpret.

### Step 0 Baselines

1. Original baseline prompt.

   This is the current minimal decoder instruction, represented by
   `configs/prompts/templates/baseline_dec_variantB.md`:

   ```text
   Write functional code in Python according to the description.
   ```

   `configs/prompts/templates/baseline_dec_variantA.md` is the same baseline
   adapted for the signature side-channel variant, so it is not a literal
   historical workflow prompt.

2. Minimal manual baseline prompt (`manual_dec_v0`).

   This fixes the obvious formatting constraints while staying short and
   intentionally non-clever:

   ```text
   Write Python code that implements the requested function.

   Output only code, with no markdown fences or explanation.
   The code must parse and run as a complete Python snippet.
   ```

   Option A version:

   ````text
   Write Python code that implements the requested function.

   Output only code, with no markdown fences or explanation.
   The code must parse and run as a complete Python snippet.

   Use this required function signature:

   ```python
   {signature}
   ```

   Description:

   ```markdown
   {encoded_description}
   ```
   ````

   Option B version:

   ````text
   Write Python code that implements the requested function.

   Output only code, with no markdown fences or explanation.
   The code must parse and run as a complete Python snippet.

   Description:

   ```markdown
   {encoded_description}
   ```
   ````

3. Stronger manual baseline prompt (`manual_dec_v1`).

   This keeps the `manual_dec_v0` formatting constraints and adds only compact
   behavioral guardrails: preserve the public interface, use straightforward
   standard-library Python, handle implied edge cases, and avoid tests,
   placeholders, or unrelated top-level behavior. It is intentionally still a
   manual baseline, not an optimizer-generated template.

### Target Optimizer Set

1. Curated prompt-grid search.

   Use `configs/optim/decoder_slot_grid_v0.yaml` to exhaustively evaluate three
   candidates for each of the three addendum slots, for 27 combinations per
   variant. This is the first "optim" stage after baseline evaluation because
   it is deterministic, cheap to inspect, and uses the same slot boundaries as
   learned optimizers.

2. Minimal templated optimization over fixed templates.

   Keep the baseline decoder template fixed and optimize only three short
   addendum slots:

   - `{task_instructions}` - what kind of implementation the decoder should
     produce.
   - `{output_instructions}` - output format controls such as code only, no
     markdown, and complete snippet.
   - `{failure_avoidance}` - common failure prevention such as no placeholders,
     no tests, valid syntax, and arity-compatible function recovery.

   Each slot is capped at 100 characters. The cap keeps optimizer freedom
   focused on compact decoder-control instructions instead of full prompt
   rewriting. It also makes individual slot choices easier to inspect, compare,
   and report.

3. Custom prompt-grid or best-of-N search.

   Generate or enumerate larger prompt variants, evaluate them, and keep the
   best. This is broader than slot optimization and is the cleanest
   optimizer-loop smoke test.

4. `COPRO`.

   COPRO directly proposes and refines instructions/output prefixes, making it
   the best existing DSPy fit for decoder-format optimization.

5. `MIPROv2`, configured zero-shot first.

   Start with `max_bootstrapped_demos=0` and `max_labeled_demos=0` so the first
   run tests instruction optimization alone. Add demos later only if useful.

6. `GEPA`.

   Use GEPA as the most complex reflective method once parse/test failures are
   well-instrumented enough to become useful feedback.

The likely outcome is to keep the most illustrative two or three methods for
later correctness and encoder-compression experiments.

### First Implementation Pass: COPRO-Style Optimization

For the first learned-optimization implementation pass, use COPRO-style search
for all optimization flows rather than trying to integrate every candidate
method immediately. The near-term goal is to get clean baselines, grid results,
and a reusable candidate-generation/evaluation loop that can later support more
complex optimizers.

First produce these non-learned baselines:

- True baseline prompt evaluation.
- Manually selected prompt evaluation.
- Curated grid-sweep evaluation over the bounded decoder slots.

Then use COPRO-style optimization. The first actual learned run should be
intentionally narrow:

```text
bounded slot generation x decoder formatting
```

After that path produces clean artifacts and interpretable scores, reuse the
same loop for the broader matrix:

```text
(small field generation, full prompt generation)
  x (decoder formatting, decoder pass rate, encoder-decoder compression/pass rate)
```

Settled choices for this first pass:

- Keep the two proposal phases from COPRO:
  - initial generation from the current candidate;
  - refinement generation from prior candidate/score history.
- Treat each proposal completion as one candidate bundle.
- Update the proposal TaskSpecs so their outputs match the candidate surface for
  the run. For small field generation, the outputs are the actual bounded slots
  such as `task_instructions`, `output_instructions`, and
  `failure_avoidance`. For full prompt generation, the output is the full prompt
  text or prompt section being optimized.
- Use normal TaskSpec-to-prompt rendering for candidate generation. The proposal
  prompts should not include dataset examples in this first pass; they should
  include candidate score history after the first round.
- For the first-round proposal phase, adapt COPRO's basic instruction proposal
  shape:

  ```python
  class BasicGenerateInstructionTaskSpec(TaskSpec):
      name: str = "framework.copro.basic_generate_instruction"
      instructions: str = (
          "You are an instruction optimizer for large language models. "
          "I will give you the current candidate text. "
          "Your task is to propose an improved candidate that will lead a good "
          "language model to perform the task well. Don't be afraid to be creative."
      )
      inputs = (
          input_field("basic_instruction", str, desc="The initial candidate before optimization"),
      )
      outputs = (
          output_field("proposed_candidate", str, desc="The improved candidate text"),
      )
  ```

  For bounded slot generation, use one output field per optimized slot rather
  than a single `proposed_candidate` field. For full-prompt generation, the
  output can be the full prompt text or prompt section being optimized.
- For later refinement rounds, adapt COPRO's attempted-instructions proposal
  shape:

  ```python
  class GenerateInstructionGivenAttemptsTaskSpec(TaskSpec):
      name: str = "framework.copro.generate_instruction_given_attempts"
      instructions: str = (
          "You are an instruction optimizer for large language models. "
          "I will give candidate texts I've tried, along with their validation "
          "scores and short diagnostics. The candidates are arranged in "
          "increasing order based on their scores, where higher scores indicate "
          "better quality.\n\n"
          "Your task is to propose a new candidate that will lead a good "
          "language model to perform the task even better. Don't be afraid to "
          "be creative."
      )
      inputs = (
          input_field(
              "attempted_candidates",
              str,
              desc="Previously attempted candidates, scores, and diagnostics.",
          ),
      )
      outputs = (
          output_field("proposed_candidate", str, desc="The improved candidate text"),
      )
  ```

  The attempted-candidates payload should include candidate text, metric id,
  aggregate score, parse rate, functional-recovery pass rate, and a short
  failure summary. Keep it bounded by including the incumbent plus top/bottom
  candidates rather than every artifact from the workflow.
- Own evaluation separately from candidate generation. Candidate generation
  produces candidate text; evaluation renders that candidate into the actual
  decoder or encoder/decoder prompt, submits the workflow jobs, collects raw LLM
  outputs and eval artifacts, and computes the metric.
- Treat `breadth`, `depth`, and `n_seeds_per_eval` as the main hparams for the
  first pass:
  - `breadth`: number of candidate bundles generated per round, including the
    incumbent/baseline candidate.
  - `depth`: number of proposal/evaluation rounds.
  - `n_seeds_per_eval`: number of stochastic workflow repeats used for each
    `(round, candidate, data_sample)` tuple when estimating the metric.
  - `data_sample_count`: number of distinct dataset samples evaluated per
    candidate.

Implementation guidance inferred from COPRO's current shape:

- The reusable optimizer loop can stay small:

  ```text
  generate candidates -> evaluate candidates -> keep best -> generate from scored attempts
  ```

- The optimizer should only receive the resulting score and candidate history
  needed for the next proposal round, not the workflow internals.
- Candidate cap enforcement, candidate validation, parsing, and artifact logging
  should live in the experiment workflow rather than inside a generic COPRO
  loop.
- Proposal-model choice, proposal temperature, evaluator model/sampling config,
  data-sample count, candidate cap policy, and metric id should be recorded for
  every run. These are not the main hparams for the first pass, but they affect
  interpretation.
- Total workflow cost scales with the product of rounds, candidate count, data
  samples, and seeds, so every report should include the realized number of
  submitted workflow jobs and successful eval results.
- Prompts selected by COPRO-style optimization should be rerun on a held-out
  sample set before being frozen.

The first planning emphasis is on understanding the interaction of breadth,
depth, and `n_seeds_per_eval`; the other recorded controls are context for
interpreting those runs.

## Method Selection Criteria

Use two passes when selecting methods.

First pass: quality and conceptual fit. Ignore implementation practicality,
custom generation routing, one-example smoke-test support, trace quality, and
cost.

First-pass criteria:

- Optimizes prompts, instructions, templates, or decoder behavior in a way that
  could plausibly improve parseability or correctness.
- Represents a distinct optimization strategy, rather than a minor variant of
  another candidate.
- Can use scalar feedback such as AST parse rate or test pass rate.
- Could teach us something useful about the optimization problem if it worked.
- Is present in this DSPy fork or close enough to the existing teleprompt
  surface to evaluate as a candidate.

Second pass: practicality. Apply this only after the broad candidate list is
formed.

Second-pass criteria:

- Can route generations through `../dr-providers/` or be adapted cleanly.
- Can run, or be approximated, in the one-example exploitability smoke test.
- Produces inspectable artifacts: candidate prompts, rendered prompts, raw
  outputs, scores, and failure buckets.
- Has acceptable cost for `openrouter/openai/gpt-oss-20b/low/v1`.
- Has enough implementation support to avoid turning this phase into a large
  optimizer-implementation project.

## First-Pass Candidate Methods

These are broad candidates from the current `dspy.teleprompt` surface, chosen
for quality and conceptual fit before practicality filtering.

### Strong First-Pass Candidates

1. `COPRO`

   COPRO directly proposes and refines predictor instructions and output
   prefixes, evaluates candidates, and iterates over breadth/depth. This is a
   strong fit for decoder-format optimization because the first target is
   exactly instruction/template behavior: produce parseable Python.

2. `MIPROv2`

   MIPROv2 proposes instruction candidates using program-aware, data-aware,
   tip-aware, and few-shot-aware context, then searches over prompt parameters.
   With zero demos, it is a natural candidate for instruction-only optimization;
   with demos enabled, it can test whether examples help decoder format and
   correctness.

3. `GEPA`

   GEPA uses reflective mutation with metric feedback and can optimize
   instructions from richer failure information. It is the most conceptually
   powerful prompt optimizer in the set, especially once parseability failures
   and test failures can be converted into useful feedback.

4. `SIMBA`

   SIMBA samples program trajectories and mutates programs by appending demos or
   rules. It is less directly a pure template optimizer, but its rule-append
   strategy could be useful for turning observed failures into decoder guidance.

5. `InferRules`

   InferRules bootstraps demos, induces natural-language rules from examples,
   appends those rules to instructions, and evaluates candidates. It is a good
   conceptual fit for learning decoder formatting/correctness heuristics from
   HumanEval examples.

6. `AvatarOptimizer`

   AvatarOptimizer compares positive and negative examples, asks for feedback,
   and updates actor instructions. It appears agent/tool-oriented, but the
   positive/negative feedback loop is conceptually relevant to parseability and
   pass/fail optimization.

### Useful Baselines Or Comparators

7. `LabeledFewShot`

   LabeledFewShot does not optimize instructions; it samples labeled examples
   as demos. It is useful as a simple comparator for whether examples alone
   improve parseability or pass rate.

8. `BootstrapFewShot`

   BootstrapFewShot generates and keeps successful demos from a teacher program.
   It is primarily demo optimization rather than decoder-template optimization,
   but it can test whether successful generated-code examples help.

9. `BootstrapFewShotWithRandomSearch`

   This searches over random candidate demo sets and evaluates them. It is not
   directly prompt-template optimization, but it is a useful simple search
   baseline and already fits the "candidate -> evaluate -> keep best" shape.

10. `BootstrapFewShotWithOptuna`

    This uses Optuna to select demo choices after bootstrapping. It is another
    demo-selection baseline, less central than random search but useful if demo
    composition turns out to matter.

11. `BetterTogether`

    BetterTogether chains multiple optimizers, typically prompt and weight
    optimization. It is not a first single-method target, but it may be useful
    later as a composition strategy once a smaller set of prompt optimizers has
    been validated.

### Likely Out Of Scope For This Stage

12. `BootstrapFinetune` and `GRPO`

    These target weight or reinforcement-style optimization rather than prompt
    formatting. They are important capabilities, but they do not match the
    current goal of freezing a decoder prompt/template.

13. `KNNFewShot` and `Ensemble`

    These change demo retrieval or combine programs rather than optimizing the
    decoder prompt itself. They may become useful later, but they are not central
    to this phase.

## Proposed Experiment Sequence

Before numbered evaluation runs, freeze the dataset split identifiers. The
decoder-only dataset view should be expanded in `dr-dspy`. For the first
clean-break pass, use the description-only decoder input and defer
signature-side-channel variants.

0. Run baseline evaluation batches.

   Evaluate `baseline_dec`, `manual_dec_v0`, and `manual_dec_v1` for the
   description-only variant. These establish the parseability and functional
   recovery floor that later "optim" results must beat.

1. Run curated grid-search "optim" batches.

   Use `configs/optim/decoder_slot_grid_v0.yaml` with the description-only
   `optim_dec_slot_addendum` template. Evaluate all 27 slot combinations, then
   analyze results by full combination and by individual slot candidate.

2. Run minimal templated optimization.

   Use `configs/optim/decoder_slot_addendum_v0.yaml` with the COPRO-style
   candidate-generation loop. In the first learned run, produce only the three
   bounded slot values, not the full decoder prompt.

3. Run a one-example exploitability smoke test for learned optimizers.

   Use `train = val = test = one example` with minimal guardrails. Run this on
   the COPRO-style loop first. The goal is to validate wiring and expose the
   expected catastrophic failure modes early, especially candidate prompts that
   leak the target code or otherwise exploit the metric.

   This is not an optimization result. Treat it as an exploitability and
   observability test.

4. Optimize for AST parseability with the COPRO-style loop.

   Use the decoder-only task and `parse_binary` / parse-rate reporting first.
   This should expose whether the optimizer can improve simple
   formatting/control behavior and whether any model needs an even easier
   initial target.

5. Measure pass rate after parseability optimization.

   Do not assume parseability optimization improves correctness. Report pass
   rate on the parse-optimized prompt to understand the tradeoff.

6. Cull flows before moving to harder objectives.

   Drop candidate surfaces or evaluation flows that are brittle, too expensive,
   or hard to interpret. The goal is to finish this stage with a validated
   setup and a small set of COPRO-style flows worth applying to correctness and
   later encoder-compression optimization.

7. Optimize for test pass rate.

   After the formatting probe produces enough evidence to choose the next
   objective, switch the metric to HumanEval/HumanEval+ functional recovery
   pass rate. Compare with both the baseline and the parse-optimized prompt.

8. Select and freeze the decoder prompt/template.

   Prefer the prompt that gives stable parseability and the best pass rate while
   remaining compatible with later encoder-generated descriptions.

## Remaining Implementation Steps

Near-term evaluator work:

- Run a live one-example queue smoke test with RabbitMQ, MongoDB, OpenRouter,
  one workflow worker, and one result consumer.
- Add thin CLIs or notebooks for submitting baseline/manual/grid workflow jobs.
- Persist optimizer-run bookkeeping in `dr-dspy`: submitted candidate ids,
  result status, metric target, score, feedback, and linked run ids.
- Add analysis helpers that read workflow results and produce the planned
  baseline/grid reports by model, sampling config, metric, and failure bucket.

Optimizer integration work:

- Implement the curated 27-combination grid submission/collection loop.
- Implement the COPRO-style learned loop with separate proposal TaskSpecs for
  initial generation and refinement from candidate/score history.
- Add candidate-generation TaskSpecs for bounded slot bundles first. Add
  full-prompt candidates only after the bounded-slot loop is stable.
- Use workflow submission/collection as the metric boundary: render candidates,
  submit workflow jobs, collect raw outputs/eval artifacts, and aggregate scores
  before returning a metric to the optimizer loop.
- Track `breadth`, `depth`, `n_seeds_per_eval`, data-sample count, metric id,
  proposal model, proposal temperature, evaluator model/sampling config, cap
  policy, candidate ids, submitted workflow job ids, and successful eval counts.
- Rerun selected candidates on a held-out sample set before freezing a decoder
  prompt/template.
- Start with scalar parse and functional-recovery metrics. Add GEPA-style
  feedback strings from eval diagnostics only after the COPRO-style loop and
  workflow artifacts are stable.
- Decide after live COPRO-style probes which additional learned methods are
  worth keeping beyond the grid baseline.

Later encoder-path work:

- Extend the workflow-job path to support two-step encoder/decode workflows
  from a static encoder LLM job, a previous-output decoder LLM job, and an
  eval-from-previous job.
- Make compressed encoder-output size the secondary metric for
  bounded correctness/compression metrics.
- Reuse the concrete workflow-job contract for encoder candidates, with
  additional provenance for encoder prompt, encoder output, and compression
  metrics in workflow outputs.

## Metrics To Report

Report metrics by model, sampling config, metric id, and template.

- AST parse rate.
- Functional recovery test pass rate from the selected arity-compatible
  function.
- Selected function name.
- Candidate function names and arities.
- Expected entry point present as a diagnostic only.
- Pass rate after optimizing only for parseability.
- Pass rate after optimizing for passing.
- Compression ratio and bounded compression-penalty score once encoder output
  exists.
- For smoke tests: whether source/code leakage or other objective exploitation
  occurred.
- Failure buckets:
  - invalid Python
  - no top-level functions
  - no arity-matching functions
  - runtime error
  - failed assertions

## Concerns

- Option A gives the signature for free. This should be stated clearly in any
  future report because it changes the compression problem. It is deferred for
  the first clean-break pass.
- The frozen decoder prompt should not overfit to docstring-style inputs. It
  should still work with dense, encoder-generated descriptions.
- Parseability is necessary but not sufficient. A prompt can produce valid
  Python while making behavior worse.
- The metric used for each stage should match the stage objective:
  parseability for formatting control, functional recovery for decoder
  correctness, and bounded correctness-plus-compression only after the encoder
  is introduced.
- Because the full plan may bypass DSPy's normal formatting, optimizer traces
  must still record the exact rendered prompts and raw LLM outputs.
- The one-example smoke test is expected to be vulnerable to exploitative
  solutions. Its value is in proving that the logs and metrics make those
  failures obvious.
- Avoid spending too much time making every method work perfectly on
  parseability. The useful outcome is identifying a smaller set of predictable,
  interpretable methods.

## Open Questions

- Which task should be used for the one-example exploitability smoke test?

Deferred planning decisions:

- Do not set a hard success threshold for "decoder formatting solved" before
  the probe runs. Use the formatting results to choose future thresholds.
- Do not set detailed optimizer candidate budgets in this plan. Use the probe
  results to choose budgets for later, more conclusive experiments.
- Leave implementation details for a later planning pass.
