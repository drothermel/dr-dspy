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
ground-truth signature + ground-truth docstring/description + decoder template
  -> decoder LLM
  -> generated Python code
  -> formatting/correctness evaluation
```

The immediate objective is not compression. The objective is to learn how to run
prompt optimization reliably and to select decoder formatting that produces
valid, testable Python before the encoder is introduced.

## Starting Assumptions

- DSPy's default prompt formatting may be bypassed for this work.
- LLM generations are expected to route through `../dr-providers/` rather than
  the normal DSPy adapter formatting path.
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
  encode/decode/evaluate workflow.
- `../dr-bottleneck/src/dr_bottleneck/experiments/humaneval.py` - HumanEval+
  loading, source construction, job expansion, and the current AST/compression
  process handler.
- `../dr-bottleneck/scripts/preview_humaneval_prompts.py` - prompt rendering
  helper for inspecting workflow prompts without LLM calls.
- `../dr-bottleneck/scripts/run_humaneval_demo.py` - current HumanEval sweep
  entry point.

Initial prompt templates live in this repo under
`configs/prompts/templates/`. They are plain Markdown templates so the first
iteration can focus on exact prompt text before adding YAML sweep config or slot
metadata.

- `baseline_enc.md` - current encoder prompt from the HumanEval workflow.
- `baseline_dec_variantA.md` - current decoder prompt adapted to include
  `{signature}` side-channel context.
- `baseline_dec_variantB.md` - current decoder prompt, using only
  `{encoded_description}`.
- `manual_dec_v0_variantA.md` / `manual_dec_v0_variantB.md` - minimal manual
  decoder baseline with explicit code-only and parseability constraints.
- `manual_dec_v1_variantA.md` / `manual_dec_v1_variantB.md` - slightly stronger
  manual decoder baseline with compact guidance about interface preservation,
  standard-library Python, edge cases, and avoiding tests/placeholders.
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

During the decoder-formatting pass, Option A has equal priority with Option B.
It tests formatting behavior when the public interface is known and the decoder
is responsible for producing a valid implementation under that interface.

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

During the decoder-formatting pass, Option B has equal priority with Option A.
This pass is a probe: the goal is to observe the failure modes and collect
evidence for later choices, not to pre-decide the final formulation.

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
     no tests, valid syntax, and expected entry point recovery.

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

Before numbered evaluation runs, build the decoder-only dataset view and freeze
the split identifiers. For each HumanEval/HumanEval+ task, construct examples
containing at least the ground-truth signature, the docstring text used as
`{encoded_description}`, the expected entry point, and the test harness metadata
needed for evaluation.

0. Run baseline evaluation batches.

   Evaluate `baseline_dec`, `manual_dec_v0`, and `manual_dec_v1`, each crossed
   with `variantA` and `variantB`. These establish the parseability and
   pass-rate floor that later "optim" results must beat.

1. Run curated grid-search "optim" batches.

   Use `configs/optim/decoder_slot_grid_v0.yaml` with the two
   `optim_dec_slot_addendum` templates. Evaluate all 27 slot combinations per
   variant, then analyze results by full combination and by individual slot
   candidate.

2. Run minimal templated optimization.

   Use `configs/optim/decoder_slot_addendum_v0.yaml` with learned prompt
   optimizers such as COPRO, zero-shot-first MIPROv2, and GEPA. The optimizers
   should produce only the three bounded slot values, not rewrite the full
   decoder prompt.

3. Run a one-example exploitability smoke test for learned optimizers.

   Use `train = val = test = one example` with minimal guardrails. Run this on
   every method that can tolerate the degenerate setup. The goal is to validate
   wiring and expose the expected catastrophic failure modes early, especially
   candidate prompts that leak the target code or otherwise exploit the metric.

   This is not an optimization result. Treat it as an exploitability and
   observability test.

4. Optimize for AST parseability method by method.

   Use the decoder-only task and a parseability metric first. This should expose
   whether the optimizer can improve simple formatting/control behavior and
   whether any model needs an even easier initial target.

5. Measure pass rate after parseability optimization.

   Do not assume parseability optimization improves correctness. Report pass
   rate on the parse-optimized prompt to understand the tradeoff.

6. Cull methods before moving to harder objectives.

   Drop methods that are brittle, too expensive, or hard to interpret. The goal
   is to finish this stage with a validated setup and a small set of methods
   worth applying to correctness and later encoder-compression optimization.

7. Optimize for test pass rate.

   After the formatting probe produces enough evidence to choose the next
   objective, switch the metric to HumanEval/HumanEval+ test pass rate. Compare
   with both the baseline and the parse-optimized prompt.

8. Select and freeze the decoder prompt/template.

   Prefer the prompt that gives stable parseability and the best pass rate while
   remaining compatible with later encoder-generated descriptions.

## Metrics To Report

Report metrics by model/lane and template option.

- AST parse rate.
- Expected entry point exists.
- Callable signature is compatible with the harness.
- Test pass rate.
- Pass rate after optimizing only for parseability.
- Pass rate after optimizing for passing.
- For smoke tests: whether source/code leakage or other objective exploitation
  occurred.
- Failure buckets, especially for Option B:
  - invalid Python
  - missing expected entry point
  - incompatible callable signature
  - runtime error
  - failed assertions

## Concerns

- Option A gives the signature for free. This should be stated clearly in any
  report because it changes the compression problem.
- Option A and Option B should be treated as equal-priority variants during the
  decoder-formatting pass. If later results show that Option B is too noisy,
  that should be a conclusion from the probe rather than an initial assumption.
- The frozen decoder prompt should not overfit to docstring-style inputs. It
  should still work with dense, encoder-generated descriptions.
- Parseability is necessary but not sufficient. A prompt can produce valid
  Python while making behavior worse.
- The metric used for each stage should match the stage objective:
  parseability for formatting control, test pass rate for decoder correctness,
  and correctness-plus-compression only after the encoder is introduced.
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
