# GEPA

## Synthesis For Decoder-Format Optimization

GEPA is the most powerful and most complex learned optimizer in the current
decoder-format plan. It is attractive because our decoder failures can produce
natural-language feedback that is much richer than a scalar parse/test score:
invalid Python, markdown leakage, missing entry point, incompatible signature,
runtime error, failed assertion, or suspected code leakage. GEPA is designed to
use exactly that kind of feedback.

The main risk is that dr-dspy's GEPA wrapper currently evolves whole predictor
instructions, while our experiment wants bounded slot addenda in rendered
Markdown templates. GEPA can also be too powerful for the first formatting
probe: if the reflective feedback includes target code, hidden tests, or
over-specific examples, it may optimize the probe rather than the decoder
format.

Open questions:

- Can we provide enough textual feedback for GEPA without leaking target
  implementations or creating an exploitative one-example result?
- Should GEPA optimize three bounded slots through a custom
  `instruction_proposer`, or should it first run against a single combined
  instruction surface?
- Does GEPA's Pareto frontier help Option B by preserving candidates that solve
  different interface-recovery failures, or does it mostly add cost/noise during
  formatting-only optimization?

## 1. Method Overview

GEPA, "Genetic-Pareto," was introduced by Agrawal et al. in 2025 and published
as an ICLR 2026 oral paper. The OpenReview page describes it as a prompt
optimizer that uses natural-language reflection over trajectories to optimize
compound AI systems, and reports that it outperforms GRPO and MIPROv2 while
using far fewer rollouts.

GEPA is currently considered one of the stronger prompt optimizers in the DSPy
ecosystem, especially when a task can expose useful failure traces or textual
feedback. The DSPy GEPA documentation frames it as a reflective optimizer for
evolving textual components of complex systems. The GEPA repository also
positions DSPy as the recommended integration for AI pipelines.

Because it is newer and more capable, GEPA is not the first thing to trust as a
clean formatting baseline. It is better treated as the rich-feedback method to
try after the grid, COPRO, and zero-shot MIPROv2 have clarified the basic
failure modes.

## 2. How It Works

GEPA maintains a pool of textual candidates and repeatedly improves them:

1. Start from a seed candidate, usually the current prompt/instruction text.
2. Evaluate candidates on train and validation examples.
3. Capture traces, outputs, failures, and metric feedback.
4. Build a reflective dataset from those traces.
5. Ask a reflection LM to diagnose failures and propose updated text.
6. Select candidates through a Pareto strategy rather than only global best
   score, so candidates that excel on different examples can survive.
7. Optionally merge candidates that cover complementary examples.

Strengths:

- Uses rich textual feedback, not just scalar scores.
- Well-suited to structured failure buckets and execution traces.
- Can be sample efficient when each rollout is expensive.
- Pareto selection may preserve diverse fixes for heterogeneous failures.
- Produces human-readable optimization traces and candidate text.

Weaknesses:

- Requires good feedback design. Poor feedback can lead to noisy or
  exploitative prompt changes.
- More moving parts than COPRO or MIPROv2: reflection LM, candidate selection,
  merge behavior, reflective minibatches, tracking options, and adapter logic.
- More likely to exploit target leakage in degenerate smoke tests.
- Local dr-dspy wrapper only updates predictor instructions by default.
- Strong reflection models may cost more than the target decoder model.

## 3. Interaction With Our Experiment

GEPA should become useful once the decoder-format evaluator can produce clear
feedback strings. For AST parseability, feedback can include the syntax error,
whether markdown fences appeared, and whether the raw output was empty or
truncated. For Option B, feedback can say whether the expected entry point was
missing, whether the callable signature was incompatible, and whether the model
defined the wrong function.

Expected behavior:

- GEPA may outperform scalar-only methods when failures are diverse and
  diagnosable.
- It may produce longer, more rule-like instructions unless constrained by a
  custom proposer or slot cap.
- It may learn useful distinctions between Option A and Option B because their
  failures should produce different feedback.
- It may overfit hard to the probe data if train and validation are too small or
  if feedback exposes target-specific facts.

Questions to watch:

- Does rich feedback improve formatting faster than scalar parseability alone?
- Which feedback fields are useful versus leakage-prone?
- Does `add_format_failure_as_feedback=True` help with invalid outputs in our
  adapter path?
- Does Pareto selection preserve candidates that solve different failure
  buckets, or does the best candidate dominate quickly?
- Are GEPA candidates compatible with dense encoder-generated descriptions, or
  do they over-specialize to HumanEval docstring style?

## 4. dr-dspy Implementation

Local entry points:

- `dspy/teleprompt/gepa/gepa.py`
- `dspy/integrations/optimizers/gepa/adapter.py`
- `dspy/integrations/optimizers/gepa/task_specs.py`
- `dspy/teleprompt/compile_params.py`
- Tests under `tests/integrations/optimizers/gepa/`

The class is registered as a teleprompter for `GEPACompileParams`. Constructor
hyperparameters currently include:

- `metric`: must accept five arguments:
  `(gold, pred, trace, pred_name, pred_trace)`. It can return a float or
  `ScoreWithFeedback`.
- Exactly one of `auto`, `max_full_evals`, or `max_metric_calls`: controls the
  run budget. `auto` maps to light/medium/heavy candidate counts.
- `reflection_minibatch_size`: number of examples used for reflection batches.
  Smaller values reduce reflection context size but may increase noise.
- `candidate_selection_strategy`: `pareto` or `current_best`.
- `reflection_lm`: LM used for reflective mutation. Required unless a custom
  async `instruction_proposer` is supplied.
- `skip_perfect_score`: avoids reflecting on already-perfect examples.
- `add_format_failure_as_feedback`: includes parse/format failures in the
  reflective dataset.
- `instruction_proposer`: optional async custom proposer. This is the likely
  extension point for bounded slot proposals.
- `component_selector`: chooses which predictor/component to update.
- `use_merge` and `max_merge_invocations`: enable and bound candidate merging.
- `max_concurrency`, `failure_score`, `perfect_score`, `log_dir`,
  `track_stats`, `track_best_outputs`, `seed`, and external tracking options.
- `gepa_kwargs`: passed through to the external `gepa.optimize`, except
  `reflection_prompt_template` is explicitly rejected because the DSPy adapter
  owns proposal formatting.

Compile parameters currently include:

- `trainset`
- `teacher`, inherited from `BootstrapFewShotCompileParams`, but GEPA asserts
  that teacher is not supported.
- `valset`; if omitted, trainset is reused as validation.

Important implementation behavior:

- The wrapper imports and calls external `gepa.optimize` in a thread.
- The seed candidate is `{predictor_name: predictor.task_spec.instructions}`.
- `DspyAdapter.build_program` deep-copies the student and replaces
  `TaskSpec.instructions` for named predictors.
- `DspyAdapter.evaluate` can collect traces and capture parse failures.
- `DspyAdapter.make_reflective_dataset` converts inputs, generated outputs, and
  feedback into the text examples used by the reflection proposer.
- With `track_stats=True`, the optimized program receives `detailed_results`.

## 5. Likely Changes Needed For The Decoder Flow

GEPA is the method most likely to need a custom bridge because its value depends
on high-quality feedback and because unconstrained instruction mutation is
broader than our slot-addendum contract.

Likely changes or wrappers:

- Implement a decoder-format `GEPAFeedbackMetric` that returns
  `ScoreWithFeedback` with non-leaky failure explanations.
- Decide whether to enable `add_format_failure_as_feedback` for parse failures.
  It is probably useful for formatting, but the resulting structure guidance
  must match our raw-code decoder output.
- Add a custom async `instruction_proposer` if we want GEPA to update
  `task_instructions`, `output_instructions`, and `failure_avoidance` as
  bounded slots rather than whole predictor instructions.
- Validate and record raw versus truncated slot values when GEPA proposes text
  over 100 characters.
- Make the reflective dataset include failure buckets and raw model output
  previews, but not canonical solutions, hidden tests, or target code.
- Ensure the GEPA adapter path can evaluate candidates through the planned
  dr-providers decoder flow instead of normal DSPy adapter rendering.
- Persist GEPA-specific artifacts: parents, candidate texts, Pareto scores,
  per-example best candidates, reflection feedback, rendered prompts, raw
  decoder outputs, and failure buckets.
- Be careful with one-example smoke tests. GEPA is specifically designed to
  exploit rich feedback, so exploitability should be treated as an expected
  observability finding, not as a result.

## Sources

- GEPA OpenReview page: https://openreview.net/forum?id=RQm2KQTM5r
- GEPA arXiv page: https://arxiv.org/abs/2507.19457
- DSPy GEPA docs: https://dspy.ai/api/optimizers/GEPA/overview/
- GEPA repository: https://github.com/gepa-ai/gepa
- Local implementation: `dspy/teleprompt/gepa/gepa.py`
- Local adapter: `dspy/integrations/optimizers/gepa/adapter.py`
- Local compile params: `dspy/teleprompt/compile_params.py`
