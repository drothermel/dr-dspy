# MIPROv2

## Synthesis For Decoder-Format Optimization

MIPROv2 is the strongest "standard DSPy" optimizer we plan to try before GEPA.
It is built to optimize instructions and few-shot demonstrations jointly, but
our decoder-format pass should start in zero-shot mode:
`max_bootstrapped_demos=0` and `max_labeled_demos=0`. That lets us test
instruction optimization without confounding the result with example selection.

MIPROv2 is likely to be more systematic than COPRO because it proposes
task-grounded instruction candidates and searches combinations with Bayesian
optimization. The main mismatch is the same slot-boundary issue: local MIPROv2
selects among full instruction candidates for DSPy predictors, while the plan
wants three bounded slots inside rendered Markdown decoder templates.

Open questions:

- Should the first zero-shot MIPROv2 run optimize one combined addendum or three
  separately represented slots?
- Which proposer context should stay enabled when the "program" is really a
  decoder-template wrapper rather than a normal multi-stage DSPy program?
- Does MIPROv2's Bayesian search add signal over the deterministic grid when
  the search space is intentionally tiny?

## 1. Method Overview

MIPRO comes from "Optimizing Instructions and Demonstrations for Multi-Stage
Language Model Programs" by Opsahl-Ong et al., published at EMNLP 2024. The
paper studies prompt optimization for language-model programs without access to
module-level labels or gradients. It factorizes each module's prompt into
free-form instructions and few-shot demonstrations, then proposes task-grounded
instructions and searches over prompt parameters.

The DSPy documentation describes MIPROv2 as an optimizer that can jointly
optimize instructions and few-shot examples, or run as zero-shot instruction
optimization. The paper reports that MIPRO outperformed baseline optimizers on
five of seven multi-stage programs with a best-in-class open-source model, by
up to 13 percentage points accuracy.

As of this plan, MIPROv2 is a mature, mainstream DSPy optimizer. It is not as
new or feedback-rich as GEPA, but it is more capable than COPRO and is a natural
middle point between simple instruction search and reflective evolution.

## 2. How It Works

MIPROv2 has three main phases:

1. Bootstrap candidate few-shot examples from the trainset. Successful teacher
   or task-model trajectories become candidate demos.
2. Propose instruction candidates using grounded context: dataset summaries,
   program-aware information, task demos, and prompting tips.
3. Use Bayesian optimization, via Optuna in dr-dspy, to select combinations of
   instructions and demos that score well on a validation set. It can use
   minibatches and periodic full evaluations to control evaluation cost.

In zero-shot mode, the demo candidates are removed before search, so the final
search only selects among instruction candidates.

Strengths:

- Stronger proposal context than COPRO.
- Can run instruction-only or instruction-plus-demo optimization.
- Bayesian search is more systematic than greedy candidate refinement.
- Supports train/validation splits, minibatching, seeds, and stats.
- Good fit for scalar metrics such as parseability or pass rate.

Weaknesses:

- More expensive and less transparent than COPRO.
- The auto settings can silently choose candidate counts, validation sizes, and
  minibatch behavior that may not match a small probe.
- Zero-shot mode still runs through bootstrapping code before dropping demos.
- It optimizes full instruction candidates, not bounded template slots.
- It depends on Optuna for search.
- If parseability is the only metric, it can still select prompts that make
  valid but incorrect code.

## 3. Interaction With Our Experiment

MIPROv2 should be run after the curated grid and COPRO. The grid tells us how
much can be gained by tiny hand-written slot changes; COPRO tells us whether
simple learned instruction refinement moves the metric. MIPROv2 then tests
whether grounded proposal and Bayesian selection improve on those simpler
baselines.

Expected behavior:

- In zero-shot mode, MIPROv2 should search over instruction candidates without
  adding examples to the decoder prompt.
- It may be better than COPRO at incorporating the Option A versus Option B
  task context if the proposer sees representative examples and program/task
  summaries.
- It may overfit to docstring-style wording if the trainset and proposer
  summaries emphasize HumanEval prompt conventions too strongly.
- If auto settings are used, it may run more evaluation than we want for an
  exploratory formatting probe. We should record actual candidate counts,
  trials, and metric calls even if we do not set a hard budget in the plan.

Questions to watch:

- Does zero-shot MIPROv2 produce better formatting prompts than COPRO, or does
  the extra machinery mainly help when demos are enabled?
- Does the best MIPROv2 prompt improve Option B's interface recovery, or only
  reduce markdown/syntax failures?
- Do program-aware and data-aware proposers help when the "program" is a thin
  decoder wrapper?
- Are selected instructions stable across seeds?
- Does MIPROv2's best prompt remain compact enough to translate back into our
  slot-addendum template?

## 4. dr-dspy Implementation

Local entry points:

- `dspy/teleprompt/mipro/optimizer.py`
- `dspy/teleprompt/mipro/bootstrap.py`
- `dspy/teleprompt/mipro/propose.py`
- `dspy/teleprompt/mipro/search.py`
- `dspy/teleprompt/mipro/settings.py`
- `dspy/teleprompt/compile_params.py`
- Basic protocol test in `tests/teleprompt/test_mipro_types.py`

The class is registered as a teleprompter for `MIPROv2CompileParams`.
Constructor hyperparameters currently include:

- `metric`: scalar optimizer metric.
- `prompt_model`: model used to propose instructions.
- `task_model`: model used for bootstrapping/evaluating task trajectories.
- `teacher_run`: stored but not used directly in the visible compile flow.
- `max_bootstrapped_demos`: default 4; set to 0 for our first zero-shot run.
- `max_labeled_demos`: default 4; set to 0 for our first zero-shot run.
- `auto`: `light`, `medium`, `heavy`, or `None`. When not `None`, it controls
  candidate counts, validation size, minibatching, and trial count.
- `num_candidates`: required when `auto=None`; forbidden when `auto` is set.
- `max_concurrency`, `max_errors`: evaluation controls.
- `seed`: random seed for minibatches and proposal/search behavior.
- `init_temperature`: instruction proposal temperature.
- `verbose`, `track_stats`, `log_dir`, `metric_threshold`: observability and
  filtering controls.

Compile parameters currently include:

- `trainset`
- `teacher`
- `valset`
- `num_trials`
- `max_bootstrapped_demos`
- `max_labeled_demos`
- `seed`
- `minibatch`
- `minibatch_size`
- `minibatch_full_eval_steps`
- `program_aware_proposer`
- `data_aware_proposer`
- `view_data_batch_size`
- `tip_aware_proposer`
- `fewshot_aware_proposer`
- `provide_traceback`

Important implementation behavior:

- Compile resolves task and prompt models from the run context if not provided.
- If `auto=None`, both `num_candidates` and `num_trials` must be supplied.
- If `auto` is set, `num_candidates` and `num_trials` are rejected because auto
  settings override them.
- If no `valset` is provided, the settings code splits the trainset and requires
  at least two examples.
- Auto modes use candidate counts of 6, 12, or 18 and validation sizes of 100,
  300, or 1000 for light, medium, and heavy.
- The local search implementation uses Optuna to choose instruction/demo
  combinations.
- In zero-shot mode, demo candidates are set to `None` after instruction
  proposal, so search only varies instructions.

## 5. Likely Changes Needed For The Decoder Flow

MIPROv2 can probably run through a thin DSPy module wrapper, but the output
will not naturally be the three-slot structure defined in
`decoder_slot_addendum_v0.yaml`.

Likely changes or wrappers:

- Represent the decoder template as a DSPy program whose optimized instruction
  surface maps cleanly to rendered Markdown prompts.
- Decide whether MIPROv2 gets one combined addendum candidate or a custom
  representation of the three slots.
- Enforce zero-shot settings for the first pass by setting both
  `max_bootstrapped_demos=0` and `max_labeled_demos=0` in compile params or
  optimizer construction.
- Record actual auto-derived settings or explicit settings in experiment
  artifacts, even though the planning doc intentionally avoids setting a hard
  budget now.
- Consider disabling `fewshot_aware_proposer` in zero-shot mode if it adds
  irrelevant context after demos are removed.
- Ensure prompt rendering and decoder LLM calls route through the planned
  dr-providers path, while MIPROv2 still receives enough program/task context to
  propose useful instructions.
- Persist candidate instructions, selected Optuna params, rendered prompts, raw
  decoder outputs, scores, and failure buckets.
- Add validation that MIPROv2 candidates can be translated back to the
  slot-addendum contract before comparing them to the curated grid.

## Sources

- EMNLP 2024 paper page:
  https://aclanthology.org/2024.emnlp-main.525/
- arXiv paper page: https://arxiv.org/abs/2406.11695
- DSPy MIPROv2 docs: https://dspy.ai/api/optimizers/MIPROv2/
- Local implementation: `dspy/teleprompt/mipro/optimizer.py`
- Local settings/search/proposal code: `dspy/teleprompt/mipro/`
- Local compile params: `dspy/teleprompt/compile_params.py`
