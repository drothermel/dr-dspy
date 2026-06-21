# COPRO

## Synthesis For Decoder-Format Optimization

COPRO is the simplest learned optimizer we plan to try after the manual slot
grid. It is useful here because it directly tests the question, "Can an LLM
propose better decoder-control wording than our hand-written addenda, using
only metric feedback?" That makes it a good first learned-method probe for AST
parseability.

The main mismatch is that dr-dspy's current COPRO implementation optimizes a
predictor's `TaskSpec.instructions` and the final output field prefix. Our
decoder-format design wants three bounded slots inside plain Markdown templates:
`task_instructions`, `output_instructions`, and `failure_avoidance`. We should
expect COPRO to be easy to run only if we first create a slot-aware bridge, or
if we accept a less-controlled COPRO run that rewrites a larger instruction
surface.

Open questions:

- Should COPRO be used only as a baseline over a single combined addendum, or
  should it be adapted to propose the three slots separately?
- Do we want COPRO candidates capped and validated like
  `decoder_slot_addendum_v0.yaml`, or should the first run intentionally expose
  what unconstrained COPRO wants to write?
- How much value do we get from COPRO's output-prefix optimization if our final
  decoder prompt is rendered as a plain Markdown template and raw code output?

## 1. Method Overview

COPRO is an early DSPy teleprompter for automatic instruction optimization. I
did not find a standalone COPRO research paper analogous to the MIPRO or GEPA
papers. The reliable public description is the DSPy optimizer documentation,
which presents COPRO as an optimizer for a student program's signature and notes
that it can optimize a zero-shot or already pre-optimized program.

In the broader DSPy ecosystem, COPRO is now best treated as a simple,
interpretable instruction-search baseline rather than the current frontier.
Public DSPy learning material describes it as a coordinate-ascent or
hill-climbing method that generates and refines instructions using a metric and
training set. Later methods such as MIPROv2 and GEPA are considered stronger
because they use richer proposal context, better search, few-shot/demo
selection, or natural-language feedback.

For our purposes, that older/simple status is a feature. COPRO should tell us
whether a minimal learned instruction search can beat the manual prompt grid on
formatting control before we spend complexity on MIPROv2 or GEPA.

## 2. How It Works

COPRO runs a propose/evaluate/refine loop:

1. For each predictor, read the current task instructions and final output
   field prefix.
2. Ask a prompt model to propose `breadth - 1` new instruction/prefix pairs.
   The original instruction/prefix pair is appended as a baseline candidate.
3. Evaluate candidate programs against the trainset with the task metric.
4. Keep the best observed candidate for the predictor.
5. For additional `depth` rounds, give the optimizer model prior attempts and
   scores, then ask it to propose another batch of candidates.
6. Return the highest-scoring candidate program, with optional candidate and
   depth stats.

Strengths:

- Simple and inspectable. Candidates are explicit instruction/prefix strings.
- Good for zero-shot instruction optimization where examples/demos are not the
  thing being optimized.
- Useful as a smoke test for metric wiring because the control loop is small.
- Cheaper and easier to reason about than MIPROv2 or GEPA.

Weaknesses:

- It receives scalar scores, not rich structured failure feedback.
- It does not naturally optimize multiple named slots with independent caps.
- It can overfit quickly when the trainset is tiny.
- It is likely sensitive to noisy metrics, because it greedily keeps candidates
  based on observed score.
- It changes the final output field prefix as well as instructions, which may
  not map cleanly to our plain-template decoder flow.

## 3. Interaction With Our Experiment

COPRO is a good fit for the formatting pass if we treat it as a constrained
instruction-wording probe. It should be run after the curated grid because the
grid gives us deterministic candidate baselines and failure buckets to compare
against.

Expected behavior:

- COPRO may quickly discover "output only code" and "no markdown fences" style
  instructions if parseability is the metric.
- It may improve AST parseability without improving entry-point recovery,
  signature compatibility, or test pass rate.
- Option A may look easier because the signature is fixed outside the
  docstring. Option B will tell us whether COPRO learns wording that helps the
  model recover interface details from docstring-only input.
- If the metric is only AST parseability, COPRO may learn instructions that
  produce valid but semantically useless code. We should report pass-rate and
  failure buckets alongside parseability even in parse-first runs.

Questions to watch:

- Does COPRO mostly rediscover the manual slot grid, or does it find genuinely
  different wording?
- Does COPRO's best instruction transfer across Option A and Option B, or are
  their failure modes different enough to require separate prompts?
- Does the optimized prefix matter once we render raw Markdown templates and ask
  for code-only output?
- Are score ties and duplicate candidates common enough that candidate
  deduplication affects reporting?

## 4. dr-dspy Implementation

Local entry points:

- `dspy/teleprompt/copro_optimizer.py`
- `dspy/teleprompt/copro/task_specs.py`
- `dspy/teleprompt/compile_params.py`
- Tests in `tests/teleprompt/test_copro_optimizer.py`

The class is registered as a teleprompter for `COPROCompileParams`. Constructor
hyperparameters currently include:

- `metric`: optimizer metric used by the evaluator. For the formatting pass,
  this is initially AST parseability and later may include entry-point or test
  information.
- `prompt_model`: optional separate LM for generating/refining candidates. If
  unset, the optimizer uses the run LM.
- `breadth`: number of candidates considered per proposal round. It must be
  greater than 1. Higher values improve exploration and increase candidate
  evaluation cost.
- `depth`: number of refinement rounds. Higher depth gives COPRO a chance to
  learn from previous attempts, but also makes overfitting and cost more likely.
- `init_temperature`: temperature for instruction proposal. Higher values
  increase diversity and instability.
- `track_stats`: when true, returns depth-level score summaries in
  `CompileStats`.

Compile parameters currently include:

- `trainset`: examples evaluated during optimization.
- `evaluate.max_concurrency`: controls evaluation concurrency.
- `evaluate.display_progress`, `display_table`, `save_as_csv`,
  `save_as_json`: reporting controls forwarded to evaluation.
- `evaluate.max_errors`: read when building the evaluator.

Important implementation behavior:

- COPRO deep-copies the student, then mutates each predictor's
  `TaskSpec.instructions` and final output field prefix.
- Candidate generation uses `BasicGenerateInstructionTaskSpec` first and
  `GenerateInstructionGivenAttemptsTaskSpec` for refinement.
- Each candidate is evaluated with `make_optimizer_evaluator`.
- The returned `CompileResult` includes the best program, ranked candidates,
  and metric-call count.

## 5. Likely Changes Needed For The Decoder Flow

COPRO can run as-is only if the decoder experiment is represented as a normal
DSPy predictor where the prompt surface we care about is encoded in
`TaskSpec.instructions` and the final output prefix. That is not the plan's
preferred shape. The plan wants plain Markdown templates and three bounded
slots.

Likely changes or wrappers:

- Add a slot-aware optimization wrapper that maps COPRO proposals into
  `task_instructions`, `output_instructions`, and `failure_avoidance`, or choose
  a single combined addendum for the COPRO baseline.
- Enforce the same 100-character slot cap and record raw versus rendered text
  if COPRO is allowed to propose slot values.
- Record rendered prompts and raw model outputs for each candidate, not only the
  optimized `TaskSpec`.
- Decide whether COPRO's final output prefix should be ignored, mapped into
  `output_instructions`, or preserved as a separate candidate dimension.
- Ensure candidate evaluation can route decoder LLM calls through
  `../dr-providers/` instead of relying on normal DSPy adapter formatting.
- Add explicit candidate artifacts: candidate id, option A/B, slot values or
  instruction/prefix, rendered prompt, raw output, score, and failure bucket.

## Sources

- DSPy COPRO docs: https://dspy.ai/api/optimizers/COPRO/
- DSPy optimizer overview mentioning COPRO coordinate ascent:
  https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/optimization/optimizers.md
- Local implementation: `dspy/teleprompt/copro_optimizer.py`
- Local compile params: `dspy/teleprompt/compile_params.py`
- Local tests: `tests/teleprompt/test_copro_optimizer.py`
