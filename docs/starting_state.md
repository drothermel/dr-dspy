# Code Compression Optimization Starting State

## Goal

The top-level goal is to optimize an LLM-based code compression pipeline:

```text
code sample
  -> encoder LLM
  -> description
  -> lossless compression
  -> compressed-size measurement
  -> lossless decompression
  -> decoder LLM
  -> reconstructed code
  -> evaluation against the original sample's tests
```

The reconstructed code does not need to preserve the original surface form.
The main objective is behavioral correctness: ideally the reconstructed code
passes 100% of the tests written for the original code sample. Among outputs
that preserve behavior, the secondary objective is smaller compressed
description size.

In optimization terms:

1. Maximize correctness, measured by test pass rate.
2. Subject to correctness, minimize the compressed size of the encoder output.

## Current Context

The full code-sample-to-evaluation pipeline is treated as a black box for the
DSPy optimization work. It takes harness information and configuration, runs the
encode/decode/evaluate sequence, and returns the resulting outputs, metrics, and
logs.

Earlier attempts used relatively naive MIPRO and GEPA setups. The results were
hard to interpret, likely because several parts of the system were still in
flux: sample sizes may have been too small, initial prompts may have been weak,
LLM output parsing into code may have been brittle, and pipeline observability
was not yet strong enough.

The current state is that the pipeline has been solidified with clearer metrics,
logging, and evaluation outputs. The next phase is to use this `dr-dspy` fork,
possibly with adapter changes, to optimize the encoder prompt on the
HumanEval/HumanEval+ dataset.

## Relevant Files

This repo, `dr-dspy`, is the expected optimizer surface:

- `dspy/teleprompt/` - DSPy optimizers, including MIPRO/GEPA-adjacent optimizer
  machinery.
- `dspy/adapters/` - adapter layer that may need changes for prompt rendering,
  parsing, or metric-friendly interaction with the black-box pipeline.
- `dspy/task_spec/` - TaskSpec contracts for describing optimizer-facing tasks.
- `dspy/runtime/` - async run context, logging, tracing, and call accounting.
- `tests/teleprompt/`, `tests/adapters/`, `tests/runtime/` - nearby test
  coverage for optimizer, adapter, and runtime behavior.

The black-box pipeline currently lives in the sibling `dr-bottleneck` repo:

- `../dr-bottleneck/configs/workflows/humaneval_encode_decode.yaml` - baseline
  encode/decode/evaluate workflow and prompts.
- `../dr-bottleneck/configs/openrouter_profiles.yaml` - OpenRouter model
  profiles used by workflow lanes.
- `../dr-bottleneck/src/dr_bottleneck/workflow/engine.py` - YAML workflow
  loading, prompt rendering, lane/profile resolution, and LLM/process handler
  construction.
- `../dr-bottleneck/src/dr_bottleneck/experiments/humaneval.py` - HumanEval+
  loading, job expansion, source construction, and the current
  `humaneval_compress_ast` process handler.
- `../dr-bottleneck/scripts/run_humaneval_demo.py` - command-line entry point
  for HumanEval encode/decode/evaluate sweeps.
- `../dr-bottleneck/scripts/preview_humaneval_prompts.py` - prompt preview
  helper that renders workflow prompts without making LLM calls.
- `../dr-bottleneck/README.md` - operational notes for local services,
  workflow runs, metrics storage, and HumanEval demo usage.

## Baseline Workflow Prompts

From `../dr-bottleneck/configs/workflows/humaneval_encode_decode.yaml`.

Encoder prompt:

````text
Provide a concise natural language description of the code using at most {budget} characters.

```python
{source_code}
```
````

Decoder prompt:

```text
Write functional code in Python according to the description.

"""
{encode}
"""
```

The workflow then runs a process step:

```yaml
name: evaluate
kind: process
handler: humaneval_compress_ast
config:
  encode_step: encode
  decode_step: decode
  zstd_level: 22
```

The configured lanes use the same model for encode and decode within each lane:

- `nemotron` - `openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5/off/v1`
- `gemini` - `openrouter/google/gemini-3.1-flash-lite/off/v1`
- `gpt-oss` - `openrouter/openai/gpt-oss-20b/low/v1`

## Notes About The Baseline Prompts

- The encoder prompt is intentionally minimal. It asks for a concise natural
  language description under a character budget, but it does not explicitly ask
  the model to preserve signatures, imports, constants, edge cases, invariants,
  exception behavior, or hidden-test-relevant details.
- The encoder prompt optimizes for character count, not necessarily compressed
  byte size after zstd. Those are related but not identical objectives.
- The "natural language description" constraint may leave compression
  performance on the table. A compact semi-structured representation may be more
  efficient and more decoder-friendly than prose.
- The decoder prompt is generic. It does not explicitly require "code only",
  complete module output, preserving the public API, matching the original entry
  point, avoiding explanations, or inferring behavior needed for hidden tests.
- The prompts do not state the metric ordering: correctness first, compressed
  size second.
- Using the same model for encode and decode within a lane makes model-lane
  comparisons straightforward, but it also couples encoder-prompt quality with
  decoder capability.
- The visible process handler in the current workflow measures raw encoded
  length, zstd-compressed encoded length, and whether the decoded text parses as
  Python AST. The broader target state is test-based correctness; optimizer
  experiments should make sure the metric used by DSPy reflects that target
  rather than only AST validity.

## Proposed Optimization Sequence

Because both the encoder and decoder prompts affect the final result, the first
step is to isolate and validate the decoder prompt before optimizing the full
encode/decode pathway. The proposed sequence is:

1. Optimize a decoder-only formatting/control task.

   ```text
   ground-truth signature + ground-truth docstring + decoder prompt
     -> decoder LLM
     -> generated code
   ```

   The initial metric should be AST parseability. This is a low-noise way to
   test whether each candidate decoder model can reliably emit valid Python code
   and to compare optimizer behavior on an easier objective.

2. Optimize decoder correctness on the same isolated task.

   After parseability is reliable, switch the metric to test pass rate using the
   HumanEval/HumanEval+ tests. This measures whether prompt optimization can
   improve behavioral reconstruction when the decoder receives high-quality
   semantic input: the original signature and docstring.

3. Freeze a validated decoder prompt.

   The frozen decoder prompt should be selected from the decoder-only
   experiments, with reporting that shows how much optimization moved the
   relevant metrics.

4. Optimize the real encoder-only objective.

   With the decoder prompt fixed, optimize only the encoder prompt in the full
   pathway:

   ```text
   source code
     -> encoder prompt + encoder LLM
     -> compressed/decompressed description
     -> frozen decoder prompt + decoder LLM
     -> reconstructed code
     -> tests and compression metrics
   ```

   This is the actual target task: improve end-to-end compression/correctness by
   changing only the encoder prompt.

Recommended report metrics for the decoder-freezing phase:

- Baseline AST parse rate.
- AST parse rate after optimizing for parseability.
- Test pass rate after optimizing for parseability.
- Test pass rate after optimizing for test passing.
- Metrics broken out by model/lane, since decoder reliability may vary sharply
  across the configured models.

Areas to take care:

- Avoid overfitting the decoder prompt to the exact "signature + docstring"
  input if the later decoder input will be an encoder-generated compressed
  description.
- Prefer a decoder prompt that can handle both clear docstring-style
  descriptions and denser encoder-produced descriptions.
- Keep the decoder-only task close enough to the final setting that improvements
  transfer to the full pipeline.
- Make the optimization metric match the stage: parseability for initial
  formatting control, test pass rate for decoder correctness, and
  correctness-plus-compression for encoder optimization.
- Report parseability and pass rate together. Optimizing for valid Python may
  improve formatting without improving behavioral correctness.
- Freeze the decoder prompt before encoder optimization so later experiments can
  attribute performance changes to the encoder prompt instead of decoder drift.

## Decoder Template Boundary

The planned DSPy work may discard DSPy's default prompt formatting and route LLM
generations through `../dr-providers/`. That makes the prompt boundary explicit:
the experiment needs to decide which facts are hard-coded into the decoder
template and which facts must be recovered from the encoder output.

Two fixed decoder template options are under consideration.

Option A gives the decoder the expected signature as side-channel context:

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

Option B gives the decoder only the encoded description:

````text
{text_a}

```markdown
{encoded_description}
```

{text_c}
````

The conclusion for now is to evaluate both options before deciding how to frame
the full encode/decode optimization.

Initial prompt templates for this decoder-format phase live in
`configs/prompts/templates/`. The current baseline encoder prompt is
`baseline_enc.md`. Decoder templates are materialized as
`baseline_dec`, `manual_dec_v0`, and `manual_dec_v1` crossed with
`variantA`/`variantB`:

- `variantA` includes `{signature}` as side-channel context plus
  `{encoded_description}`.
- `variantB` receives only `{encoded_description}`.
- `manual_dec_v0` is the minimal manual decoder baseline with code-only and
  parseability constraints.
- `manual_dec_v1` adds compact guidance about preserving the interface,
  handling implied edge cases, using straightforward standard-library Python,
  and avoiding tests/placeholders.

Option A is likely the best first mainline because it keeps the test-facing
interface outside the compression budget. The encoder can then focus on
implementation behavior: algorithm, edge cases, constants, imports, helper
logic, and invariants. This makes the optimization task easier to interpret and
keeps decoder prompt validation closer to "fill in a correct implementation
under a known interface."

Option B is more faithful to whole-program compression because the encoder must
communicate the interface and the behavior. If it works cleanly, it is the
stronger formulation. The concern is that it may be much noisier: failures can
come from missing or malformed signatures, wrong entry points, invalid Python,
or incorrect behavior. If Option B is fiddly, it should be treated as a future
direction while Option A is used for the first end-to-end optimization pass.

Concerns to keep in mind:

- Track signature recovery separately for Option B. A failed test run should be
  distinguishable from "expected entry point missing" or "call signature
  incompatible."
- Record at least AST parseability, expected entry point existence, callable
  signature compatibility, test pass rate, and compressed description size.
- Option A changes the compression problem by giving the signature for free.
  Reports should state this clearly.
- Option B may require the encoder to spend budget on interface details that are
  mechanically available from the source/harness.
- The decoder prompt chosen for Option A should still be robust to dense,
  encoder-generated descriptions, not only natural docstrings.
- If Option B works, compare it against Option A on both pass rate and compressed
  size. A smaller or more faithful formulation is only useful if it remains
  stable enough for optimization.
