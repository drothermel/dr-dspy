# Agent Prompt: Decoder Prompt Templates And Configs

You are helping with a `dr-dspy` experiment-planning task. Do not implement
runtime behavior yet. Your job is to investigate and prepare prompt template and
YAML/config design options that codify the decisions already made for the
decoder-format optimization phase.

## Overall Goal

We are building an LLM + code pipeline for code compression:

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

The long-term goal is to optimize the encoder prompt so reconstructed code
preserves behavior while minimizing compressed description size. Before that,
we want to freeze a good decoder prompt/template. The current phase is
`opt-dec-format`: optimize decoder formatting first, with AST parseability as
the initial metric, then test pass rate.

## Read First

Read these docs before investigating files:

- `docs/starting_state.md`
- `docs/exps/opt-dec-format/PLAN.md`

Relevant sibling repos:

- `../dr-bottleneck/` contains the current HumanEval encode/decode/evaluate
  workflow and YAML config style.
- `../dr-providers/` contains the typed provider client that may be used for LLM
  generations.

## Decisions Already Made

Initial model:

- `openrouter/openai/gpt-oss-20b/low/v1`

Decoder template options:

- Option A: decoder receives `{signature}` as side-channel context plus
  `{encoded_description}`.
- Option B: decoder receives only `{encoded_description}` and must recover the
  interface from the description.

Prompt template files:

- Store initial prompt templates as Markdown files under
  `configs/prompts/templates/`.
- Keep the encoder baseline as a single file:
  `configs/prompts/templates/baseline_enc.md`.
- Materialize the decoder template Cartesian product for the initial manual
  comparison:
  - `baseline_dec_variantA.md`
  - `baseline_dec_variantB.md`
  - `manual_dec_v0_variantA.md`
  - `manual_dec_v0_variantB.md`
  - `manual_dec_v1_variantA.md`
  - `manual_dec_v1_variantB.md`
- `variantA` means the template includes `{signature}` as side-channel context.
- `variantB` means the template receives only `{encoded_description}`.

Step 0 baselines:

- Original baseline:

  ```text
  Write functional code in Python according to the description.
  ```

- Minimal manual baseline:

  ```text
  Write Python code that implements the requested function.

  Output only code, with no markdown fences or explanation.
  The code must parse and run as a complete Python snippet.
  ```

- `manual_dec_v1` is a slightly stronger manual decoder baseline that preserves
  the same formatting constraints while adding compact guidance about interface
  preservation, standard-library code, edge cases, and avoiding tests,
  placeholders, or unrelated top-level behavior.

Target optimizer methods:

1. slot optimization over fixed templates
2. custom prompt-grid / best-of-N search
3. `COPRO`
4. zero-shot-first `MIPROv2`
5. `GEPA`

## Investigation Scope

Inspect existing config and prompt patterns:

- `../dr-bottleneck/configs/workflows/humaneval_encode_decode.yaml`
- `../dr-bottleneck/configs/openrouter_profiles.yaml`
- `../dr-bottleneck/src/dr_bottleneck/workflow/`
- `../dr-bottleneck/scripts/preview_humaneval_prompts.py`
- `../dr-bottleneck/scripts/run_humaneval_demo.py`
- `docs/exps/opt-dec-format/PLAN.md`

Look for the cleanest way to represent:

- prompt templates
- template variants
- optimized slots
- baseline labels
- model/profile choices
- dataset split identifiers
- metric target, such as `ast_parse` vs `pass_rate`
- Option A vs Option B template families

## Questions To Answer

Prepare notes on:

- Where should prompt template files live?
- What file format should they use: Markdown, YAML, plain text, or a mix?
- How should fixed template text be separated from optimized slots?
- How should Option A and Option B be represented without duplication?
- How should the original and minimal manual baselines be named?
- What YAML shape would make experiment runs reproducible and easy to compare?
- What config fields are needed now versus later for encoder optimization?
- What should be recorded so final reports can explain which template and slots
  produced each result?

For slot optimization, propose a small initial family of templates and slots,
for example:

- role or skill phrase
- warning list
- output constraint phrase
- failure-avoidance sentence
- verbosity or target-size hint

Keep these proposals compact. The goal is to support early experiments, not to
design a large prompt-template framework.

## Deliverable For The Conversation

Bring back a structured summary, not code:

- proposed file layout
- proposed template/config schema
- names for baseline/template variants
- candidate slot families for the first run
- example YAML snippets if useful
- open questions that need user decisions

Keep the output focused on preparing the next planning conversation.
