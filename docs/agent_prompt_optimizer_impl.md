# Agent Prompt: Optimizer Implementation Investigation

You are helping with a `dr-dspy` experiment-planning task. Do not implement
changes yet. Your job is to investigate how the target optimizer approaches are
implemented in this repo and prepare for an iterative planning conversation.

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

Read these docs before investigating code:

- `docs/starting_state.md`
- `docs/exps/opt-dec-format/PLAN.md`

The current practical target optimizer set is:

1. slot optimization over fixed templates
2. custom prompt-grid / best-of-N search
3. `COPRO`
4. zero-shot-first `MIPROv2`
5. `GEPA`

## Investigation Scope

Focus on how each target optimizer would actually be wired for this decoder
format experiment.

Inspect these areas:

- `dspy/teleprompt/copro_optimizer.py`
- `dspy/teleprompt/mipro/`
- `dspy/teleprompt/gepa/`
- `dspy/teleprompt/compile_params.py`
- `dspy/teleprompt/compilation.py`
- `dspy/teleprompt/core/`
- `dspy/task_spec/`
- `dspy/runtime/`
- relevant tests under `tests/teleprompt/`

Also keep in mind that LLM calls may eventually route through
`../dr-providers/` instead of normal DSPy formatting, but do not solve that
integration unless asked.

## Questions To Answer

For each of the five target methods, prepare concise notes on:

- What object/function is the main entry point?
- What compile params are required?
- What kind of program/module shape does it expect?
- Does it optimize instructions, output prefixes, demos, full templates, or
  something else?
- What metric shape does it expect?
- Where are candidate prompts/programs represented?
- What does it return in `CompileResult`?
- What traces/logs/candidate artifacts are already available?
- What assumptions might conflict with a decoder-only custom prompt/template
  experiment?
- What is the smallest smoke-test version we could run?

For `MIPROv2`, specifically check how to configure a zero-shot run with no
bootstrapped or labeled demos.

For `GEPA`, specifically check the feedback metric contract and what it needs
from parse/test failures to produce useful reflective mutations.

For slot optimization and custom best-of-N, identify whether they should be
implemented as small local experiment harnesses or as teleprompter-compatible
classes.

## Deliverable For The Conversation

Bring back a structured summary, not code:

- recommended implementation shape for each method
- likely blockers
- smallest runnable smoke test for each method
- places where we should preserve or extend logging
- open questions that need user decisions

Keep the output focused on preparing the next planning conversation.

