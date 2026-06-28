# Current goal

Consolidate the two HumanEval code-generation eval pipelines (`direct` and
`enc-dec`) into **one node-graph experiment platform** with a single unified DB
schema, so that the *generation shape* and (next) the *prompt optimizer* are
pluggable without reshaping storage or orchestration.

After consolidation, the next milestone is an **outer prompt optimizer (COPRO,
instruction-only)** that searches the encoder instruction.

Full design + roadmap: [`docs/generation-experiment-design.md`](docs/generation-experiment-design.md).
Live design viewer (open in a browser): `dbos_flow.html`.

## Working context

- Dedicated worktree (`eval-platform-v1`) branched off `enc-dec`; existing
  experiments on `enc-dec` are untouched.
- `_v1` is built **alongside** `_v0`; retire `_v0` only after validation.
- **Validation is done jointly, step by step, with live runs** — the user wants
  to be actively involved in debugging/validation, so agents do non-live checks
  only and check back before any live (paid / daemon) runs.

## Status

**Phase 0 — contracts & unified schema — DONE**
- `src/dr_dspy/experiment_spec.py`, `src/dr_dspy/eval_records.py` + tests.
- Viewer schema section → unified table.

**Phase 1 — node executor + unified pipeline — CODE DONE, non-live validated**
- `src/dr_dspy/node_runner.py` (+ tests) — generic topological executor;
  per-node instruction-built signatures (cached); budgeted-encoder handling.
- `src/dr_dspy/humaneval_eval_dbos.py` — the single DBOS module replacing both
  v0 modules (workflows/steps/CLI + `EvalExperiment` backend; generation via
  `node_runner.execute_graph`, writes via `eval_records`).
- `scripts/humaneval_eval_dbos_v1.py` — direct (1-node) + enc-dec (2-node)
  graphs + sweeps (`EVAL_V1_PIPELINE` selects pipeline).
- `tests/test_eval_submit_offsets.py` — offset/total-jobs mixed-radix.
- Viewer pipeline section → single node-graph pipeline.
- Non-live checks passing: **99 unit tests**, `ruff` + `ty` clean (pre-commit
  enforced), `init-db` applied the unified schema to the real DB
  (`dr_dspy_predictions` 26 cols, `dr_dspy_experiments` 9 cols), CLI registers
  `init-db/submit/worker/enqueue-scores/repair` for both pipelines.

**Phase 1 — live parity gate — DEFERRED to joint validation**
- Reframed: exact v1-vs-v0 score equality is NOT a sound test (LLM generation
  isn't bit-reproducible even at temp 0). The deterministic parts (offset/
  identity math, schema) are unit-verified. The live gate is an **end-to-end
  smoke**: `submit → generate → score` produces correctly-shaped, sensibly-
  valued scored rows (direct: 1 artifact; enc-dec: 2 artifacts, compression
  source = encode output), using the unchanged `score_humaneval_prediction`.
- To run together: `init-db` (done) → `submit --sample-count 1` → background
  `worker` → poll predictions until scored → inspect rows. Direct floor ≈ 16
  cheap LLM calls (16-graph sweep); enc-dec ≈ 112.

## Next (implement through Phase 4, validating step by step together)
- **Phase 2** — DuckDB read-side projection (bronze→silver) over the unified
  JSONB.
- **Phase 3** — optimizer substrate: pinned eval-set split, `config → score`
  contract, study/candidate tables, outer DBOS study driven by a manual/grid
  candidate list (no proposer yet).
- **Phase 4** — COPRO: coordinate-ascent study with the two (reimplemented)
  proposal ops, optimizing the encoder instruction; reps + val/test discipline.
- After live parity passes: retire `_v0` modules/scripts + `experiment_dimensions.py`.

## Commits on `eval-platform-v1`
GOAL → Phase 0 contracts → Phase 0 viewer → Phase 1 executor → Phase 1 viewer
→ Phase 1 unified module/script/offsets test.

## Out of scope (later)
DAG branch/merge execution, node-level artifact caching, MIPRO/demos.
