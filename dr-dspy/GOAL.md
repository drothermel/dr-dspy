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

- This is a dedicated worktree (`eval-platform-v1`) branched off `enc-dec`, kept
  separate so in-progress experiments on `enc-dec` are untouched.
- Build the new `_v1` design **alongside** the existing `_v0` modules; retire
  `_v0` only after the parity gate passes.

## Active plan: Phase 0 + Phase 1

**Locked decisions**
- One unified predictions table, discriminated by a `pipeline` column: typed
  control-plane spine + JSONB `dimensions`/`artifacts` payload.
- `instruction` is a per-node runtime parameter (in `dimensions`, hashed into
  identity) — the enabler for COPRO.
- Generation = a DAG of nodes; implement linear-chain execution now, keep the
  data model DAG-ready. Scoring stays shape-agnostic (reads the terminal code).
- The viewer (`dbos_flow.html`) is updated within each phase so it always
  reflects the current design.

**Phase 0 — contracts & unified schema (no behavior change)**
- `src/dr_dspy/experiment_spec.py`: `NodeSpec` / `GraphSpec` / `Dimensions` /
  `ExperimentConfig` / `PredictionPayload`; canonical identity hashing.
- `src/dr_dspy/eval_records.py`: unified `dr_dspy_predictions` +
  `dr_dspy_experiments` DDL and payload (de)serialization.
- Tests: `tests/test_experiment_spec.py`, `tests/test_eval_records.py` (DB-mocked).
- Viewer: collapse the schema section to the single unified table.

**Phase 1 — node executor + unified pipeline (parity-gated)**
- `src/dr_dspy/node_runner.py`: op registry + `execute_graph` (per-node signature
  build = instruction override; per-node checkpoint knob).
- `src/dr_dspy/humaneval_eval_dbos.py`: the single DBOS module (workflows/steps/
  CLI) replacing both `humaneval_*_dbos.py`, reusing existing shared infra.
- `scripts/humaneval_eval_dbos_v1.py`: direct (1-node) + enc-dec (2-node) graph
  configs.
- Tests: `tests/test_node_runner.py`.
- Viewer: collapse the two stacked pipeline rows into one node-graph pipeline.
- **Parity gate:** `_v1` per-task scores/compression match `_v0` at a fixed seed;
  then mark `_v0` (+ `experiment_dimensions.py`) for deletion.

## Out of scope (later phases)
COPRO + optimizer substrate (Phases 3–4), DuckDB read-side (Phase 2), persistent
scoring process pool (Phase 5), DAG branch/merge, node-level artifact caching.
