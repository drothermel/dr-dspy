# Generalizing the code-generation experiment platform

**Status:** design note · **Date:** 2026-06-28
**Scope:** HumanEval eval pipelines (`humaneval_direct_dbos`, `humaneval_encdec_dbos`) and their successor

---

## 1. Goal

We are experimenting with **different ways to make a model produce code**, and we
want to **optimize** those ways (starting with prompt optimization). The current
two scripts — `direct` (`prompt → code`) and `enc-dec`
(`code → description → code`) — are really two points in one family: *direct is
enc-dec where a human wrote the "encode" stage (the HumanEval prompt).* If we keep
going we'll add more shapes (`enc → compress → dec`, branch/merge graphs, …), and
each one feels very different conceptually but is a **small box** inside a large,
mostly-shared implementation (submit → queue → generate → score → repair → log).

The objective is **correctness × compression**, where compression is a much more
stable signal than correctness. Correctness is high-variance — cheap OpenRouter
models already near-saturate *direct* correctness, so the room to optimize shows up
in **enc-dec + compression**. HumanEval is mandated (advisor), so we fight the noise
with **cheap models + very high query volume** rather than an easier task.

**Platform goal:** a stable substrate where the *generation shape* and the
*optimization strategy* are pluggable, **every inference output is persisted**, and
runs are durable/resumable — so that trying a new generation shape, or a new
optimizer, does **not** require reshaping the database or the orchestration.

Guiding principle: **stable substrate, pluggable core.**

---

## 2. Intended generalized shape

Three concerns are currently fused; the whole design is about decoupling them so two
stay fixed while one varies:

- **(A) durable-execution topology** — DBOS workflows / steps / queues (checkpoint +
  retry boundaries)
- **(B) generation compute graph** — the DAG of LLM calls / transforms / merges that
  turns task → code  ← *this is the only thing that should vary per experiment*
- **(C) result / log schema** — what we persist

### 2a. Generation is a graph (B)

Represent generation as a **DAG of typed nodes**: `{id, op, inputs:[node_ids],
config}`. `op ∈ {llm_call, transform, merge, …}`.

- direct = 1 node; enc-dec = 2 (linear); `enc→compress→dec` = 3; branch/merge = a
  node with multiple `inputs`.
- A designated **terminal node emits the code**; scoring reads only that, so
  **scoring is 100% shape-agnostic and never changes**.
- **Start with linear-chain execution** (covers direct, enc-dec, enc→compress→dec).
  Keep the data model DAG-ready (explicit `inputs`) so branch/merge is a later,
  non-breaking extension — don't build the merge engine yet.

### 2b. Execution & durability (A)

- One generic `execute_node_step(node, upstream_outputs)` walked over nodes in
  **deterministic topological order** by the generation workflow. Per-node
  checkpointing then comes for free, with **one** step function instead of bespoke
  per-shape steps.
- **Splitting enc/dec into separate steps = the first instance of this.** Don't
  hand-write `encode_step`/`decode_step`; implement the node-runner and let enc-dec
  be a 2-node chain. Same effort, no throwaway code.
- **Durability granularity is a knob, not a fixed choice:** run the whole graph in
  one step (cheap graphs) vs checkpoint per node (expensive multi-LLM graphs, so a
  failed `dec` doesn't re-pay `enc`). Net cost of splitting is ~one extra checkpoint
  write per node; the win is real whenever stages use different models/providers or
  on crash recovery.

### 2c. Schema (C) — typed spine + JSONB payload

Hybrid, **not** full EAV. Split columns by *who reads them*:

- **Typed + indexed control-plane spine** (orchestrator / dispatcher / repair / DBOS
  / indexes touch these hot): `prediction_id`, `experiment_name`, `task_id`,
  `submission_id`, `generation_status`, `scoring_status`, `score`, `failure_class`,
  `provider_cost`, timestamps.
- **JSONB payload** for everything that varies by shape: `dimensions` (the
  experiment axes, **including the graph spec and each node's config**),
  `artifacts` (keyed by node id: `{output, usage, cost, response_metadata}`),
  eval/compression metrics, verbose error detail. `_experiments` config
  (instructions, signatures, static params) collapses into one JSONB too.
- **Identity:** `prediction_id = stable hash of canonicalized dimensions` (PK does
  dedup; keep the existing `stable_json` discipline as the single source of truth).
- **One unified predictions table** discriminated by a `pipeline` column instead of
  two parallel tables — that's the logical end of "don't enforce a split between
  near-identical workflows."
- **Read side = DuckDB**, not Postgres ETL: the raw table is an append-only log
  (bronze); analysis parses/validates the JSONB into typed shapes via DuckDB
  views/marimo (silver). Cheap projection, given the stack.

This makes the generation shape collapse into **(1) a graph spec in config (hashed
into identity) + (2) a node-keyed artifacts bag** — and *nothing else moves.*

### 2d. Optimization as an outer DBOS "study"

The optimizer is a **driver on top of the substrate**, not a library we hand control
to. A candidate config is just a normal experiment run, so **every inference it
evaluates is persisted** (the thing DSPy optimizers throw away).

- A **study workflow** reuses the existing submit-dispatcher loop pattern: propose
  candidate config(s) → submit over a **pinned eval set** (fixed task subset + seed +
  reps) → read **aggregate score (+ full distribution)** → select / propose next →
  loop. Durable + resumable = the checkpointing control we want.
- Substrate already supplies the two things any optimizer needs: a **metric**
  (test-pass score) and a **dataset** (HumanEval samples).
- A thin **study / candidate** table tracks the outer loop (study_id, strategy,
  params, search-space spec, eval-set spec, per-candidate: round, node, config,
  proposal provenance, linked experiment, aggregate score + distribution). Inner
  predictions unchanged.

### 2e. First strategy: COPRO (instruction-only), worked through

COPRO (`dspy/teleprompt/copro_optimizer.py`) is greedy coordinate ascent that
proposes instruction text via a prompt model and **leaves few-shot demos untouched**
— exactly right for "demos don't help me, optimize the prompt itself." (It also tunes
the last output field's `prefix`; **we intentionally use instruction only.**)

Our **single optimization input = the encoder instruction**, which is one knob
whether the graph is direct (1 node) or enc-dec (2 nodes) — so the optimizer's
interface is identical across shapes.

Mapping COPRO onto the substrate — it needs four things, three of which Phase 0–3
already build:

1. **Enabler: instruction as a per-node *runtime* config**, threaded into the
   signature build per job (today it's baked at construction in
   `build_dspy_signature → make_signature`), and **placed in `dimensions` → into
   `prediction_id`.** Once instruction is a dimension, every candidate's per-task
   outputs are individually addressable and logged.
2. **`config → aggregate-score over a pinned eval set`** — a normal experiment
   submission + a group-by-mean read of `score` (keep the distribution too).
3. **Outer study round** = propose step (meta-LLM → `breadth` instructions, logged)
   → one batch experiment of `candidates × evalset × reps` (existing dispatcher) →
   select best → propose-given-attempts step (sorted instruction+score history →
   next batch). Loop `depth`.
4. **Reimplement the two proposal signatures** (`BasicGenerateInstruction`,
   `GenerateInstructionGivenAttempts`) as our own logged ops; keep COPRO's
   **`prompt_model` ≠ task_model** separation.

Two wins for free vs stock COPRO: **content-addressed dedup** (duplicate candidates
reuse cached predictions via `ON CONFLICT DO NOTHING` instead of re-scoring — COPRO
re-evaluates duplicates), and **full per-eval provenance** (COPRO keeps only the
aggregate `.score` and candidate-program copies, discarding every per-example
generation).

---

## 3. Gotchas

- **Canonical hashing.** `dimensions` JSON must be canonicalized (stable key order)
  before hashing or dedup/identity breaks. Instruction (and the whole graph spec)
  must be in the hash.
- **Pin the eval set across candidates** so scores are comparable *and* so
  content-addressed reuse is valid (same tasks/seed/reps, only the instruction
  differs).
- **Repair ordering & partial indexes reference dimension columns today**
  (`REPAIR_ORDER_COLUMNS` = `model, temperature, …`; `idx_…_model`). Moving
  dimensions into JSONB means these must move to `created_at` / `sample_index` / a
  generated `dimensions_digest` column / GIN.
- **"Log anything, never fail" can hide bugs.** Validate with Pydantic *before*
  `model_dump()` into JSONB; stamp `schema_version` + `pipeline` so the read-side
  projection can evolve without re-migrating.
- **Don't bury the control plane.** Status / score / ids / failure_class must stay
  typed — orchestration, repair selection, and partial indexes depend on them.
- **DBOS step memoization is keyed by call sequence**, so the topo order driving
  `execute_node_step` must be deterministic for correct recovery.
- **Large node outputs** → checkpoint a reference + store the blob in the app DB (not
  an issue at current text sizes).
- **Noise is the enemy.** Correctness is high-variance (small subsets, binary
  pass/fail); compression is steadier. COPRO's greedy ascent **will chase noise** —
  mitigate with a sizable pinned **val** set, **repetitions** to average, and a
  **held-out test** set never selected on. Because we keep the full per-task
  distribution, select on mean-with-variance / paired comparison, not bare mean.
- **enc-dec has the optimization headroom**, not direct (cheap models saturate direct
  correctness; the compression objective opens room). Start single-node to prove the
  loop, then enc-dec.
- **COPRO multi-predictor ascent is acknowledged-imperfect** (explicit `TODO`) and
  cost grows as candidate sets accumulate — our own coordinate-ascent loop can fix
  the ordering and bound cost.

---

## 4. Implementation phases (high level)

Ordered by dependency; the substrate must exist before anything that rides on it.

- **Phase 0 — Consolidate impl + storage.** One unified design: typed spine + JSONB
  payload, single predictions table (discriminated by `pipeline`), graph-spec config,
  **instruction as per-node runtime config in `dimensions`**, canonical hashing. Fresh
  `_v1` tables (the current ones are `_v0`).
- **Phase 1 — Generic node execution.** Replace the bespoke generate step with
  `execute_node_step` over a linear node chain; enc-dec becomes a 2-node chain;
  per-node checkpoint knob. Scoring untouched.
- **Phase 2 — Read side.** DuckDB projection/views (bronze→silver) that parse +
  validate the JSONB into typed analysis shapes.
- **Phase 3 — Optimizer substrate.** `config → score` contract over a pinned eval
  set; study/candidate tables; outer DBOS study skeleton driven by manual/grid
  candidates first (prove the loop before adding a proposer).
- **Phase 4 — COPRO.** Coordinate-ascent study with the two (reimplemented) proposal
  ops, optimizing the **encoder instruction**, with reps + val/test discipline.
- **Phase 5 — Scoring throughput (independent; slot when it bites).** Replace the
  per-evaluation `subprocess.run` spawn with a **persistent scoring process pool** —
  see §5. Directly enables the "run many more queries" goal.
- **Deferred.** DAG branch/merge execution; node-level content-addressed artifact
  caching (reuse identical encoder outputs across decoder configs); richer optimizers
  (MIPRO-style); few-shot demos (only if they ever start helping).

---

## 5. Appendix — scoring process pool (Phase 5 detail)

Today the scoring stage spawns a fresh OS process for **every** test evaluation and
reaps it; there is **no persistent pool**. The path:
`score_prediction_step` → `score_humaneval_prediction` (`scoring.py:152`) →
`evaluate_human_eval_code` (`human_eval.py:328`) → `run_subprocess_batch`
(`human_eval.py:383`) → `subprocess.run([sys.executable,"-c",runner_script()], …,
timeout=…)` (`human_eval.py:403-410`), spawned **once per top-level function name**.
Scoring parallelism = the scoring queue's `worker_concurrency` (32), which is also the
only throttle on concurrent spawns.

**Problem:** no startup amortization — every evaluation eats a fresh interpreter
launch, multiplied by the per-function-name loop. At high query volume this is a real
ceiling.

**Proposal:** a fixed pool of long-lived child interpreters per worker, reused across
evaluations. **Hard requirements (must not regress):**

1. **Crash isolation** — a child dying from untrusted code must not take down the
   worker; detect + replace it, report the eval as an error.
2. **Hard timeout** — preserve the 15s wall-clock bound. With a warm pool this means
   an explicit **watchdog that kills and respawns** a hung child (a `ProcessPoolExecutor`
   won't cancel a running task), so one bad eval can't permanently consume a slot.
3. **No state bleed** — each eval must `exec` into a fresh namespace; a reused
   interpreter must not leak globals / imports / threads / open resources from prior
   runs.

**Open questions:** pool size vs queue concurrency (which is the real CPU bound?);
pool lifecycle owned by `run_worker_command` startup; revisit the FD budget
(`worker_resources.scoring_subprocess_fd_budget`). **Out of scope:** real sandboxing
(seccomp/rlimits/chroot) — this is process *reuse*, not hardening.

> Symbol-based references are used for the DBOS module back-halves
> (`humaneval_*_dbos.py`, `humaneval_dbos_flow.py`) because those files were being
> reformatted and line numbers drift; `scoring.py` / `human_eval.py` /
> `worker_resources.py` citations are exact.

---

## 6. Implementation status & handoff (Phases 0–1)

All work is on branch **`eval-platform-v1`** (a worktree off `enc-dec`). `_v1` is
built **alongside** `_v0`; retire `_v0` only after live validation passes.
Validation is done **jointly, step by step, with live runs** — automated agents
do non-live checks only and check back before anything paid / daemon-based.

### What's built (code complete, non-live validated)

- **`src/dr_dspy/experiment_spec.py`** — the spec layer (pure data + validation +
  hashing, dependency-light). `FieldSpec` (JSON-serializable signature field;
  `type_name` ∈ {str,int,float,bool,code}), `NodeConfig`
  (model/temperature/reasoning/**instruction**/signature_name/fields/
  output_field/input_bindings/extra), `NodeSpec` (id/op/config; `dependencies()`
  derived from bindings), `GraphSpec` (nodes/terminal_node_id/compression_source;
  validates acyclic + terminal + refs; `topological_order()`). Identity:
  `prediction_id(experiment_name, task_id, graph, repetition_seed)`,
  `dimensions_digest(graph)`, `canonical_dimensions(graph)` =
  `{"graph": graph.model_dump(mode="json")}` hashed via `stable_json` (mirrors
  `humaneval_dbos_flow.stable_prediction_id_from_dimensions`). Payloads:
  `ArtifactRecord`, `PredictionPayload`, `ExperimentConfig`.
- **`src/dr_dspy/eval_records.py`** — unified DDL + row IO. `dr_dspy_experiments`
  (PK `experiment_name`; `pipeline/script_kind/seed/sample_count/config JSONB/
  metadata`) and `dr_dspy_predictions` (typed spine: `prediction_id` PK,
  `experiment_name` FK, `pipeline`, `schema_version`, `script_kind`,
  `submission_id`, `task_id`, `sample_index`, `repetition_seed`,
  `dimensions_digest`, `generation_status`, `generation_failure_class`,
  `scoring_status`, `scoring_failure_class`, `score`, `provider_cost`,
  `raw_generation`, timestamps; JSONB: `dimensions`, `task_inputs`, `artifacts`,
  `metrics`, `errors`; UNIQUE(experiment_name, task_id, dimensions_digest,
  repetition_seed)). `eval_schema_statements()` also emits the shared
  `batch_operation` tables. `PredictionRow`, `CREATION_COLUMNS`,
  `parse_prediction_payload`.
- **`src/dr_dspy/node_runner.py`** — `execute_graph(graph, *, task_inputs,
  run_node)` → `GraphRun` (artifacts / terminal_output / `total_cost()` /
  `compression_source_text()`). `build_node_signature(config)` builds + caches a
  dspy signature **injecting the node instruction**. `make_llm_run_node(...)` =
  the real `llm_call` op. Budgeted encoder: `extra.budget_ratio` →
  `max_characters = max(50, round(ratio*len(source)))`.
- **`src/dr_dspy/humaneval_eval_dbos.py`** — the single DBOS module replacing both
  v0 modules. Unified `PredictionJob` (carries `graph: GraphSpec` + task fields)
  and `EvalSubmitSpec` (carries `graphs: list[GraphSpec]` = the pre-enumerated
  sweep; `total_jobs = sample_count × len(graphs) × repetitions`). Generation via
  `execute_graph`; scoring via the unchanged `score_humaneval_prediction`.
  Implements `ExperimentBackend` (`EvalExperiment`) and reuses the shared infra
  exactly as v0. CLI: `init-db/submit/worker/enqueue-scores/repair`. DBOS
  app/queue/workflow/step names are `*_v1`-suffixed to coexist with v0.
- **`scripts/humaneval_eval_dbos_v1.py`** — builds the direct (1-node) + enc-dec
  (2-node, budgeted encoder) graph families + sweeps (matches v0
  models/instructions/defaults). `EVAL_V1_PIPELINE` selects pipeline.
- **`dbos_flow.html`** — unified schema section + single node-graph pipeline.

### Things the next phases MUST know

- **Identity = the whole graph.** `prediction_id`/`dimensions_digest` hash the
  entire GraphSpec (topology + every node's model/temperature/reasoning/
  **instruction**/extra). Changing an instruction (COPRO) → new `prediction_id`s,
  so every candidate's per-task outputs are individually addressable and logged.
  Canonicalization is `stable_json` (key-order independent) — keep it the single
  source of truth.
- **Where data lives.** Failure *classes* are typed spine columns; full failure
  detail is in the `errors` JSONB keyed `"generation"`/`"scoring"` (cleared on
  success). All scoring detail (compile/extraction/`evaluation_*`/
  `compression_metrics`/ratios) is in the `metrics` JSONB. Generation usage/cost
  live inside `artifacts[node]`. The read-side (Phase 2) and optimizer (Phase 3)
  read these from JSONB, not columns.
- **Repair ordering** uses `dimension_columns=("dimensions_digest",)`,
  `order_columns=("sample_index","repetition_seed")` — the per-axis columns are
  gone.
- **Sweep is data, not flags.** `submit` takes no model/temperature flags; the
  sweep is the pre-enumerated `graphs` list from the script. To change the search
  space (Phases 3–4) build a different `graphs` list — the offset/identity math is
  agnostic. Offset mixed-radix order is (repetition_seed, graph_index,
  sample_index).
- **Deterministic sampler.** `human_eval_sampling.sample_human_eval_tasks(seed,
  sample_count, ...)` is a seeded shuffle — partition its order to build the
  pinned train/val/test split in Phase 3.
- **`config → score` hook (Phase 3).** A candidate = a config → GraphSpec(s);
  evaluate by submitting over a **pinned** task set and reading mean `score`
  (+ distribution) grouped from `dr_dspy_predictions`. Content-addressing means
  re-submitting an identical graph reuses cached rows (`ON CONFLICT DO NOTHING`)
  → free dedup across a study, provided the eval set (seed/samples/reps) is fixed.
- **COPRO (Phase 4).** Instruction is already a `NodeConfig` field — vary
  `node.config.instruction` per candidate; reimplement the two proposal signatures
  as own logged ops; keep `prompt_model ≠ task_model`; start single-node (encoder
  instruction). Greedy ascent + noisy metric → pin a real val set, use reps,
  hold out a test set.
- **Retirement.** `experiment_dimensions.py` (the Dimension/dynamic-DDL
  machinery) is unused by `_v1`; delete it with the v0 modules after live parity.

### Validation status

- **Done (non-live):** 99 unit tests (`test_experiment_spec`, `test_eval_records`,
  `test_node_runner`, `test_eval_submit_offsets`); `ruff` + `ty` clean
  (pre-commit enforced on every commit); **`init-db` applied the unified schema
  to the real DB** (verified `dr_dspy_predictions` 26 cols / `dr_dspy_experiments`
  9 cols); CLI registers all commands for both pipelines.
- **Remaining (live, jointly):** the end-to-end smoke — `submit --sample-count 1`
  → background `worker` → poll `dr_dspy_predictions` until `scored` → inspect rows
  (direct: 1 artifact, score ∈ {0,1}, compression metrics present; enc-dec: 2
  artifacts, compression source = encode output). Reframed gate: **no exact v0
  score equality** (LLM non-determinism); the deterministic parts (offset/identity
  math, schema) are unit-verified. Direct floor ≈16 cheap LLM calls; enc-dec ≈112.
  **Nothing paid has run yet.**
