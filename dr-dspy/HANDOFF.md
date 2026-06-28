# Handoff — Phases 3 + 4 (optimizer substrate + COPRO)

**Status: CODE COMPLETE, non-live validated.** 126 unit tests pass
(99 prior + 27 new), `ruff check` + `ty` clean on all new/edited files.
**Nothing paid/live has run.** Live smoke is the next step, done jointly.

Plan: `~/.claude/plans/ogreat-for-experiment-name-snazzy-piglet.md`.

## What was built (branch `eval-platform-v1`)

New modules:
- `experiment_spec.with_node_instruction()` — clone a graph with one node's
  instruction swapped (COPRO candidate addressing; changes the digest).
- `eval_set.py` — `build_eval_split()` pinned disjoint train/val/test over the
  seeded shuffle.
- `eval_scores.py` — `combined_reward = score x max(0, 1 - best_compression_ratio)`
  (⚠ direction corrected vs the literal "score x ratio"); `read_candidate_scores`,
  `wait_for_scored` over `dr_dspy_predictions`.
- `study_records.py` — `dr_dspy_studies` + `dr_dspy_study_candidates` DDL + IO.
- `study.py` — pure core: `make_candidate_graphs`, `select_best`,
  `proposer_history`, `history_entry`.
- `copro_proposers.py` — logged `propose_basic` / `propose_given_attempts`
  (`prompt_model != task_model`), reimplementing COPRO's two signatures.
- `humaneval_study_dbos.py` — durable DBOS study workflow
  (`BatchOperationKind.STUDY`, one round per dispatcher batch_step, finalize
  round = test eval); `init-db` / `study` / `worker` CLI.
- `scripts/humaneval_study_v1.py` — `STUDY_PIPELINE` (direct|enc-dec) +
  `STUDY_STRATEGY` (grid|copro).

Edited (minimal): `humaneval_eval_dbos.py` (`build_submit_spec(graphs=...)`,
`build_humaneval_samples_for_task_ids`, `enqueue_round_jobs`),
`batch_operation.py` (`STUDY` enum), `experiment_spec.py`.

## Deviations from the plan (intentional)
- Skipped extracting `run_experiment_submission`; the study round uses the
  lower-level `enqueue_round_jobs` directly inside the DBOS step (avoids a
  nested dispatcher + runtime reconfigure). `submit` CLI untouched.
- Study script is self-contained (duplicates a few instruction strings)
  rather than importing the validated eval script.
- Added an **anchor candidate** (base / best-so-far instruction) to every
  COPRO round so the selected reward can't regress on noise.

## Live smoke (do together, with DATABASE_URL set)
1. `python scripts/humaneval_study_v1.py init-db`
2. Grid loop (cheap, proves the loop):
   `STUDY_PIPELINE=direct STUDY_STRATEGY=grid python scripts/humaneval_study_v1.py study --experiment-name studytest1 --val 4 --test 4 --repetitions 1`
   plus a background `worker --experiment-name studytest1`.
   Inspect `dr_dspy_study_candidates` (per-candidate `val_mean_reward` +
   `val_scores` distribution) and `dr_dspy_predictions` (content-addressed
   reuse across rounds; one experiment_name).
3. Then small COPRO enc-dec:
   `STUDY_PIPELINE=enc-dec STUDY_STRATEGY=copro ... study --experiment-name studytest2 --val 4 --test 4 --repetitions 2 --breadth 2 --depth 2`.
   Verify proposer calls are logged in candidate `provenance`, the test
   finalize round records a `phase=test` history entry.

## #1 risk to watch live
`study_round_step` **blocks (polls) inside the dispatcher workflow** waiting
for the round's predictions to score — while the same worker must also run the
gen/scoring queues. Confirm the worker has enough concurrency that the blocked
study step doesn't starve gen/scoring (else the round never completes). If
awkward, split the wait into its own step/queue. Tunable via
`--wait-interval` / `--wait-timeout` and worker concurrency flags.
