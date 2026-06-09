# Behavioral Review: Prioritized Change Chunks

Manual review notes for behavior-changing fixes, robustness defects, validation
changes, cleanup guarantees, and focused regression coverage. These were
extracted from `docs/structural.md` so the structural plan can stay focused on
behavior-preserving refactors.

Before implementing each chunk, re-verify the finding against the current source
tree; some items may already have been fixed during structural work or follow-up
changes.

Source tags:

- **Both reviews**: independently identified in both manual passes.
- **Manual review**: identified in the Codex manual pass in this thread.
- **Spine parallel review**: identified in the parallel-agent pass over the
  spine submodules; each item carrying this tag was re-verified against source
  before inclusion.
- **External cross-package review**: identified in the follow-up external
  findings covering the non-spine submodules and tests.

## P0: Eliminate Hangs, Lost Exceptions, and Control-Flow Hazards

These are the highest-priority behavioral fixes because they can leave callers
blocked, hide failed work, catch cancellation, or leak launched resources.

### P0.1 Harden `Unbatchify` completion semantics

**Sources:** Both reviews; Manual review.

Problem:

- `dspy/_internal/unbatchify.py` can leave callers blocked forever:
  - `__call__` accepts new work after `close()`, but the worker has already
    joined, so `future.result()` never resolves.
  - `_worker()` zips batch outputs to futures with `strict=False`; if `batch_fn`
    returns fewer outputs than inputs, remaining futures never resolve.
- `close()` now drains the input queue and fails pending futures with
  `RuntimeError("Unbatchify is closed")`, but post-close submissions are still
  not rejected at the call site.

Target shape:

- Reject `Unbatchify` calls after close with a clear exception.
- Validate batch output count before setting futures.
- Prefer `zip(..., strict=True)` or explicit length checks so mismatch failures
  are immediate and diagnosable.

Details to preserve:

- Add focused regression coverage for close-after-call and output-count mismatch.

### P0.2 Keep runtime retry and sampling control flow explicit

**Sources:** Manual review; External cross-package review.

Problem:

- `dspy/predict/sampling.py` catches `BaseException` and then filters
  retryability, catching cancellation and interpreter-control exceptions before
  re-raising them.
- Sampling failure-budget accounting compares the attempt index against the
  decrementing failure budget (`if idx > failures_remaining`). With successes
  interspersed, failure exhaustion can depend on where failures occur rather
  than how many failures happened.
- Broad transient-error classifiers treat generic `ValueError` and
  `RuntimeError` as retryable or demo-shrinkable.

Target shape:

- Catch `Exception` instead of `BaseException`.
- Keep retryable-error classification separate from failure-budget accounting.
- Track `failures_seen` separately from attempt index.
- Preserve `SamplingExhaustedError` chaining from the last transient exception.
- Narrow transient classifiers to failures that are plausibly transient in the
  relevant boundary.

Details to preserve:

- Cancellation and interpreter-control exceptions should propagate normally.
- Add coverage for interspersed successes and transient failures.

### P0.3 Guarantee finetune launch, polling, futures, and cleanup behavior

**Sources:** Manual review; External cross-package review.

Problem:

- `dspy/integrations/finetune/local.py` mixes provider behavior, subprocess
  launch, server readiness polling, log buffering, tokenizer setup, and
  training.
- Its `wait_for_server` timeout check only runs after an HTTP response, so
  repeated connection failures can wait indefinitely.
- Local finetune launch mutates `lm.kwargs` and provider options to target the
  local server, but `kill()` does not restore the previous endpoint. The LM can
  remain pointed at a dead local server.
- `dspy/clients/finetune/service.py` stores exceptions with
  `Future.set_result` instead of `Future.set_exception`, forcing every caller to
  manually check for an exception result.

Target shape:

- Move launch, process, log, and polling behavior into a focused local-server
  helper.
- Make the timeout check unconditional in the polling loop.
- Snapshot endpoint-related LM state before launch.
- Restore endpoint state on kill and cleanup, including failure paths.
- Use `Future.set_exception` for failed finetune jobs.

Details to preserve:

- Keep provider methods as thin orchestration around launch, kill, and training
  helpers.
- Audit callers that currently inspect `Exception` return values.
- Add tests for server timeout, endpoint restoration, failed launch cleanup, and
  failed job futures.

### P0.4 Ensure launched BetterTogether LMs are always cleaned up

**Source:** Manual review.

Problem:

- `dspy/teleprompt/bettertogether.py` launches LMs before baseline evaluation
  and kills them after the strategy loop.
- If baseline evaluation or early setup fails, launched local LMs may not be
  killed.

Target shape:

- Wrap the launch, baseline evaluation, and strategy block in `try/finally`.
- Keep the existing `flag_lms_launched` behavior.

Details to preserve:

- Cleanup should run for baseline failures as well as later optimizer failures.
- Add a regression test with a failing baseline evaluation and a launched local
  LM double.

## P1: Enforce Boundary Schemas and Serialization Contracts

These fixes should follow P0 because they make parser, stream, retrieval, and
persistence behavior fail fast instead of silently accepting drift or crashing in
secondary code paths.

### P1.1 Align adapter parsing with exact field and repair contracts

**Sources:** Manual review; External cross-package review.

Problem:

- `dspy/adapters/utils/parse.py` documents that adapter parse results must match
  the task spec's output field keys exactly.
- `dspy/adapters/json_adapter.py` filters unexpected fields before validation,
  so extra LM output keys are silently discarded instead of reported as schema
  drift.
- `XMLAdapter.parse` calls `parse_output_field` without forwarding
  `repair=self.allow_json_repair`, unlike ChatAdapter and JSONAdapter.

Target shape:

- Decide whether unexpected JSON fields are accepted or rejected.
- If rejected, validate before filtering and keep `validate_parsed_fields` as
  the single contract check.
- If accepted, update the parse helper documentation.
- Forward the XML repair flag consistently.

Details to preserve:

- Keep existing successful parse behavior for exact fields.
- Add focused tests for unexpected JSON output fields and XML parsing with
  repair enabled.

### P1.2 Make stream error events serialize predictably

**Sources:** Spine parallel review.

Problem:

- `dspy/core/types/stream_events.py` defines `LMStreamErrorEvent` with
  `error: Exception` under `arbitrary_types_allowed=True`, but inherits the base
  `to_json()` -> `model_dump_json()`. Pydantic cannot serialize a bare
  `Exception`, so consumers calling `.to_json()` on an error event raise a
  serialization error instead of producing a payload.

Target shape:

- Override `LMStreamErrorEvent.to_json` or add a field serializer so error
  events emit a stable JSON-safe shape, such as
  `{"type": "error", "error": str(self.error)}`.

Details to preserve:

- Keep the other stream-event payloads unchanged.
- Add regression tests for `LMStream` and `AsyncLMStream` error/result
  serialization.

### P1.3 Add typed Databricks retrieval response boundaries

**Sources:** Manual review; External cross-package review.

Problem:

- `dspy/integrations/retrieval/databricks.py` has SDK and requests paths that
  both return raw response dictionaries.
- The requests path manually reads JSON without `raise_for_status`.
- Response validation is spread through the query method.
- Databricks retrieval sorts with `row["score"]` while passage conversion uses
  `.get("score")`. Missing score values can raise during sorting even though
  later code treats score as optional.

Target shape:

- Add a small parser/helper for the expected Databricks vector-search response
  shape.
- Call `raise_for_status()` in the requests path before parsing JSON.
- Share column validation and passage conversion between SDK and requests paths.
- Either validate score as required at the boundary or sort with an explicit
  default; keep SDK and requests paths aligned.

Details to preserve:

- Preserve current successful passage conversion shape.
- Add mocked unit tests for response parsing, missing score handling, path
  validation, and request construction.

### P1.4 Resolve citation URL serialization semantics

**Source:** External cross-package review.

Problem:

- `adapters/types/citation.py` copies a URL into serialized data via
  `_citation_part_to_dict`, but the `Citation` model has no `url` field, so the
  value is dropped when citations are reconstructed from dicts.

Target shape:

- Add a `url` field if URL is a supported contract.
- Otherwise remove the copy path and document that URL is not part of the model.

Details to preserve:

- Make the supported serialized citation shape explicit in tests.

## P2: Fix Optimizer Selection, Trace, and Candidate Semantics

These should come after the boundary fixes because optimizer behavior depends on
the runtime, adapter, and persistence contracts being stable.

### P2.1 Preserve shared demo pools during bootstrap

**Source:** External cross-package review.

Problem:

- `BootstrapFewShot._train` in `dspy/teleprompt/bootstrap.py` reassigns the
  shared labeled-demo pool inside the per-predictor loop (`raw_demos =
  rng.sample(raw_demos, sample_size)`). Later predictors then sample from a
  permanently smaller pool.

Target shape:

- Sample into a fresh local variable.
- Leave the original labeled-demo pool intact.

Details to preserve:

- Add coverage for multi-predictor demo allocation from a shared labeled-demo
  pool.

### P2.2 Evaluate optimizer candidates before accepting them

**Source:** External cross-package review.

Problem:

- `avatar_optimizer.py` commits a candidate instruction based on the score of
  the previous instruction. The candidate should be evaluated before it is
  accepted.

Target shape:

- Evaluate the candidate instruction before accept/reject.
- Preserve current scoring and selection shape after the evaluation point moves.

Details to preserve:

- Add a deterministic test where the previous instruction scores well but the
  new candidate should be rejected.

### P2.3 Avoid mutating shared SIMBA logging and trace inputs

**Source:** External cross-package review.

Problem:

- `simba_utils.py` helpers mutate bucket and trace input dictionaries in place.
- The same helpers overwrite numeric scores with `"N/A"`, which can leak
  truncation or corruption into other strategies and logs.

Target shape:

- Operate on copies before truncation or redaction.
- Keep logged shapes stable except for avoiding shared-reference mutation.

Details to preserve:

- Add tests that prove the original bucket and trace inputs remain unchanged
  after logging or display helpers run.

### P2.4 Check optimizer edge cases before broader helper consolidation

**Source:** External cross-package review.

Problem:

- COPRO stats paths can call `max([])` or divide by zero when latest scores are
  empty (`dspy/teleprompt/copro_optimizer.py`).
- `Refine` predictor-name mapping is keyed by `TaskSpec`, which can collapse
  when two predictors share the same spec (`dspy/predict/refine.py`).

Target shape:

- Make COPRO empty-score stats behavior explicit.
- Key Refine predictor-name mapping by predictor identity or another stable
  per-predictor identifier, not shared `TaskSpec` value equality.

Details to preserve:

- Keep current optimizer output shapes where inputs are valid and non-empty.
- Add focused tests for empty COPRO scores and duplicate-task-spec Refine
  predictors.

## P3: Normalize Object Aliasing, Defaults, and Determinism

These are lower priority than hang, cleanup, and schema issues, but they reduce
surprising mutation and environment-dependent behavior.

### P3.1 Stop `Completions` from aliasing caller-owned dictionaries

**Source:** Spine parallel review.

Problem:

- `dspy/primitives/prediction.py` `Completions.__init__` builds fresh lists for
  the `list[dict]` input but stores the caller's dict by reference for the
  `dict[str, list]` input.
- The two input paths therefore have different aliasing semantics, and the
  validated equal-length invariant can be silently broken by later in-place
  mutation.

Target shape:

- Shallow-copy on the dict path:
  `{k: list(v) for k, v in list_or_dict.items()}`.

Details to preserve:

- Confirm no caller intentionally relies on aliasing.
- Add a test that mutating the original dict after construction does not affect
  `Completions`.

### P3.2 Make LM truncation warnings robust with default config

**Source:** External cross-package review.

Problem:

- `LM._check_truncation` indexes `self.kwargs["max_tokens"]` and
  `self.kwargs["temperature"]`, but those keys are only present when
  configured.
- A truncated response from an LM without explicit values can raise `KeyError`
  inside warning construction.

Target shape:

- Use `.get(...)` or a typed config snapshot for warning text.
- Add a test for truncation warnings with default LM kwargs.

Details to preserve:

- Warning text should remain useful when values are configured.

### P3.3 Add explicit network timeouts where retries already exist

**Source:** External cross-package review.

Problem:

- LiteLLM transport calls configure retries and backoff but no default timeout
  (`dspy/clients/lm/transport.py`).

Target shape:

- Set an explicit default timeout at the transport boundary where retries are
  configured.
- Keep caller-provided timeout overrides intact.

Details to preserve:

- Avoid changing provider-specific timeout semantics without focused tests or
  documentation.

### P3.4 Make randomness contracts explicit

**Source:** External cross-package review.

Problem:

- Ensemble and evaluation helpers use unseeded global randomness.

Target shape:

- Thread an explicit seed or random generator through the helper where
  deterministic behavior is part of the contract.
- Document any helper that intentionally remains nondeterministic.

Details to preserve:

- Preserve current default behavior unless the helper already accepts seed-like
  configuration.

## Verification Gaps

Add focused tests before or while fixing the relevant behavior:

- `Unbatchify` close-after-call and output-count mismatch behavior.
- `LMStream` and `AsyncLMStream` error/result serialization behavior.
- Adapter JSON parsing with unexpected output fields.
- XML adapter JSON repair behavior.
- Citation serialization with or without URL support.
- Databricks retrieval response parsing, `raise_for_status`, and score handling.
- Local finetune server timeout, endpoint restoration, and cleanup behavior.
- Finetune job futures that fail.
- BetterTogether cleanup when baseline evaluation fails.
- Sampling failure budgets with interspersed successes and transient failures.
- Multi-predictor bootstrap demo allocation from a shared labeled-demo pool.
- Avatar candidate accept/reject after evaluating the candidate.
- SIMBA helper behavior around copied vs shared bucket and trace inputs.
- COPRO empty-score stats and duplicate-task-spec Refine predictors.
- `Completions` aliasing after caller-owned dict mutation.
- LM truncation warnings with default LM kwargs.
- Code-fence parser parity if multiple parsers remain.
