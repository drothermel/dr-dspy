# Behavioral Review Notes

Manual review notes for behavior-changing fixes, robustness defects, validation
changes, cleanup guarantees, and focused regression coverage. These were
extracted from `docs/structural.md` so the structural plan can stay focused on
behavior-preserving refactors.

Source tags:

- **Both reviews**: independently identified in both manual passes.
- **Manual review**: identified in the Codex manual pass in this thread.
- **Spine parallel review**: identified in the parallel-agent pass over the
  spine submodules; each item carrying this tag was re-verified against source
  before inclusion.
- **External cross-package review**: identified in the follow-up external
  findings covering the non-spine submodules and tests.

## Robustness And Correctness Fixes

### B.1 Harden `_internal.Unbatchify`

**Source:** Both reviews.

`dspy/_internal/unbatchify.py` can leave callers blocked forever:

- `__call__` accepts new work after `close()`, but the worker has already joined,
  so `future.result()` never resolves.
- `_worker()` zips batch outputs to futures with `strict=False`; if `batch_fn`
  returns fewer outputs than inputs, remaining futures never resolve.

Fix direction:

- Reject calls after close with a clear exception.
- Validate output count before setting futures.
- Prefer `zip(..., strict=True)` or explicit length checks so mismatch failures
  are immediate and diagnosable.

### B.2 Stop using `None` as the completion sentinel in `run_bounded`

**Source:** Manual review.

`dspy/runtime/async_parallel.py` initializes `results` with `None` and counts
completed work by non-`None` entries. A valid task result of `None` is therefore
indistinguishable from an unfinished or failed task.

Fix direction:

- Track completion separately from result values, or use a private sentinel.
- Keep the public return shape unchanged.
- Preserve current cancellation and error-count behavior.

### B.3 Guard `DummyLM._use_example` against an empty field set

**Source:** Spine parallel review.

`dspy/testing/dummy_lm.py` builds `fields = defaultdict(int)` and then calls
`max(fields.values())`. When `follow_examples=True` and no message matches
`FIELD_HEADER_PATTERN`, `fields` is empty and `max()` raises
`ValueError: max() arg is an empty sequence`, crashing the test double instead of
cleanly reporting no match.

Fix direction:

- Return `None` when `fields` is empty, before calling `max()`.
- Add a regression test exercising `follow_examples=True` with no field headers.

### B.4 Make `LMStreamErrorEvent.to_json()` serializable

**Source:** Spine parallel review.

`dspy/core/types/stream.py` defines `LMStreamErrorEvent` with `error: Exception`
under `arbitrary_types_allowed=True`, but inherits the base
`to_json()` -> `model_dump_json()`. Pydantic cannot serialize a bare `Exception`,
so any consumer calling `.to_json()` on an error event raises a serialization
error instead of producing a payload.

Fix direction:

- Override `to_json` or add a field serializer so error events emit a stable
  shape such as `{"type": "error", "error": str(self.error)}`.
- Keep the other stream-event payloads unchanged.

### B.5 Stop `Completions` from aliasing the caller's dict

**Source:** Spine parallel review.

`dspy/primitives/prediction.py` `Completions.__init__` builds fresh lists for the
`list[dict]` input but stores the caller's dict by reference for the
`dict[str, list]` input. The two input paths therefore have different aliasing
semantics, and the validated equal-length invariant can be silently broken by
later in-place mutation.

Fix direction:

- Shallow-copy on the dict path: `{k: list(v) for k, v in list_or_dict.items()}`,
  after confirming no caller relies on the aliasing.
- Add a test that mutating the original dict after construction does not affect
  `Completions`.

### B.6 Harden `inspect_call_log` against content-less assistant messages

**Source:** Spine parallel review.

`dspy/runtime/inspect_call_log.py` mixes required-key indexing (`msg["content"]`,
`msg["role"]`) with defensive `.get()` elsewhere. An OpenAI-format assistant
message carrying only `tool_calls` has no `content`, so `msg["content"]` raises
`KeyError` and aborts the entire debug print.

Fix direction:

- Use `.get("content")` and treat absent content as empty.
- Keep the rest of the pretty-printer behavior unchanged.

### B.7 Clarify `Module.load_state` atomicity

**Source:** Spine parallel review.

`dspy/primitives/module.py` `load_state` calls `_apply(self.deepcopy())` and
discards the result before `_apply(self)`. Applying to a throwaway deep copy
first means a missing `state[name]` key raises before `self` is mutated, giving
all-or-nothing semantics. The intent is undocumented and the deep copy of a full
program is not free.

Fix direction:

- If the atomicity guarantee is intended, add a one-line comment naming it.
- If it is not intended, drop the `deepcopy` pass.

### B.8 Replace recursive polling in GRPO

**Source:** Manual review.

`dspy/teleprompt/grpo.py` implements `_wait_until` with recursive async calls.
A long-running GRPO job can build an unbounded call stack while waiting for an
available training batch.

Fix direction:

- Replace recursion with a simple `while not predicate(): await sleep(...)` loop.
- Keep the same poll interval and completion behavior.

### B.9 Replace GRPO runtime `assert` validation with explicit errors

**Source:** Manual review.

`dspy/teleprompt/grpo.py` uses `assert` for constructor checks, compile-time
input validation, and runtime invariants. These checks disappear under optimized
Python, which would remove user-facing validation and internal safety checks.

Fix direction:

- Convert user/config validation to explicit `ValueError` or `TypeError`.
- Convert data-shape and job-state checks to explicit runtime exceptions with
  the existing messages.

### B.10 Align JSON adapter parsing with the documented exact-field contract

**Source:** Manual review.

`dspy/adapters/utils/parse.py` documents that adapter parse results must match
the task spec's output field keys exactly. `dspy/adapters/json_adapter.py`,
however, filters unexpected fields before validation, so extra LM output keys are
silently discarded instead of reported as schema drift.

Fix direction:

- Decide whether unexpected JSON fields are accepted or rejected.
- If rejected, validate before filtering and keep `validate_parsed_fields` as the
  single contract check.
- If accepted, update the parse helper documentation.

### B.11 Narrow transient sampling exception handling

**Source:** Manual review.

`dspy/predict/sampling.py` catches `BaseException` and then filters retryability.
That catches cancellation and interpreter-control exceptions before re-raising
them.

Fix direction:

- Catch `Exception` instead of `BaseException`.
- Keep `is_transient_sampling_error` as the retry classifier.
- Preserve `SamplingExhaustedError` chaining from the last transient exception.

### B.12 Fix local finetune server polling and cleanup

**Source:** Manual review.

`dspy/integrations/finetune/local.py` mixes provider behavior, subprocess launch,
server readiness polling, log buffering, tokenizer setup, and training. Its
`wait_for_server` timeout check only runs after an HTTP response, so repeated
connection failures can wait indefinitely.

Fix direction:

- Move launch/process/log handling into a focused local-server helper.
- Make the timeout check unconditional in the polling loop.
- Keep provider methods as thin orchestration around launch, kill, and training
  helpers.

### B.13 Guarantee BetterTogether LM cleanup

**Source:** Manual review.

`dspy/teleprompt/bettertogether.py` launches LMs before baseline evaluation and
kills them after the strategy loop. If baseline evaluation or early setup fails,
launched local LMs may not be killed.

Fix direction:

- Wrap the launch/evaluation/strategy block in `try/finally`.
- Keep the existing `flag_lms_launched` behavior, but ensure cleanup runs for
  baseline failures as well as later optimizer failures.

### B.14 Add a typed response boundary for Databricks retrieval

**Source:** Manual review.

`dspy/integrations/retrieval/databricks.py` has SDK and requests paths that both
return raw response dictionaries. The requests path manually reads JSON without
`raise_for_status`, and response validation is spread through the query method.

Fix direction:

- Add a small parser/helper for the expected Databricks vector-search response
  shape.
- Call `raise_for_status()` in the requests path before parsing JSON.
- Share column validation and passage conversion between SDK and requests paths.

## Behavior-Changing Findings To Track

### C.1 Bootstrap labeled-demo sampling can shrink the shared pool

**Source:** External cross-package review.

`BootstrapFewShot._train` reportedly reassigns the shared labeled-demo pool
inside the per-predictor loop. Later predictors then sample from a permanently
smaller pool.

Fix direction:

- Sample into a fresh local variable and leave the original labeled-demo pool
  intact.
- Add coverage for multi-predictor demo allocation.

### C.2 Sampling failure budget can count attempts instead of failures

**Source:** External cross-package review.

`predict/sampling.py` reportedly compares the attempt index against the
decrementing failure budget. With successes interspersed, failure exhaustion can
depend on where failures occur, not how many failures happened.

Fix direction:

- Track `failures_seen` separately from attempt index.
- Keep retryable-error classification separate from budget accounting.

### C.3 LM truncation warning can crash when kwargs are absent

**Source:** External cross-package review.

`LM._check_truncation` reportedly indexes `self.kwargs["max_tokens"]` and
`self.kwargs["temperature"]`, but those keys are only present when configured.
A truncated response from an LM without explicit values can raise `KeyError`
inside warning construction.

Fix direction:

- Use `.get(...)` or a typed config snapshot for warning text.
- Add a test for truncation warnings with default LM kwargs.

### C.4 XML adapter JSON repair flag can be ignored

**Source:** External cross-package review.

`XMLAdapter.parse` reportedly calls `parse_output_field` without forwarding
`repair=self.allow_json_repair`, unlike ChatAdapter and JSONAdapter.

Fix direction:

- Forward the repair flag consistently.
- Add a focused XML parse test with repair enabled.

### C.5 Avatar optimizer can accept an unevaluated instruction

**Source:** External cross-package review.

`avatar_optimizer.py` reportedly commits a candidate instruction based on the
score of the previous instruction. The candidate should be evaluated before it is
accepted.

Fix direction:

- Evaluate the candidate instruction before accept/reject.
- Preserve current scoring and selection shape after the evaluation point moves.

### C.6 SIMBA helpers can mutate shared bucket or trace inputs

**Source:** External cross-package review.

`simba_utils.py` helpers reportedly mutate bucket/trace input dictionaries in
place and overwrite numeric scores with `"N/A"`, which can leak truncation or
corruption into other strategies and logs.

Fix direction:

- Operate on copies before truncation or redaction.
- Keep logged shapes stable except for avoiding shared-reference mutation.

### C.7 Finetune job futures can return exceptions as successful results

**Source:** External cross-package review.

`clients/finetune/lm.py` reportedly stores exceptions with `Future.set_result`
instead of `Future.set_exception`, forcing every caller to manually check for an
exception result.

Fix direction:

- Use `set_exception` for failed jobs.
- Audit callers that currently inspect `Exception` return values.

### C.8 Local finetune launch mutates LM endpoint state without restoring it

**Source:** External cross-package review.

`integrations/finetune/local.py` reportedly mutates `lm.kwargs` and provider
options to target the local server, but `kill()` does not restore the previous
endpoint. The LM can remain pointed at a dead local server.

Fix direction:

- Snapshot endpoint-related state before launch.
- Restore it on kill/cleanup, including failure paths.

### C.9 Databricks retrieval score handling is inconsistent

**Source:** External cross-package review.

Databricks retrieval reportedly sorts with `row["score"]` while passage
conversion uses `.get("score")`. Missing score values can raise during sorting
even though later code treats score as optional.

Fix direction:

- Either validate score as required at the boundary or sort with an explicit
  default.
- Keep SDK and requests paths aligned.

### C.10 Citation URL is silently dropped

**Source:** External cross-package review.

`adapters/types/citation.py` reportedly copies a URL into serialized data while
the `Citation` model has no `url` field, so Pydantic ignores it.

Fix direction:

- Add a `url` field if URL is a supported contract.
- Otherwise remove the copy path and document that URL is not part of the model.

### C.11 Additional targeted defect checks

**Source:** External cross-package review.

The same pass flagged a few smaller correctness checks worth verifying before
large refactors:

- GEPA adapter guard logic that checks `hasattr(x, "__class__")`, which is always
  true and therefore likely dead.
- COPRO stats paths that can call `max([])` or divide by zero when latest scores
  are empty.
- `Refine` predictor-name mapping keyed by `TaskSpec`, which can collapse when
  two predictors share the same spec.
- LiteLLM transport calls that configure retries/backoff but no default timeout.
- Persistence state loads that mutate `self` before validating required keys or
  surface opaque `KeyError`s for missing metadata.
- Broad transient-error classifiers that treat generic `ValueError` and
  `RuntimeError` as retryable or demo-shrinkable.
- Unseeded global randomness in ensemble/evaluation helpers.

## Verification Gaps

Add focused tests before or while fixing these behaviors:

- `Unbatchify` close-after-call and output-count mismatch behavior.
- `run_bounded` with valid `None` results.
- `LMStream` and `AsyncLMStream` error/result serialization behavior.
- GRPO validation behavior under optimized Python.
- Adapter JSON parsing with unexpected output fields.
- Local finetune server timeout and cleanup behavior.
- Persistence round trips with missing and legacy keys.
- Sampling failure budgets with interspersed successes and transient failures.
- Multi-predictor bootstrap demo allocation from a shared labeled-demo pool.
- SIMBA helper behavior around copied vs shared bucket/trace inputs.
- Code-fence parser parity if multiple parsers remain.

## Recommended Sequence

1. Harden `Unbatchify` and `run_bounded`.
2. Fix the concrete serialization, warning, and missing-key crashes.
3. Tighten sampling, retry, and transient-error semantics.
4. Fix GRPO polling and runtime validation.
5. Align adapter JSON/XML parsing contracts.
6. Fix local finetune and BetterTogether cleanup guarantees.
7. Add the targeted regression tests listed above.
