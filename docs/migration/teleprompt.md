# Teleprompt migration guide

Teleprompters (optimizers) now share one compile contract: typed `*CompileParams`, a registry-backed `Teleprompter` protocol, and a `CompileResult` return type. Candidate demo generation for search optimizers uses a shared seed ladder instead of magic negative integer seeds.

Public imports: `from dspy.teleprompt import ...` (see curated exports in `dspy/teleprompt/__init__.py`).

## Compile return type

| Old | New |
| --- | --- |
| `program = await teleprompter.compile(student, params=..., run=run)` | `result = await teleprompter.compile(student, params=..., run=run)` |
| Ad-hoc attrs on `Module` (`candidate_programs`, `total_calls`, `trial_logs`, `score`, `flag_compilation_error_occurred`, `results_best`, `simba_idx`, …) | `result.program`, `result.candidates`, `result.stats` |

```python
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotCompileParams, CompileResult

result: CompileResult = await teleprompter.compile(
    student,
    params=BootstrapFewShotCompileParams(trainset=trainset),
    run=run,
)
program = result.program
best_score = result.stats.best_score
for candidate in result.candidates:
    print(candidate.label, candidate.score, candidate.seed)
```

`CompileResult.with_compiled_program(program)` sets `program._compiled = True` before return. The `_compiled` flag on `Module` is unchanged — it marks opaque subgraphs for transparency, not optimizer metadata.

### `CompileResult` fields

| Field | Purpose |
| --- | --- |
| `program` | Best (or final) compiled module |
| `candidates` | `list[ProgramCandidate]` — scored alternatives with optional `label`, `seed`, `subscores`, `full_eval` |
| `stats` | `CompileStats` — `metric_calls`, `prompt_model_calls`, `error_occurred`, `best_score`, `trial_logs`, `copro_depth_stats` |

## Compile params

Every compile-capable teleprompter has a matching `*CompileParams` model in `dspy.teleprompt.compile_params`. Pass params via the nested `params=` keyword; keep `run=` top-level.

```python
await bootstrap.compile(
    student,
    params=BootstrapFewShotCompileParams(trainset=trainset, teacher=teacher),
    run=run,
)
```

### Removed aliases / passthrough types

| Removed | Use instead |
| --- | --- |
| `TelepromptOptunaCompileParams` | `BootstrapOptunaCompileParams` |
| `BootstrapFinetuneCompileParams` (empty model) | `BootstrapFewShotCompileParams` |
| `PassthroughCompileParams` (BetterTogether) | Per-optimizer `*CompileParams` validated by registry |

## BetterTogether

Strategy is a list of optimizer keys, not a `"p -> w -> p"` string:

```python
from dspy.teleprompt import BetterTogether, BetterTogetherCompileParams, RandomSearchCompileParams

bt = BetterTogether(metric=my_metric)
result = await bt.compile(
    student,
    params=BetterTogetherCompileParams(
        trainset=trainset,
        valset=valset,
        strategy=["p", "w"],
        optimizer_compile_args={
            "p": RandomSearchCompileParams(trainset=trainset, restrict=[0, 1]),
        },
    ),
    run=run,
)
```

- Default strategy: `["p", "w", "p"]` (prompt → weights → prompt).
- Sub-optimizers must be registered with `@register_teleprompter(params=...)`. Unknown optimizers fail at `BetterTogether` init.
- When the default prompt optimizer is `BootstrapFewShotWithRandomSearch`, BetterTogether passes `include_baselines=False` so the baseline is not re-evaluated.
- On partial failure: `result.stats.error_occurred` replaces `flag_compilation_error_occurred`; best program is still in `result.program`.

## Candidate seed ladder

Random-search and MIPRO bootstrap share `dspy.teleprompt.candidate_ladder`:

| Old | New |
| --- | --- |
| Magic seeds `-3`, `-2`, `-1` | Typed `CandidateSeed` (`BaselineSeed`, `LabeledFewShotSeed`, `BootstrapSeed`, `RandomizedBootstrapSeed`) |
| `num_candidate_programs=16` → 19 total (baselines + random) | `num_random_candidates=16` + additive baselines via `CandidateLadderConfig.include_*` |
| `demo_sets.create_n_fewshot_demo_sets` | `generate_demo_candidate_sets` |

```python
from dspy.teleprompt import CandidateLadderConfig, generate_demo_candidate_sets

config = CandidateLadderConfig(
    num_random=16,
    max_labeled_demos=4,
    max_bootstrapped_demos=4,
    include_baseline=True,
)
demo_candidates = await generate_demo_candidate_sets(
    student=student,
    config=config,
    trainset=trainset,
    metric=my_metric,
    run=run,
)
```

`RandomSearchCompileParams.include_baselines=False` skips baseline seeds when a parent optimizer already evaluated the baseline.

## Adding a new optimizer

1. Define `MyOptimizerCompileParams(BaseModel)` in `compile_params.py`.
2. Implement `async def compile(self, student, *, params: BaseModel, run: RunContext) -> CompileResult`.
3. Decorate with `@register_teleprompter(params=MyOptimizerCompileParams)`.
4. Export the class and params from `dspy/teleprompt/__init__.py`.
5. Return `CompileResult.with_compiled_program(...)` when the student should be marked compiled.

Nested compiles (bootstrap → labeled few-shot, optuna → bootstrap, knn → bootstrap) unwrap the inner `.program`:

```python
inner = await teleprompter.compile(student, params=..., run=run)
student = inner.program
```

## Evaluation helpers

```python
from dspy.teleprompt import make_optimizer_evaluator, resolve_max_errors, run_program_with_trace, trace_to_demos

evaluate = make_optimizer_evaluator(
    run,
    devset=valset,
    metric=my_metric,
    max_concurrency=8,
    max_errors=resolve_max_errors(None, run),
)
prediction, trace = await run_program_with_trace(program, example, run)
demos_by_predictor = trace_to_demos(trace, predictor2name)
```

## Optimizer metric contract

Teleprompters accept an `OptimizerMetric` (`from dspy.teleprompt import OptimizerMetric`): a sync or async callable `(example, prediction, trace) -> bool | float | Prediction`, or a `Module` metric. `invoke_metric` (in `dspy.evaluate.metric_invoke`) normalizes scores to `[0, 1]`. GEPA uses the separate five-argument `GEPAFeedbackMetric` protocol.

## Resolving optimizer LMs

Use `resolve_optimizer_lm(lm=None, run=run)` from `dspy.teleprompt.task_spec_context` (replacing `get_prompt_model`). When `lm` is `None`, the active run default (`run.lm`) is used.

## Contract tests

`tests/teleprompt/` encodes optimizer contracts: metric dispatch (`test_teleprompt_metrics.py`), LM fallback (`test_lm_resolver.py`), structural zip safety (`test_structural_zip.py`), infer-rules error handling (`test_infer_rules_errors.py`), and SIMBA resampling immutability (`test_simba_utils.py`).

## Breaking changes summary

| Area | Change |
| --- | --- |
| Return type | `compile(...) -> CompileResult` (not `Module`) |
| Module metadata | Removed optimizer attrs; use `CompileResult.candidates` / `.stats` |
| BetterTogether strategy | `list[str]` instead of `"p -> w -> p"` string |
| Random search count | `num_random_candidates` (randomized seeds only); baselines are additive |
| Registry | All compile-capable teleprompters must register params; BT rejects unregistered sub-optimizers |
| Deleted params | `TelepromptOptunaCompileParams`, `BootstrapFinetuneCompileParams`, `PassthroughCompileParams` |
| Deleted module | `demo_sets.py` → `candidate_ladder.py` |
