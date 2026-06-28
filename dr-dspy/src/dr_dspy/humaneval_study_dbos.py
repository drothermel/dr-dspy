"""Outer optimizer study as a durable DBOS workflow.

A *study* runs coordinate ascent over one node's instruction (the encoder
for enc-dec; the solver for direct) under a single ``experiment_name`` and
a pinned eval-set split. Each round proposes candidate instructions
(``grid``: a fixed list; ``copro``: logged proposers), submits their
graphs over the val set by reusing the eval pipeline's generation/scoring
queues, waits for scoring, reads ``correctness x compression`` rewards,
and selects the best. A final round evaluates the overall best on the
held-out test set.

The round loop rides the generic ``batch_operation`` dispatcher
(``BatchOperationKind.STUDY``) so it is resumable at round granularity:
``next_offset`` is the round index, and per-round candidates/results are
persisted in ``study_records`` (so a recovered round reuses its already
proposed candidates instead of re-proposing). The optimizer *logic* lives
in the pure ``study`` module; this file is the DBOS + DB orchestration
shell.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

import typer
from dbos import DBOS
from pydantic import BaseModel, ConfigDict, StrictStr

from dr_dspy import batch_operation as shared_batch
from dr_dspy import copro_proposers, eval_scores, study_records
from dr_dspy import dbos_runtime as shared_dbos
from dr_dspy import eval_set as shared_eval_set
from dr_dspy import humaneval_dbos_flow as shared_flow
from dr_dspy import study as study_core
from dr_dspy.experiment_spec import GraphSpec, dimensions_digest
from dr_dspy.humaneval_eval_dbos import _BACKEND as EVAL_BACKEND
from dr_dspy.humaneval_eval_dbos import (
    DEFAULT_GENERATION_CONCURRENCY,
    DEFAULT_SCORING_CONCURRENCY,
    DEFAULT_WORKER_MONITOR_INTERVAL_SECONDS,
    DEFAULT_WORKER_MONITOR_SUMMARY_INTERVAL_SECONDS,
    DEFAULT_WORKER_OPEN_FILE_LIMIT,
    PREDICTION_TABLE_NAME,
    EvalHumanEvalExperimentConfig,
    build_humaneval_samples_for_task_ids,
    build_submit_spec,
    common_config,
    configure_experiment,
    enqueue_round_jobs,
    experiment_config,
    upsert_experiment,
)
from dr_dspy.humaneval_eval_dbos import (
    _configure_dbos_runtime as configure_dbos_runtime,
)
from dr_dspy.humaneval_eval_dbos import (
    _create_eval_schema as create_eval_schema,
)
from dr_dspy.humaneval_eval_dbos import _operator_log as operator_log
from dr_dspy.humaneval_eval_dbos import (
    _resolve_operation_log_path as resolve_operation_log_path,
)
from dr_dspy.lm_utils import ModelConfig

STUDY_LOGGER_NAME = "dr_dspy.humaneval_eval_v1_study"
DATABASE_URL_ENV = "DATABASE_URL"
GRID_STRATEGY = "grid"
COPRO_STRATEGY = "copro"


# --- specs ------------------------------------------------------------


class StudySpec(BaseModel):
    """Stored as the STUDY operation spec (drives every round)."""

    model_config = ConfigDict(extra="forbid")

    study_id: StrictStr
    experiment_name: StrictStr
    strategy: StrictStr
    node_id: StrictStr
    base_graph: GraphSpec
    base_instruction: StrictStr
    val_ids: list[str]
    test_ids: list[str]
    repetitions: int
    score_timeout: float
    seed: int
    depth: int
    breadth: int
    grid_instructions: list[str] = []
    prompt_model: ModelConfig | None = None
    proposal_temperature: float = copro_proposers.DEFAULT_PROPOSAL_TEMPERATURE
    proposal_max_tokens: int = copro_proposers.DEFAULT_PROPOSAL_MAX_TOKENS
    wait_interval_seconds: float = 5.0
    wait_timeout_seconds: float = 3600.0


class StudyPlanConfig(BaseModel):
    """Script-level study shape (which node + base graph to optimize)."""

    model_config = ConfigDict(extra="forbid")

    node_id: StrictStr
    base_graph: GraphSpec
    base_instruction: StrictStr
    default_strategy: StrictStr = GRID_STRATEGY
    grid_instructions: tuple[str, ...] = ()
    prompt_model: ModelConfig | None = None
    default_breadth: int = 4
    default_depth: int = 3
    default_repetitions: int = 1
    default_train: int = 0
    default_val: int = 16
    default_test: int = 16


_PLAN_CONFIG: StudyPlanConfig | None = None


def configure_study_plan(plan: StudyPlanConfig) -> None:
    global _PLAN_CONFIG
    _PLAN_CONFIG = plan


def study_plan() -> StudyPlanConfig:
    if _PLAN_CONFIG is None:
        raise RuntimeError(
            "study plan is not configured; call create_study_app(...) first."
        )
    return _PLAN_CONFIG


# --- logging ----------------------------------------------------------


def _configure_study_file_logging(log_file: Path) -> None:
    shared_batch.configure_operation_file_logging(
        log_file, logger_name=STUDY_LOGGER_NAME
    )


def _emit_study_log(event: str, payload: Mapping[str, Any]) -> None:
    shared_batch.emit_operation_log(
        event, payload, logger_name=STUDY_LOGGER_NAME
    )


# --- candidate proposal (recovery-safe) -------------------------------


def _dedup_pairs(
    pairs: Sequence[tuple[str, dict[str, Any]]],
) -> list[tuple[str, dict[str, Any]]]:
    seen: set[str] = set()
    out: list[tuple[str, dict[str, Any]]] = []
    for instruction, provenance in pairs:
        if instruction and instruction not in seen:
            seen.add(instruction)
            out.append((instruction, provenance))
    return out


def _propose_instruction_pairs(
    database_url: str, spec: StudySpec, round_index: int
) -> list[tuple[str, dict[str, Any]]]:
    if spec.strategy == GRID_STRATEGY:
        return _dedup_pairs(
            [
                (instruction, {"strategy": GRID_STRATEGY})
                for instruction in spec.grid_instructions
            ]
        )
    if spec.prompt_model is None:
        raise ValueError("copro study requires a prompt_model")
    if round_index == 0:
        proposals = copro_proposers.propose_basic(
            prompt_model=spec.prompt_model,
            basic_instruction=spec.base_instruction,
            breadth=spec.breadth,
            temperature=spec.proposal_temperature,
            max_completion_tokens=spec.proposal_max_tokens,
        )
        anchor = (spec.base_instruction, {"anchor": True})
    else:
        study_row = study_records.read_study(database_url, spec.study_id)
        history = study_row.history if study_row else []
        attempts = study_core.proposer_history(history)
        proposals = copro_proposers.propose_given_attempts(
            prompt_model=spec.prompt_model,
            history=attempts,
            breadth=spec.breadth,
            temperature=spec.proposal_temperature,
            max_completion_tokens=spec.proposal_max_tokens,
        )
        best_so_far = attempts[0][0] if attempts else spec.base_instruction
        anchor = (best_so_far, {"anchor": True})
    pairs = [anchor] + [
        (
            proposal.instruction,
            {
                "strategy": COPRO_STRATEGY,
                "round_index": round_index,
                "usage": proposal.usage,
                "cost": proposal.cost,
                "raw": proposal.raw,
            },
        )
        for proposal in proposals
    ]
    return _dedup_pairs(pairs)


def _candidates_for_round(
    database_url: str, spec: StudySpec, round_index: int
) -> list[study_core.CandidateSpec]:
    existing = study_records.read_candidates(
        database_url, spec.study_id, round_index=round_index
    )
    if existing:
        return [
            study_core.CandidateSpec(
                instruction=row.instruction,
                graph=GraphSpec.model_validate(row.graph),
                dimensions_digest=row.dimensions_digest,
                provenance=row.provenance,
            )
            for row in existing
        ]
    pairs = _propose_instruction_pairs(database_url, spec, round_index)
    if not pairs:
        raise ValueError(f"no candidate instructions for round {round_index}")
    instructions = [instruction for instruction, _ in pairs]
    provenances = [provenance for _, provenance in pairs]
    candidates = study_core.make_candidate_graphs(
        spec.base_graph, spec.node_id, instructions, provenances=provenances
    )
    for index, candidate in enumerate(candidates):
        study_records.upsert_candidate(
            database_url,
            study_records.CandidateRow(
                study_id=spec.study_id,
                round_index=round_index,
                candidate_index=index,
                instruction=candidate.instruction,
                dimensions_digest=candidate.dimensions_digest,
                graph=candidate.graph.model_dump(mode="json"),
                provenance=candidate.provenance,
            ),
        )
    return candidates


# --- round execution --------------------------------------------------


def _submit_and_wait(
    database_url: str,
    spec: StudySpec,
    *,
    graphs: Sequence[GraphSpec],
    task_ids: Sequence[str],
    submission_id: str,
) -> tuple[int, int, dict[str, eval_scores.CandidateScores]]:
    submit_spec = build_submit_spec(
        experiment_name=spec.experiment_name,
        seed=spec.seed,
        sample_count=len(task_ids),
        repetitions=spec.repetitions,
        score_timeout=spec.score_timeout,
        graphs=list(graphs),
    )
    samples = build_humaneval_samples_for_task_ids(task_ids=list(task_ids))
    inserted, enqueued = enqueue_round_jobs(
        database_url,
        submit_spec=submit_spec,
        submission_id=submission_id,
        samples=samples,
        score_timeout=spec.score_timeout,
    )
    digests = [dimensions_digest(graph) for graph in graphs]
    expected = len(digests) * len(task_ids) * spec.repetitions
    eval_scores.wait_for_scored(
        database_url,
        experiment_name=spec.experiment_name,
        dimensions_digests=digests,
        task_ids=list(task_ids),
        expected_count=expected,
        interval_seconds=spec.wait_interval_seconds,
        timeout_seconds=spec.wait_timeout_seconds,
    )
    scores = eval_scores.read_candidate_scores(
        database_url,
        experiment_name=spec.experiment_name,
        dimensions_digests=digests,
        task_ids=list(task_ids),
    )
    return inserted, enqueued, scores


def _optimize_round(
    database_url: str, spec: StudySpec, round_index: int
) -> shared_batch.BatchOperationResult:
    candidates = _candidates_for_round(database_url, spec, round_index)
    inserted, enqueued, scores = _submit_and_wait(
        database_url,
        spec,
        graphs=[candidate.graph for candidate in candidates],
        task_ids=spec.val_ids,
        submission_id=f"{spec.study_id}:r{round_index}",
    )
    scored = [
        study_core.ScoredCandidate(
            candidate=candidate,
            scores=scores[candidate.dimensions_digest],
        )
        for candidate in candidates
    ]
    best = study_core.select_best(scored)
    for index, item in enumerate(scored):
        study_records.upsert_candidate(
            database_url,
            study_records.CandidateRow(
                study_id=spec.study_id,
                round_index=round_index,
                candidate_index=index,
                instruction=item.candidate.instruction,
                dimensions_digest=item.candidate.dimensions_digest,
                graph=item.candidate.graph.model_dump(mode="json"),
                provenance=item.candidate.provenance,
                val_mean_reward=item.mean_reward(),
                val_coverage=item.scores.coverage(),
                val_scores={"distribution": item.scores.reward_distribution()},
                selected=(
                    item.candidate.dimensions_digest
                    == best.candidate.dimensions_digest
                ),
            ),
        )
    study_row = study_records.read_study(database_url, spec.study_id)
    history = list(study_row.history) if study_row else []
    history.append(study_core.history_entry(round_index, best))
    study_records.update_study_state(
        database_url, spec.study_id, status="running", history=history
    )
    _emit_study_log(
        "study_round_completed",
        {
            "study_id": spec.study_id,
            "round_index": round_index,
            "candidates": len(candidates),
            "best_instruction": best.candidate.instruction,
            "best_mean_reward": best.mean_reward(),
        },
    )
    return shared_batch.BatchOperationResult(
        start_offset=round_index,
        next_offset=round_index + 1,
        batch_size=1,
        processed=1,
        inserted=inserted,
        enqueued=enqueued,
        counters={"rounds": 1},
    )


def _select_overall_best(
    candidates: Sequence[study_records.CandidateRow], *, depth: int
) -> study_records.CandidateRow | None:
    scored = [
        candidate
        for candidate in candidates
        if candidate.round_index < depth
        and candidate.val_mean_reward is not None
    ]
    if not scored:
        return None
    return min(
        scored,
        key=lambda candidate: (
            -(candidate.val_mean_reward or 0.0),
            candidate.dimensions_digest,
        ),
    )


def _finalize_round(
    database_url: str, spec: StudySpec, round_index: int
) -> shared_batch.BatchOperationResult:
    all_candidates = study_records.read_candidates(database_url, spec.study_id)
    best = _select_overall_best(all_candidates, depth=spec.depth)
    if best is None or not spec.test_ids:
        study_records.update_study_state(
            database_url, spec.study_id, status="completed"
        )
        _emit_study_log(
            "study_finalized_without_test",
            {"study_id": spec.study_id, "has_best": best is not None},
        )
        return shared_batch.BatchOperationResult(
            start_offset=round_index,
            next_offset=round_index + 1,
            batch_size=1,
            processed=1,
            counters={"finalized": 1},
        )
    best_graph = GraphSpec.model_validate(best.graph)
    _inserted, _enqueued, scores = _submit_and_wait(
        database_url,
        spec,
        graphs=[best_graph],
        task_ids=spec.test_ids,
        submission_id=f"{spec.study_id}:test",
    )
    test_scores = scores[best.dimensions_digest]
    study_records.upsert_candidate(
        database_url,
        study_records.CandidateRow(
            study_id=spec.study_id,
            round_index=round_index,
            candidate_index=0,
            instruction=best.instruction,
            dimensions_digest=best.dimensions_digest,
            graph=best.graph,
            provenance={"phase": "test"},
            val_mean_reward=test_scores.mean_reward(),
            val_coverage=test_scores.coverage(),
            val_scores={"distribution": test_scores.reward_distribution()},
            selected=True,
        ),
    )
    study_row = study_records.read_study(database_url, spec.study_id)
    history = list(study_row.history) if study_row else []
    history.append(
        {
            "phase": "test",
            "instruction": best.instruction,
            "dimensions_digest": best.dimensions_digest,
            "test_mean_reward": test_scores.mean_reward(),
            "coverage": test_scores.coverage(),
        }
    )
    study_records.update_study_state(
        database_url, spec.study_id, status="completed", history=history
    )
    _emit_study_log(
        "study_finalized",
        {
            "study_id": spec.study_id,
            "best_instruction": best.instruction,
            "test_mean_reward": test_scores.mean_reward(),
        },
    )
    return shared_batch.BatchOperationResult(
        start_offset=round_index,
        next_offset=round_index + 1,
        batch_size=1,
        processed=1,
        counters={"finalized": 1},
    )


def study_round_step(
    database_url: str, operation_key: str
) -> shared_batch.BatchOperationResult:
    progress = shared_batch.fetch_operation_progress(
        database_url,
        operation_kind=shared_batch.BatchOperationKind.STUDY,
        operation_key=operation_key,
    )
    spec = StudySpec(
        **shared_batch.fetch_operation_spec(
            database_url,
            operation_kind=shared_batch.BatchOperationKind.STUDY,
            operation_key=operation_key,
        )
    )
    round_index = progress.next_offset
    if round_index >= spec.depth:
        return _finalize_round(database_url, spec, round_index)
    return _optimize_round(database_url, spec, round_index)


@DBOS.workflow(
    name="humaneval_eval_v1_study_dispatcher",
    max_recovery_attempts=1,
)
def study_dispatcher_workflow(database_url: str, operation_key: str) -> str:
    completion_modes = shared_batch.BatchDispatcherCompletionMode
    return shared_batch.run_operation_dispatcher(
        database_url,
        operation_kind=shared_batch.BatchOperationKind.STUDY,
        operation_key=operation_key,
        configure_logging=_configure_study_file_logging,
        emit_log=_emit_study_log,
        started_event="study_dispatcher_started",
        started_payload=lambda progress: {
            "operation_key": operation_key,
            "workflow_id": progress.workflow_id,
            "total_rounds": progress.total_items,
            "next_offset": progress.next_offset,
        },
        failed_event="study_dispatcher_failed",
        batch_step=study_round_step,
        completion_mode=completion_modes.OFFSET_TOTAL,
        completed_event="study_dispatcher_completed",
        completed_payload=lambda progress: {
            "operation_key": operation_key,
            "rounds": progress.total_items,
            "batch_count": progress.batch_count,
        },
    )


# --- CLI --------------------------------------------------------------


_STUDY_APP = typer.Typer(no_args_is_help=True)


def create_study_app(
    eval_config: EvalHumanEvalExperimentConfig,
    plan: StudyPlanConfig,
) -> typer.Typer:
    configure_experiment(eval_config)
    configure_study_plan(plan)
    return _STUDY_APP


@_STUDY_APP.command("init-db")
def init_db_command(
    database_url: Annotated[str | None, typer.Option("--database-url")] = None,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=None,
        generation_concurrency=DEFAULT_GENERATION_CONCURRENCY,
        scoring_concurrency=DEFAULT_SCORING_CONCURRENCY,
        env_file=env_file,
    )
    create_eval_schema(config.database_url)
    study_records.create_study_schema(config.database_url)
    operator_log("initialized dr-dspy eval + study tables", style="green")


@_STUDY_APP.command("study")
def study_command(
    experiment_name: Annotated[str, typer.Option("--experiment-name")],
    strategy: Annotated[str | None, typer.Option("--strategy")] = None,
    study_id: Annotated[str | None, typer.Option("--study-id")] = None,
    train: Annotated[int | None, typer.Option("--train", min=0)] = None,
    val: Annotated[int | None, typer.Option("--val", min=1)] = None,
    test: Annotated[int | None, typer.Option("--test", min=0)] = None,
    repetitions: Annotated[
        int | None, typer.Option("--repetitions", min=1)
    ] = None,
    breadth: Annotated[int | None, typer.Option("--breadth", min=1)] = None,
    depth: Annotated[int | None, typer.Option("--depth", min=1)] = None,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    score_timeout: Annotated[
        float | None, typer.Option("--score-timeout", min=0.1)
    ] = None,
    wait_interval: Annotated[
        float, typer.Option("--wait-interval", min=0.5)
    ] = 5.0,
    wait_timeout: Annotated[
        float, typer.Option("--wait-timeout", min=1.0)
    ] = 3600.0,
    database_url: Annotated[str | None, typer.Option("--database-url")] = None,
    dbos_system_database_url: Annotated[
        str | None, typer.Option("--dbos-system-database-url")
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    experiment = experiment_config()
    plan = study_plan()
    resolved_strategy = strategy or plan.default_strategy
    if resolved_strategy not in (GRID_STRATEGY, COPRO_STRATEGY):
        raise typer.BadParameter(
            f"strategy must be {GRID_STRATEGY} or {COPRO_STRATEGY}"
        )
    resolved_study_id = study_id or uuid.uuid4().hex
    resolved_train = train if train is not None else plan.default_train
    resolved_val = val if val is not None else plan.default_val
    resolved_test = test if test is not None else plan.default_test
    resolved_reps = repetitions or plan.default_repetitions
    resolved_breadth = breadth or plan.default_breadth
    resolved_depth = (
        depth
        if depth is not None
        else (1 if resolved_strategy == GRID_STRATEGY else plan.default_depth)
    )
    resolved_timeout = (
        score_timeout
        if score_timeout is not None
        else experiment.default_subprocess_timeout
    )
    if resolved_strategy == COPRO_STRATEGY and plan.prompt_model is None:
        raise typer.BadParameter(
            "copro study requires a configured prompt_model"
        )

    config = common_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=DEFAULT_GENERATION_CONCURRENCY,
        scoring_concurrency=DEFAULT_SCORING_CONCURRENCY,
        env_file=env_file,
    )
    split = shared_eval_set.build_eval_split(
        seed=seed,
        train=resolved_train,
        val=resolved_val,
        test=resolved_test,
        repetitions=resolved_reps,
        dataset_name=experiment.dataset_name,
        dataset_split=experiment.dataset_split,
    )
    spec = StudySpec(
        study_id=resolved_study_id,
        experiment_name=experiment_name,
        strategy=resolved_strategy,
        node_id=plan.node_id,
        base_graph=plan.base_graph,
        base_instruction=plan.base_instruction,
        val_ids=list(split.val_ids),
        test_ids=list(split.test_ids),
        repetitions=resolved_reps,
        score_timeout=resolved_timeout,
        seed=seed,
        depth=resolved_depth,
        breadth=resolved_breadth,
        grid_instructions=list(plan.grid_instructions),
        prompt_model=plan.prompt_model,
        wait_interval_seconds=wait_interval,
        wait_timeout_seconds=wait_timeout,
    )
    total_rounds = spec.depth + 1
    operation_key = shared_batch.operation_key(spec.model_dump(mode="json"))
    operator_log(
        f"study {resolved_study_id}: strategy={resolved_strategy}, "
        f"val={len(spec.val_ids)}, test={len(spec.test_ids)}, "
        f"reps={resolved_reps}, breadth={resolved_breadth}, "
        f"depth={resolved_depth}",
        style="cyan",
    )
    if dry_run:
        operator_log(
            "dry run only; nothing written or enqueued", style="yellow"
        )
        return

    create_eval_schema(config.database_url)
    study_records.create_study_schema(config.database_url)
    upsert_experiment(
        config.database_url,
        experiment_name=experiment_name,
        seed=seed,
        sample_count=len(spec.val_ids),
        metadata={
            "study_id": resolved_study_id,
            "strategy": resolved_strategy,
        },
    )
    study_records.upsert_study(
        config.database_url,
        study_records.StudyRow(
            study_id=resolved_study_id,
            experiment_name=experiment_name,
            strategy=resolved_strategy,
            status="pending",
            params={
                "breadth": resolved_breadth,
                "depth": resolved_depth,
                "repetitions": resolved_reps,
                "node_id": plan.node_id,
            },
            eval_set=split.model_dump(mode="json"),
        ),
    )
    resolved_log_file = resolve_operation_log_path(
        experiment_name=experiment_name,
        operation_kind=shared_batch.BatchOperationKind.STUDY,
    )
    metadata = {
        "study_id": resolved_study_id,
        "operation_key": operation_key,
        "strategy": resolved_strategy,
    }
    progress = shared_batch.prepare_operation(
        config.database_url,
        operation_kind=shared_batch.BatchOperationKind.STUDY,
        operation_key=operation_key,
        experiment_name=experiment_name,
        script_kind=experiment.script_kind,
        spec=spec.model_dump(mode="json"),
        metadata=metadata,
        total_items=total_rounds,
        log_file=resolved_log_file,
    )
    active_log_file = Path(progress.log_file)
    _configure_study_file_logging(active_log_file)
    _emit_study_log(
        "study_planned",
        {
            "study_id": resolved_study_id,
            "operation_key": operation_key,
            "total_rounds": total_rounds,
            "metadata": metadata,
        },
    )
    configure_dbos_runtime(
        config, experiment_name=experiment_name, consume_queues=False
    )
    launched = shared_batch.ensure_operation_workflow(
        workflow_id=progress.workflow_id,
        workflow=study_dispatcher_workflow,
        database_url=config.database_url,
        operation_key=operation_key,
    )
    _emit_study_log(
        "study_dispatcher_enqueued",
        {
            "operation_key": operation_key,
            "workflow_id": progress.workflow_id,
            "launched": launched,
            "log_file": str(active_log_file),
        },
    )
    operator_log(f"study detail log: {active_log_file}", style="cyan")
    final_progress = shared_batch.tail_operation_progress(
        database_url=config.database_url,
        operation_kind=shared_batch.BatchOperationKind.STUDY,
        operation_key=operation_key,
        prediction_table=PREDICTION_TABLE_NAME,
        experiment_name=experiment_name,
        operator_log=operator_log,
    )
    if final_progress.status is shared_batch.BatchOperationStatus.FAILED:
        raise typer.Exit(code=1)


@_STUDY_APP.command("worker")
def worker_command(
    experiment_name: Annotated[str, typer.Option("--experiment-name")],
    queue: Annotated[
        shared_dbos.QueueSelection,
        typer.Option("--queue"),
    ] = shared_dbos.QueueSelection.BOTH,
    database_url: Annotated[str | None, typer.Option("--database-url")] = None,
    dbos_system_database_url: Annotated[
        str | None, typer.Option("--dbos-system-database-url")
    ] = None,
    generation_concurrency: Annotated[
        int, typer.Option("--generation-concurrency")
    ] = DEFAULT_GENERATION_CONCURRENCY,
    scoring_concurrency: Annotated[
        int, typer.Option("--scoring-concurrency")
    ] = DEFAULT_SCORING_CONCURRENCY,
    open_file_limit: Annotated[
        str, typer.Option("--open-file-limit")
    ] = DEFAULT_WORKER_OPEN_FILE_LIMIT,
    log_file: Annotated[Path | None, typer.Option("--log-file")] = None,
    monitor: Annotated[bool, typer.Option("--monitor/--no-monitor")] = True,
    monitor_interval: Annotated[
        float, typer.Option("--monitor-interval", min=0.5)
    ] = DEFAULT_WORKER_MONITOR_INTERVAL_SECONDS,
    monitor_summary_interval: Annotated[
        float, typer.Option("--monitor-summary-interval", min=1.0)
    ] = DEFAULT_WORKER_MONITOR_SUMMARY_INTERVAL_SECONDS,
    db_pool_max_size: Annotated[
        str, typer.Option("--db-pool-max-size")
    ] = shared_dbos.DB_POOL_AUTO,
    env_file: Annotated[Path | None, typer.Option()] = None,
) -> None:
    config = common_config(
        database_url=database_url,
        dbos_system_database_url=dbos_system_database_url,
        generation_concurrency=generation_concurrency,
        scoring_concurrency=scoring_concurrency,
        env_file=env_file,
    )
    shared_flow.run_worker_command(
        EVAL_BACKEND,
        config=config,
        experiment_name=experiment_name,
        queue=queue,
        open_file_limit=open_file_limit,
        log_file=log_file,
        monitor=monitor,
        monitor_interval=monitor_interval,
        monitor_summary_interval=monitor_summary_interval,
        db_pool_max_size=db_pool_max_size,
    )
