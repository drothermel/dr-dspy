from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, Any, cast

from dspy._internal.lazy_import import require
from dspy.predict.parallel import Parallel
from dspy.teleprompt.simba_utils import append_a_demo, append_a_rule, prepare_models_for_resampling, wrap_program
from dspy.teleprompt.task_spec_context import resolve_optimizer_lm

if TYPE_CHECKING:
    from pydantic import BaseModel

    from dspy.clients.lm import LM
    from dspy.primitives import Module
    from dspy.runtime.run_context import RunContext
    from dspy.teleprompt.metrics import OptimizerMetric
from dspy.teleprompt.compilation import CompileResult, CompileStats, ProgramCandidate
from dspy.teleprompt.compile_params import SIMBACompileParams
from dspy.teleprompt.registry import register_teleprompter

np = require("numpy")
logger = logging.getLogger(__name__)


@register_teleprompter(params=SIMBACompileParams)
class SIMBA:
    def __init__(
        self,
        *,
        metric: OptimizerMetric,
        bsize: int = 32,
        num_candidates: int = 6,
        max_steps: int = 8,
        max_demos: int = 4,
        prompt_model: LM | None = None,
        teacher_run: RunContext | None = None,
        demo_input_field_maxlen: int = 100000,
        max_concurrency: int | None = None,
        temperature_for_sampling: float = 0.2,
        temperature_for_candidates: float = 0.2,
    ) -> None:
        self.metric = metric
        self.bsize = bsize
        self.num_candidates = num_candidates
        self.max_steps = max_steps
        self.max_demos = max_demos
        self.prompt_model = prompt_model
        self.teacher_run = teacher_run
        self.demo_input_field_maxlen = demo_input_field_maxlen
        self.max_concurrency = max_concurrency
        self.temperature_for_sampling = temperature_for_sampling
        self.temperature_for_candidates = temperature_for_candidates
        if self.max_demos > 0:
            self.strategies = [append_a_demo(demo_input_field_maxlen), append_a_rule]
        else:
            self.strategies = [append_a_rule]

    async def compile(self, student: Module, *, params: BaseModel, run: RunContext) -> CompileResult:
        params = SIMBACompileParams.model_validate(params)
        trainset = params.trainset
        assert len(trainset) >= self.bsize, f"Trainset too small: {len(trainset)} < {self.bsize}"
        prompt_model = resolve_optimizer_lm(self.prompt_model, run=run)
        rng = random.Random(params.seed)
        rng_np = np.random.default_rng(params.seed)
        programs = []
        program_scores = {}
        next_program_idx = 0

        def calc_average_score(prog_idx: int) -> float:
            scores = program_scores.get(prog_idx, [])
            if not scores:
                return 0.0
            return sum(scores) / len(scores)

        def top_k_plus_baseline(k: int) -> list[int]:
            scored_programs = sorted(range(len(programs)), key=lambda idx: calc_average_score(idx), reverse=True)
            top_k = scored_programs[:k]
            if 0 not in top_k and len(top_k) > 0:
                top_k[-1] = 0
            return list(dict.fromkeys(top_k))

        def softmax_sample(rng_obj: random.Random, program_idxs: list[int], temperature: float) -> int:
            if not program_idxs:
                raise ValueError("No programs available for softmax sampling.")
            scores = [calc_average_score(idx) for idx in program_idxs]
            exps = [np.exp(s / temperature) for s in scores]
            sum_exps = sum(exps)
            if sum_exps <= 0:
                return rng_obj.choice(program_idxs)
            probs = [val / sum_exps for val in exps]
            return rng_obj.choices(program_idxs, weights=probs, k=1)[0]

        def register_new_program(prog: Module, score_list: list[float]) -> None:
            nonlocal next_program_idx
            next_program_idx += 1
            programs.append(prog)
            program_scores[next_program_idx] = score_list

        student = student.deepcopy()
        programs.append(student)
        program_scores[0] = []
        winning_programs = [student]
        data_indices = list(range(len(trainset)))
        rng.shuffle(data_indices)
        instance_idx = 0
        run_parallel = Parallel(run=run, access_examples=False, max_concurrency=self.max_concurrency)
        trial_logs = {}
        for batch_idx in range(self.max_steps):
            trial_logs[batch_idx] = {}
            logger.info(f"Starting batch {batch_idx + 1} of {self.max_steps}.")
            if instance_idx + self.bsize > len(trainset):
                rng.shuffle(data_indices)
                instance_idx = 0
            batch_indices = data_indices[instance_idx : instance_idx + self.bsize]
            batch = [trainset[i] for i in batch_indices]
            instance_idx += self.bsize
            models = prepare_models_for_resampling(
                program=programs[0], n=self.num_candidates, run=run, teacher_run=self.teacher_run
            )
            top_programs = top_k_plus_baseline(self.num_candidates)
            exec_pairs = []
            predictor2name = {}
            for model in models:
                for example in batch:
                    chosen_prog_idx = softmax_sample(
                        rng_obj=rng, program_idxs=top_programs, temperature=self.temperature_for_sampling
                    )
                    candidate_system = programs[chosen_prog_idx].deepcopy()
                    candidate_system.set_lm(model)
                    for name, predictor in candidate_system.named_predictors():
                        predictor2name[id(predictor)] = name
                    wrapped_candidate_system = wrap_program(program=candidate_system, metric=self.metric, run=run)
                    exec_pairs.append((wrapped_candidate_system, example))
            logger.info(f"Sampling program trajectories on {self.bsize} examples x {self.num_candidates} samples.")
            outputs = cast("list[dict[str, Any]]", (await run_parallel(exec_pairs)).results)
            assert len(outputs) == len(exec_pairs) == self.bsize * self.num_candidates
            buckets = []
            largest_max_to_avg_gap = float("-inf")
            batch_10th_percentile_score = np.percentile([float(o["score"]) for o in outputs], 10)
            batch_90th_percentile_score = np.percentile([float(o["score"]) for o in outputs], 90)
            for idx, _ in enumerate(batch):
                bucket = [outputs[i] for i in range(idx, len(outputs), self.bsize)]
                bucket.sort(key=lambda x: x["score"], reverse=True)
                max_score = float(bucket[0]["score"])
                min_score = float(bucket[-1]["score"])
                avg_score = sum(x["score"] for x in bucket) / len(bucket)
                max_to_min_gap = max_score - min_score
                max_to_avg_gap = max_score - avg_score
                if max_to_avg_gap > largest_max_to_avg_gap:
                    largest_max_to_avg_gap = max_to_avg_gap
                buckets.append((bucket, (max_to_min_gap, max_score, max_to_avg_gap)))
            buckets.sort(key=lambda x: x[1], reverse=True)
            all_scores_in_this_batch = [o["score"] for o in outputs]
            baseline_score = sum(all_scores_in_this_batch) / len(all_scores_in_this_batch)
            logger.info(f"Batch {batch_idx + 1}: Baseline mini-batch score: {baseline_score}\n")
            system_candidates = []
            for bucket_idx, (bucket, bucket_stats) in enumerate(buckets):
                max_to_min_gap, max_score, max_to_avg_gap = bucket_stats
                logger.info(
                    f"Batch {batch_idx + 1}: Processing bucket #{bucket_idx + 1}, with max score {max_score}, max-to-min gap {max_to_min_gap}, and max-to-avg gap {max_to_avg_gap}."
                )
                src_prog_idx = softmax_sample(
                    rng_obj=rng,
                    program_idxs=top_k_plus_baseline(self.num_candidates),
                    temperature=self.temperature_for_candidates,
                )
                system_candidate = programs[src_prog_idx].deepcopy()
                name2predictor = {}
                num_demos_list = []
                max_demos_tmp = self.max_demos if self.max_demos > 0 else 3
                for name, predictor in system_candidate.named_predictors():
                    name2predictor[name] = predictor
                    num_demos_list.append(len(predictor.demos))
                num_demos = max(num_demos_list) if num_demos_list else 0
                num_demos_to_drop = max(rng_np.poisson(num_demos / max_demos_tmp), int(num_demos >= max_demos_tmp))
                num_demos_to_drop = min(num_demos_to_drop, num_demos)
                demos_to_drop = [rng.randrange(num_demos) for _ in range(num_demos_to_drop)]
                for predictor in name2predictor.values():
                    predictor.demos = [demo for idxd, demo in enumerate(predictor.demos) if idxd not in demos_to_drop]
                strategy = rng.choice(self.strategies)
                logger.info(
                    f"Batch {batch_idx + 1}: Invoking strategy: {strategy.__name__}"
                    + (f", having dropped {num_demos_to_drop} demos per predictor" if num_demos_to_drop else "")
                )
                try:
                    await strategy(
                        bucket,
                        system_candidate,
                        run=run,
                        predictor2name=predictor2name,
                        name2predictor=name2predictor,
                        batch_10p_score=batch_10th_percentile_score,
                        batch_90p_score=batch_90th_percentile_score,
                        prompt_model=prompt_model,
                    )
                except Exception as e:
                    logger.exception(f"Strategy failed with error: {e}")
                    continue
                system_candidates.append(system_candidate)
                logger.info("\n")
                if len(system_candidates) >= self.num_candidates + 1:
                    break
            logger.info(
                f"Batch {batch_idx + 1}: Evaluating {len(system_candidates)} programs on {self.bsize} examples."
            )
            exec_pairs = [
                (wrap_program(program=sys, metric=self.metric, run=run), ex)
                for sys in system_candidates
                for ex in batch
            ]
            outputs = cast("list[dict[str, Any]]", (await run_parallel(exec_pairs)).results)
            assert len(outputs) == len(exec_pairs) == len(system_candidates) * self.bsize
            candidate_scores = []
            for idx_cand, _ in enumerate(system_candidates):
                start = idx_cand * self.bsize
                end = (idx_cand + 1) * self.bsize
                sys_scores = [outputs[i]["score"] for i in range(start, end)]
                avg_sys_score = sum(sys_scores) / len(sys_scores)
                candidate_scores.append(avg_sys_score)
            logger.info(
                f"Scores after {batch_idx + 1} batches: {candidate_scores}, Best: {(max(candidate_scores) if candidate_scores else 'N/A')}\n"
            )
            if candidate_scores:
                best_idx_among_candidates = candidate_scores.index(max(candidate_scores))
                best_program = system_candidates[best_idx_among_candidates]
                winning_programs.append(best_program.deepcopy())
            for idx_cand, cand_sys in enumerate(system_candidates):
                start = idx_cand * self.bsize
                end = (idx_cand + 1) * self.bsize
                sys_scores = [outputs[i]["score"] for i in range(start, end)]
                register_new_program(cand_sys, sys_scores)
        M = len(winning_programs) - 1
        N = self.num_candidates + 1
        program_idxs = [0] * N if M < 1 else [round(i * M / (N - 1)) for i in range(N)]
        program_idxs = list(dict.fromkeys(program_idxs))
        candidate_programs = [winning_programs[i].deepcopy() for i in program_idxs]
        logger.info(f"VALIDATION: Evaluating {len(candidate_programs)} programs on the full trainset.")
        exec_pairs = [
            (wrap_program(program=sys, metric=self.metric, run=run), ex)
            for sys in candidate_programs
            for ex in trainset
        ]
        outputs = cast("list[dict[str, Any]]", (await run_parallel(exec_pairs)).results)
        scores = []
        for idx_prog, _ in enumerate(candidate_programs):
            start = idx_prog * len(trainset)
            end = (idx_prog + 1) * len(trainset)
            sys_scores = [outputs[i]["score"] for i in range(start, end)]
            avg_score = sum(sys_scores) / len(sys_scores) if sys_scores else 0.0
            scores.append(avg_score)
            if idx_prog != 0:
                trial_logs[idx_prog - 1]["train_score"] = avg_score
        assert len(scores) == len(candidate_programs)
        candidate_entries = [
            ProgramCandidate(score=score, program=program)
            for score, program in zip(scores, candidate_programs, strict=True)
        ]
        candidate_entries.sort(
            key=lambda entry: entry.score if entry.score is not None else float("-inf"), reverse=True
        )
        best_program = (
            candidate_entries[0].program.deepcopy() if candidate_entries else candidate_programs[0].deepcopy()
        )
        logger.info(
            f"Final trainset scores: {scores}, Best: {(candidate_entries[0].score if candidate_entries else 'N/A')}\n\n\n"
        )
        return CompileResult(
            program=best_program,
            candidates=candidate_entries,
            stats=CompileStats(best_score=max(scores) if scores else None, trial_logs=trial_logs),
        )
