import logging
import random
import threading

import tqdm
from pydantic import BaseModel

from dspy._internal.hashing import hash_pickle
from dspy.evaluate.metric_invoke import invoke_metric
from dspy.primitives import Module
from dspy.runtime import run_with_trace
from dspy.runtime.async_parallel import resolve_max_errors
from dspy.runtime.run_context import RunContext
from dspy.task_spec.predictor_context import get_task_spec
from dspy.teleprompt.compilation import CompileResult
from dspy.teleprompt.compile_params import BootstrapFewShotCompileParams, LabeledFewShotCompileParams
from dspy.teleprompt.core.demos import trace_to_demos
from dspy.teleprompt.metrics import OptimizerMetric
from dspy.teleprompt.registry import register_teleprompter

from .vanilla import LabeledFewShot

logger = logging.getLogger(__name__)


@register_teleprompter(params=BootstrapFewShotCompileParams)
class BootstrapFewShot:
    def __init__(
        self,
        metric: OptimizerMetric | None = None,
        metric_threshold=None,
        teacher_run: RunContext | None = None,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=1,
        max_errors=None,
    ) -> None:
        self.metric = metric
        self.metric_threshold = metric_threshold
        self.teacher_run = teacher_run
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max_rounds
        self.max_errors = max_errors
        self.error_count = 0
        self.error_lock = threading.Lock()

    async def compile(self, student: Module, *, params: BaseModel, run: RunContext) -> CompileResult:
        params = BootstrapFewShotCompileParams.model_validate(params)
        self.trainset = params.trainset
        await self._prepare_student_and_teacher(student=student, teacher=params.teacher, run=run)
        self._prepare_predictor_mappings()
        await self._bootstrap(run=run)
        self.student = self._train()
        return CompileResult.with_compiled_program(self.student)

    async def _prepare_student_and_teacher(self, student, teacher, *, run: RunContext) -> None:
        self.student = student.reset_copy()
        self.teacher = teacher.deepcopy() if teacher is not None else student.deepcopy()
        assert getattr(self.student, "_compiled", False) is False, "Student must be uncompiled."
        if self.max_labeled_demos and getattr(self.teacher, "_compiled", False) is False:
            teleprompter = LabeledFewShot(k=self.max_labeled_demos)
            teacher_result = await teleprompter.compile(
                self.teacher.reset_copy(),
                params=LabeledFewShotCompileParams(trainset=self.trainset),
                run=run,
            )
            self.teacher = teacher_result.program

    def _prepare_predictor_mappings(self) -> None:
        name2predictor, predictor2name = ({}, {})
        student, teacher = (self.student, self.teacher)
        assert len(student.predictors()) == len(teacher.predictors()), (
            "Student and teacher must have the same number of predictors."
        )
        for (name1, predictor1), (name2, predictor2) in zip(
            student.named_predictors(), teacher.named_predictors(), strict=True
        ):
            assert name1 == name2, "Student and teacher must have the same program structure."
            assert get_task_spec(predictor1) == get_task_spec(predictor2), (
                "Student and teacher must have the same task specs."
            )
            assert id(predictor1) != id(predictor2), "Student and teacher must be different objects."
            name2predictor[name1] = None
            predictor2name[id(predictor1)] = name1
            predictor2name[id(predictor2)] = name2
        self.name2predictor = name2predictor
        self.predictor2name = predictor2name

    async def _bootstrap(self, *, run: RunContext, max_bootstraps=None) -> None:
        max_bootstraps = max_bootstraps or self.max_bootstrapped_demos
        bootstrapped = {}
        self.name2traces = {name: [] for name in self.name2predictor}
        for example_idx, example in enumerate(tqdm.tqdm(self.trainset)):
            if len(bootstrapped) >= max_bootstraps:
                break
            for round_idx in range(self.max_rounds):
                if await self._bootstrap_one_example(example=example, round_idx=round_idx, run=run):
                    bootstrapped[example_idx] = True
                    break
        self.validation = [x for idx, x in enumerate(self.trainset) if idx not in bootstrapped]
        random.Random(0).shuffle(self.validation)

    async def _bootstrap_one_example(self, example, round_idx=0, *, run: RunContext):
        name2traces = {}
        teacher = self.teacher
        predictor_cache = {}
        trace: list = []
        success = False
        teacher_run = (self.teacher_run or run).fork(optimization_trace=[])
        lm = teacher_run.lm
        lm = lm.copy(temperature=1.0) if round_idx > 0 else lm
        if round_idx > 0:
            teacher_run = teacher_run.fork(lm=lm)
        for name, predictor in teacher.named_predictors():
            predictor_cache[name] = predictor.demos
            predictor.demos = [x for x in predictor.demos if x != example]
        try:
            prediction, trace = await run_with_trace(teacher, example, teacher_run)
            if self.metric:
                metric_val = await invoke_metric(
                    self.metric,
                    example=example,
                    prediction=prediction,
                    trace=trace,
                    run=run,
                )
                success = metric_val >= self.metric_threshold if self.metric_threshold else metric_val
            else:
                success = True
        except Exception as e:
            success = False
            with self.error_lock:
                self.error_count += 1
                current_error_count = self.error_count
            effective_max_errors = resolve_max_errors(self.max_errors, run)
            if current_error_count >= effective_max_errors:
                raise
            logger.exception(f"Failed to run or to evaluate example {example} with {self.metric} due to {e}.")
        finally:
            for name, predictor in teacher.named_predictors():
                predictor.demos = predictor_cache[name]
        if success:
            name2traces = trace_to_demos(trace, self.predictor2name)
            for name, demos in name2traces.items():
                if len(demos) > 1:
                    rng = random.Random(hash_pickle(tuple(demos)))
                    demos = [rng.choice(demos[:-1]) if rng.random() < 0.5 else demos[-1]]
                self.name2traces[name].extend(demos)
        return success

    def _train(self):
        rng = random.Random(0)
        raw_demos = self.validation
        for name, predictor in self.student.named_predictors():
            augmented_demos = self.name2traces[name][: self.max_bootstrapped_demos]
            sample_size = min(self.max_labeled_demos - len(augmented_demos), len(raw_demos))
            sample_size = max(0, sample_size)
            raw_demos = rng.sample(raw_demos, sample_size)
            predictor.demos = augmented_demos + raw_demos
        return self.student
