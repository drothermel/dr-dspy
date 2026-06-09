import logging
import random
import threading

import tqdm
from typing_extensions import override

from dspy.primitives.example import Example
from dspy.runtime.run_context import RunContext
from dspy.teleprompt.task_spec_context import get_task_spec
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.hasher import Hasher

from .vanilla import LabeledFewShot

logger = logging.getLogger(__name__)


class BootstrapFewShot(Teleprompter):
    def __init__(
        self,
        metric=None,
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

    @override
    async def compile(self, student, *, teacher=None, trainset, run: RunContext):
        self.trainset = trainset
        await self._prepare_student_and_teacher(student=student, teacher=teacher, run=run)
        self._prepare_predictor_mappings()
        await self._bootstrap(run=run)
        self.student = self._train()
        self.student._compiled = True
        return self.student

    async def _prepare_student_and_teacher(self, student, teacher, *, run: RunContext) -> None:
        self.student = student.reset_copy()
        self.teacher = teacher.deepcopy() if teacher is not None else student.deepcopy()
        assert getattr(self.student, "_compiled", False) is False, "Student must be uncompiled."
        if self.max_labeled_demos and getattr(self.teacher, "_compiled", False) is False:
            teleprompter = LabeledFewShot(k=self.max_labeled_demos)
            self.teacher = await teleprompter.compile(self.teacher.reset_copy(), trainset=self.trainset, run=run)

    def _prepare_predictor_mappings(self) -> None:
        name2predictor, predictor2name = ({}, {})
        student, teacher = (self.student, self.teacher)
        assert len(student.predictors()) == len(teacher.predictors()), (
            "Student and teacher must have the same number of predictors."
        )
        for (name1, predictor1), (name2, predictor2) in zip(
            student.named_predictors(), teacher.named_predictors(), strict=False
        ):
            assert name1 == name2, "Student and teacher must have the same program structure."
            assert get_task_spec(predictor1).equals(get_task_spec(predictor2)), (
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
        bootstrap_attempts = 0
        bootstrapped = {}
        self.name2traces = {name: [] for name in self.name2predictor}
        for example_idx, example in enumerate(tqdm.tqdm(self.trainset)):
            if len(bootstrapped) >= max_bootstraps:
                break
            for round_idx in range(self.max_rounds):
                bootstrap_attempts += 1
                if await self._bootstrap_one_example(example=example, round_idx=round_idx, run=run):
                    bootstrapped[example_idx] = True
                    break
        self.validation = [x for idx, x in enumerate(self.trainset) if idx not in bootstrapped]
        random.Random(0).shuffle(self.validation)
        self.validation = self.validation

    async def _bootstrap_one_example(self, example, round_idx=0, *, run: RunContext):
        name2traces = {}
        teacher = self.teacher
        predictor_cache = {}
        trace: list = []
        try:
            teacher_run = (self.teacher_run or run).fork(trace=[])
            lm = teacher_run.lm
            lm = lm.copy(temperature=1.0) if round_idx > 0 else lm
            if round_idx > 0:
                teacher_run = teacher_run.fork(lm=lm)
            item_run = teacher_run.fork(trace=[])
            for name, predictor in teacher.named_predictors():
                predictor_cache[name] = predictor.demos
                predictor.demos = [x for x in predictor.demos if x != example]
            prediction = await teacher(**example.as_inputs(), run=item_run)
            trace = list(item_run.trace)
            for name, predictor in teacher.named_predictors():
                predictor.demos = predictor_cache[name]
            if self.metric:
                metric_val = self.metric(example, prediction, trace)
                success = metric_val >= self.metric_threshold if self.metric_threshold else metric_val
            else:
                success = True
        except Exception as e:
            success = False
            with self.error_lock:
                self.error_count += 1
                current_error_count = self.error_count
            effective_max_errors = self.max_errors if self.max_errors is not None else run.execution.max_errors
            if current_error_count >= effective_max_errors:
                raise
            logger.exception(f"Failed to run or to evaluate example {example} with {self.metric} due to {e}.")
        if success:
            for step in trace:
                predictor, inputs, outputs = step
                demo = Example.from_record({"augmented": True, **inputs, **outputs})
                try:
                    predictor_name = self.predictor2name[id(predictor)]
                except KeyError:
                    continue
                name2traces[predictor_name] = name2traces.get(predictor_name, [])
                name2traces[predictor_name].append(demo)
            for name, demos in name2traces.items():
                if len(demos) > 1:
                    rng = random.Random(Hasher.hash(tuple(demos)))
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
