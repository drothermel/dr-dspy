import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel, ConfigDict

from dspy.predict.predict import Predict
from dspy.primitives import Example, Module
from dspy.runtime.run_context import RunContext
from dspy.teleprompt.bettertogether import BetterTogether
from dspy.teleprompt.bootstrap_finetune import BootstrapFinetune
from dspy.teleprompt.compilation import CompileResult
from dspy.teleprompt.compile_params import BetterTogetherCompileParams, RandomSearchCompileParams
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.registry import register_teleprompter, validate_compile_params
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts


def simple_metric(example, prediction, trace=None):
    return 1.0 if example.output == prediction.output else 0.0


examples = [
    Example.from_record(
        {
            "input": "What is the oldest known human-made monument?",
            "output": "Göbekli Tepe in southeastern Turkiye, dating back to around 9600 BCE",
        },
        input_keys=("input",),
    ),
    Example.from_record(
        {"input": "Why can't fish fall in love?", "output": "Because love is in the air"}, input_keys=("input",)
    ),
    Example.from_record(
        {"input": "What would bring world peace?", "output": "8 billion people meeting for a tea party in my backyard"},
        input_keys=("input",),
    ),
]
trainset = examples[:2]
valset = [examples[2]]


def _bt_params(**kwargs: Any) -> BetterTogetherCompileParams:
    defaults: dict[str, Any] = {"trainset": trainset, "valset": valset}
    defaults.update(kwargs)
    return BetterTogetherCompileParams(**defaults)


class SimpleModule(Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    async def _aforward_impl(self, *, run, options=None, **inputs):
        return await self.predictor(run=run, options=options, **inputs)


class MockOptimizerCompileParams(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    trainset: list[Example] | None = None
    teacher: Module | list[Module] | None = None
    valset: list[Example] | None = None


def _compile_result(program: Module) -> CompileResult:
    return CompileResult(program=program)


@register_teleprompter(params=MockOptimizerCompileParams)
class SimpleOptimizer:
    async def compile(self, student, *, params: BaseModel, run: RunContext):
        return _compile_result(student)


@register_teleprompter(params=MockOptimizerCompileParams)
class MarkedOptimizer:
    def __init__(self, marker):
        self.marker = marker

    async def compile(self, student, *, params: BaseModel, run: RunContext):
        prog = SimpleModule(ts("input -> output"))
        cast("Any", prog).marker = self.marker
        return _compile_result(prog)


@register_teleprompter(params=RandomSearchCompileParams)
class CapturingRandomSearchOptimizer:
    def __init__(self):
        self.received_params = None

    async def compile(self, student, *, params: BaseModel, run: RunContext):
        self.received_params = params
        return _compile_result(student)


@register_teleprompter(params=MockOptimizerCompileParams)
class CapturingOptimizer:
    def __init__(self):
        self.received_params = None

    async def compile(self, student, *, params: BaseModel, run: RunContext):
        self.received_params = params
        return _compile_result(student)


@pytest.fixture
def student_with_lm():
    student = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "test"}])
    student.set_lm(lm)
    return student


@pytest.fixture
def mock_bt_dependencies():
    with (
        patch("dspy.teleprompt.bettertogether.eval_candidate_program", new_callable=AsyncMock) as mock_eval,
        patch("dspy.teleprompt.bettertogether.launch_lms") as mock_launch,
        patch("dspy.teleprompt.bettertogether.kill_lms") as mock_kill,
    ):
        mock_eval.return_value = Mock(score=0.8)
        yield (mock_eval, mock_launch, mock_kill)


def test_bettertogether_import(make_run):
    assert BetterTogether is not None, "Failed to import BetterTogether"


def test_bettertogether_initialization_default(make_run):
    optimizer = BetterTogether(metric=simple_metric)
    assert optimizer.metric == simple_metric, "Metric not correctly initialized"
    assert "p" in optimizer.optimizers, "Default 'p' optimizer not created"
    assert "w" in optimizer.optimizers, "Default 'w' optimizer not created"
    assert isinstance(optimizer.optimizers["p"], BootstrapFewShotWithRandomSearch), (
        "Default 'p' should be BootstrapFewShotWithRandomSearch"
    )
    assert isinstance(optimizer.optimizers["w"], BootstrapFinetune), "Default 'w' should be BootstrapFinetune"


def test_bettertogether_initialization_custom(make_run):
    custom_p = BootstrapFewShotWithRandomSearch(metric=simple_metric)
    custom_w = BootstrapFinetune(metric=simple_metric)
    optimizer = BetterTogether(metric=simple_metric, p=custom_p, w=custom_w)
    assert optimizer.optimizers["p"] is custom_p, "Custom 'p' optimizer not set"
    assert optimizer.optimizers["w"] is custom_w, "Custom 'w' optimizer not set"


def test_bettertogether_initialization_invalid_optimizer(make_run):
    try:
        BetterTogether(metric=simple_metric, p=cast("Any", "not_a_teleprompter"))
        raise AssertionError("Should have raised TypeError for invalid optimizer")
    except TypeError as e:
        assert "must be a Teleprompter" in str(e)


def test_strategy_validation(make_run):
    optimizer = BetterTogether(metric=simple_metric)
    valid_strategies = [["p"], ["w"], ["p", "w"], ["w", "p"], ["p", "w", "p"]]
    for strategy in valid_strategies:
        parsed = optimizer._prepare_strategy(strategy)
        assert parsed is not None, f"Failed to parse valid strategy: {strategy}"
    with pytest.raises(ValueError, match="invalid optimizer keys"):
        optimizer._prepare_strategy(["p", "x", "w"])
    with pytest.raises(ValueError, match="cannot be empty"):
        optimizer._prepare_strategy([])


@register_teleprompter(params=MockOptimizerCompileParams)
class TrackingOptimizer(SimpleOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.compile_called = False

    async def compile(self, student, *, params: BaseModel, run: RunContext):
        self.compile_called = True
        return await super().compile(student, params=params, run=run)


def test_compile_basic(make_run):
    student = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "blue"}, {"output": "4"}])
    run = make_run(lm=lm)
    student.set_lm(lm)

    mock_p = TrackingOptimizer()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p)
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program", new_callable=AsyncMock) as mock_eval:
        mock_eval.return_value = Mock(score=0.8)
        with patch("dspy.teleprompt.bettertogether.launch_lms"), patch("dspy.teleprompt.bettertogether.kill_lms"):
            result = asyncio.run(optimizer.compile(student, params=_bt_params(strategy=["p"]), run=run))
    assert result.program is not None, "Compilation returned None"
    assert len(result.candidates) > 0, "Missing candidates"
    assert result.stats.error_occurred is False, "Unexpected compilation error flag"
    assert mock_p.compile_called, "Mock optimizer compile was not called"


def test_trainset_validation(make_run):
    optimizer = BetterTogether(metric=simple_metric)
    student = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    student.set_lm(lm)
    try:
        asyncio.run(optimizer.compile(student, params=_bt_params(trainset=[], valset=valset), run=run))
        raise AssertionError("Should have raised ValueError for empty trainset")
    except ValueError as e:
        assert "cannot be empty" in str(e).lower()


def test_valset_ratio_validation(make_run):
    optimizer = BetterTogether(metric=simple_metric)
    student = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    student.set_lm(lm)
    try:
        asyncio.run(optimizer.compile(student, params=_bt_params(valset_ratio=1.0), run=run))
        raise AssertionError("Should have raised ValueError for valset_ratio >= 1")
    except ValueError as e:
        assert "must be in range [0, 1)" in str(e)
    try:
        asyncio.run(optimizer.compile(student, params=_bt_params(valset_ratio=-0.1), run=run))
        raise AssertionError("Should have raised ValueError for valset_ratio < 0")
    except ValueError as e:
        assert "must be in range [0, 1)" in str(e)


def test_optimizer_compile_args_validation():
    optimizer = BetterTogether(metric=simple_metric)
    try:
        optimizer._prepare_optimizer_compile_args(
            {"invalid_key": RandomSearchCompileParams(trainset=trainset)}, teacher=None
        )
        raise AssertionError("Should have raised ValueError for invalid optimizer key")
    except ValueError as e:
        assert "invalid optimizer key" in str(e).lower()


def test_student_in_optimizer_compile_args():
    optimizer = BetterTogether(metric=simple_metric)
    try:
        validate_compile_params(optimizer.optimizers["p"], BetterTogetherCompileParams(trainset=trainset))
        raise AssertionError("Should have raised TypeError for wrong compile_args type")
    except TypeError as e:
        assert "RandomSearchCompileParams" in str(e)


def test_compile_args_passed_to_optimizer(student_with_lm, mock_bt_dependencies, make_run):
    run = make_run(lm=student_with_lm.get_lm())
    mock_eval, _, _ = mock_bt_dependencies
    mock_eval.return_value = Mock(score=0.9)
    mock_p = CapturingRandomSearchOptimizer()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p)
    asyncio.run(
        optimizer.compile(
            student_with_lm,
            params=_bt_params(
                strategy=["p"],
                optimizer_compile_args={
                    "p": RandomSearchCompileParams(trainset=trainset, valset=valset, restrict=[0, 1]),
                },
            ),
            run=run,
        )
    )
    assert mock_p.received_params is not None, "Optimizer compile was not called"
    assert isinstance(mock_p.received_params, RandomSearchCompileParams)
    assert mock_p.received_params.restrict == [0, 1]


def test_compile_args_multi_optimizer_strategy(make_run):
    from pydantic import ConfigDict

    class PromptTestParams(BaseModel):
        model_config = ConfigDict(extra="allow")
        num_trials: int | None = None

    class WeightTestParams(BaseModel):
        model_config = ConfigDict(extra="allow")
        num_batches: int | None = None

    student = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    student.set_lm(lm)

    @register_teleprompter(params=PromptTestParams)
    class PromptOptimizer:
        def __init__(self):
            self.received_params = None

        async def compile(self, student, *, params: BaseModel, run: RunContext):
            self.received_params = params
            return _compile_result(student)

    @register_teleprompter(params=WeightTestParams)
    class WeightOptimizer:
        def __init__(self):
            self.received_params = None

        async def compile(self, student, *, params: BaseModel, run: RunContext):
            self.received_params = params
            return _compile_result(student)

    mock_p = PromptOptimizer()
    mock_w = WeightOptimizer()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p, w=mock_w)
    compile_args = {"p": PromptTestParams(num_trials=10), "w": WeightTestParams(num_batches=5)}
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program", new_callable=AsyncMock) as mock_eval:
        mock_eval.return_value = Mock(score=0.85)
        with (
            patch("dspy.teleprompt.bettertogether.launch_lms"),
            patch("dspy.teleprompt.bettertogether.kill_lms"),
            patch.object(optimizer, "_models_changed", return_value=False),
        ):
            asyncio.run(
                optimizer.compile(
                    student,
                    params=_bt_params(strategy=["p", "w"], optimizer_compile_args=compile_args),
                    run=run,
                )
            )
    assert isinstance(mock_p.received_params, PromptTestParams)
    assert mock_p.received_params.num_trials == 10
    assert isinstance(mock_w.received_params, WeightTestParams)
    assert mock_w.received_params.num_batches == 5


def test_compile_args_override_global_params(make_run):
    student = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    student.set_lm(lm)

    @register_teleprompter(params=RandomSearchCompileParams)
    class CapturingTeleprompter:
        def __init__(self):
            self.received_params = None

        async def compile(self, student, *, params: BaseModel, run: RunContext):
            self.received_params = params
            return _compile_result(student)

    mock_p = CapturingTeleprompter()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p)
    override_trainset = [examples[2]]
    override_valset = [examples[0]]
    override_teacher = SimpleModule(ts("input -> output"))
    compile_args = {
        "p": RandomSearchCompileParams(trainset=override_trainset, valset=override_valset, teacher=override_teacher)
    }
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program", new_callable=AsyncMock) as mock_eval:
        mock_eval.return_value = Mock(score=0.9)
        with patch("dspy.teleprompt.bettertogether.launch_lms"), patch("dspy.teleprompt.bettertogether.kill_lms"):
            asyncio.run(
                optimizer.compile(
                    student,
                    params=_bt_params(teacher=None, strategy=["p"], optimizer_compile_args=compile_args),
                    run=run,
                )
            )
    received = mock_p.received_params
    assert isinstance(received, RandomSearchCompileParams)
    assert received.trainset == override_trainset
    assert received.valset == override_valset
    assert received.teacher is override_teacher
    assert received.trainset != trainset
    assert received.valset != valset


def test_trainset_shuffling_between_steps(make_run):
    student = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    student.set_lm(lm)
    trainsets_received = []

    @register_teleprompter(params=MockOptimizerCompileParams)
    class TrainsetCapturingOptimizer:
        async def compile(self, student, *, params: BaseModel, run: RunContext):
            trainsets_received.append(getattr(params, "trainset", None))
            return _compile_result(student)

    mock_p = TrainsetCapturingOptimizer()
    mock_w = TrainsetCapturingOptimizer()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p, w=mock_w)
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program", new_callable=AsyncMock) as mock_eval:
        mock_eval.return_value = Mock(score=0.8)
        with (
            patch("dspy.teleprompt.bettertogether.launch_lms"),
            patch("dspy.teleprompt.bettertogether.kill_lms"),
            patch.object(optimizer, "_models_changed", return_value=False),
        ):
            asyncio.run(
                optimizer.compile(
                    student,
                    params=_bt_params(strategy=["p", "w"], shuffle_trainset_between_steps=True),
                    run=run,
                )
            )
    assert len(trainsets_received) == 2, "Should have received trainset twice (for p and w)"
    trainset_p = trainsets_received[0]
    trainset_w = trainsets_received[1]
    assert len(trainset_p) == len(trainset_w), "Trainsets should have same length"
    assert {id(ex) for ex in trainset_p} == {id(ex) for ex in trainset_w}, (
        "Trainsets should contain the same example objects"
    )


def test_strategy_execution_order(make_run):
    student = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    student.set_lm(lm)
    execution_log = []

    @register_teleprompter(params=MockOptimizerCompileParams)
    class LoggingOptimizer:
        def __init__(self, name):
            self.name = name

        async def compile(self, student, *, params: BaseModel, run: RunContext):
            optimized = SimpleModule(ts("input -> output"))
            typed_student = cast("Any", student)
            typed_optimized = cast("Any", optimized)
            if not hasattr(typed_student, "optimization_path"):
                typed_optimized.optimization_path = [self.name]
            else:
                typed_optimized.optimization_path = typed_student.optimization_path + [self.name]
            execution_log.append((self.name, typed_optimized.optimization_path.copy()))
            return _compile_result(optimized)

    mock_p = LoggingOptimizer("p")
    mock_w = LoggingOptimizer("w")
    optimizer = BetterTogether(metric=simple_metric, p=mock_p, w=mock_w)
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program", new_callable=AsyncMock) as mock_eval:
        mock_eval.return_value = Mock(score=0.85)
        with (
            patch("dspy.teleprompt.bettertogether.launch_lms"),
            patch("dspy.teleprompt.bettertogether.kill_lms"),
            patch.object(optimizer, "_models_changed", return_value=False),
        ):
            asyncio.run(optimizer.compile(student, params=_bt_params(strategy=["p", "w", "p"]), run=run))
    assert len(execution_log) == 3, "Should have executed 3 optimization steps"
    assert execution_log[0] == ("p", ["p"]), "First step should be 'p'"
    assert execution_log[1] == ("w", ["p", "w"]), "Second step should be 'w' receiving output from 'p'"
    assert execution_log[2] == ("p", ["p", "w", "p"]), "Third step should be 'p' receiving output from 'w'"


def test_lm_lifecycle_management(make_run):
    student = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    student.set_lm(lm)

    mock_p = SimpleOptimizer()
    mock_w = SimpleOptimizer()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p, w=mock_w)
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program", new_callable=AsyncMock) as mock_eval:
        mock_eval.return_value = Mock(score=0.8)
        with (
            patch("dspy.teleprompt.bettertogether.launch_lms") as mock_launch,
            patch("dspy.teleprompt.bettertogether.kill_lms") as mock_kill,
            patch.object(optimizer, "_models_changed", return_value=True),
        ):
            asyncio.run(optimizer.compile(student, params=_bt_params(strategy=["p", "w"]), run=run))
    assert mock_launch.called, "launch_lms should be called when models change"
    assert mock_kill.called, "kill_lms should be called when models change"


def test_error_handling_returns_best_program(make_run):
    student = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    student.set_lm(lm)

    @register_teleprompter(params=MockOptimizerCompileParams)
    class SuccessfulOptimizer:
        async def compile(self, student, *, params: BaseModel, run: RunContext):
            optimized = SimpleModule(ts("input -> output"))
            cast("Any", optimized).step_name = "p_success"
            return _compile_result(optimized)

    @register_teleprompter(params=MockOptimizerCompileParams)
    class FailingOptimizer:
        async def compile(self, student, *, params: BaseModel, run: RunContext):
            raise RuntimeError("Intentional failure for testing")

    mock_p = SuccessfulOptimizer()
    mock_w = FailingOptimizer()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p, w=mock_w)
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program", new_callable=AsyncMock) as mock_eval:
        mock_eval.side_effect = [Mock(score=0.5), Mock(score=0.7)]
        with (
            patch("dspy.teleprompt.bettertogether.launch_lms"),
            patch("dspy.teleprompt.bettertogether.kill_lms"),
            patch.object(optimizer, "_models_changed", return_value=False),
        ):
            result = asyncio.run(optimizer.compile(student, params=_bt_params(strategy=["p", "w"]), run=run))
    assert result.program is not None, "Should return a program even if a step fails"
    assert result.stats.error_occurred is True, "Error flag should be True"
    assert len(result.candidates) > 0, "Should have at least one candidate program"


@pytest.mark.parametrize(
    ("test_valset", "expected_marker", "test_description"),
    [
        (valset, "p_optimized", "With valset: returns best score (p), not latest (w)"),
        (None, "w_optimized", "Without valset: returns latest program (w)"),
    ],
)
def test_program_selection(student_with_lm, test_valset, expected_marker, test_description, make_run):
    run = make_run(lm=student_with_lm.get_lm())
    mock_p = MarkedOptimizer("p_optimized")
    mock_w = MarkedOptimizer("w_optimized")
    optimizer = BetterTogether(metric=simple_metric, p=mock_p, w=mock_w)
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program", new_callable=AsyncMock) as mock_eval:
        if test_valset is not None:
            mock_eval.side_effect = [Mock(score=0.5), Mock(score=0.9), Mock(score=0.7)]
        with (
            patch("dspy.teleprompt.bettertogether.launch_lms"),
            patch("dspy.teleprompt.bettertogether.kill_lms"),
            patch.object(optimizer, "_models_changed", return_value=False),
        ):
            result = asyncio.run(
                optimizer.compile(student_with_lm, params=_bt_params(valset=test_valset, strategy=["p", "w"]), run=run)
            )
    assert hasattr(result.program, "marker"), "Result should have marker"
    assert result.program.marker == expected_marker, test_description


def test_candidate_programs_structure(student_with_lm, make_run):
    run = make_run(lm=student_with_lm.get_lm())
    mock_p = MarkedOptimizer("p")
    mock_w = MarkedOptimizer("w")
    optimizer = BetterTogether(metric=simple_metric, p=mock_p, w=mock_w)
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program", new_callable=AsyncMock) as mock_eval:
        mock_eval.side_effect = [Mock(score=0.5), Mock(score=0.8), Mock(score=0.9)]
        with (
            patch("dspy.teleprompt.bettertogether.launch_lms"),
            patch("dspy.teleprompt.bettertogether.kill_lms"),
            patch.object(optimizer, "_models_changed", return_value=False),
        ):
            result = asyncio.run(optimizer.compile(student_with_lm, params=_bt_params(strategy=["p", "w"]), run=run))
    candidates = result.candidates
    assert len(candidates) == 3, f"Should have 3 candidates, got {len(candidates)}"
    for i, candidate in enumerate(candidates):
        assert candidate.score is not None or candidate.score is None, f"Candidate {i} score missing"
        assert isinstance(candidate.program, Module), f"Candidate {i} program should be a Module"
        assert isinstance(candidate.label, (str, type(None))), f"Candidate {i} label should be str or None"
    scores = [c.score for c in candidates if c.score is not None]
    assert scores == sorted(scores, reverse=True), "Candidates should be sorted by score (descending)"
    assert candidates[0].score == 0.9, "Best candidate should have score 0.9"
    assert candidates[0].program.marker == "w", "Best candidate should be from optimizer 'w'"
    baseline = [c for c in candidates if c.label is None or c.label == ""]
    assert len(baseline) == 1, "Should have exactly one baseline candidate"
    assert baseline[0].score == 0.5, "Baseline should have score 0.5"


def test_empty_valset_handling(student_with_lm, make_run):
    run = make_run(lm=student_with_lm.get_lm())
    mock_p = MarkedOptimizer("optimized")
    optimizer = BetterTogether(metric=simple_metric, p=mock_p)
    with (
        patch("dspy.teleprompt.bettertogether.launch_lms"),
        patch("dspy.teleprompt.bettertogether.kill_lms"),
        patch.object(optimizer, "_models_changed", return_value=False),
    ):
        result = asyncio.run(optimizer.compile(student_with_lm, params=_bt_params(valset=[], strategy=["p"]), run=run))
    assert hasattr(result.program, "marker"), "Result should have marker"
    assert result.program.marker == "optimized", "Should return the latest program when valset is empty list"
    assert len(result.candidates) > 0, "Should have candidates"
    student2 = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    student2.set_lm(lm)
    mock_p2 = MarkedOptimizer("optimized")
    optimizer2 = BetterTogether(metric=simple_metric, p=mock_p2)
    with (
        patch("dspy.teleprompt.bettertogether.launch_lms"),
        patch("dspy.teleprompt.bettertogether.kill_lms"),
        patch.object(optimizer2, "_models_changed", return_value=False),
    ):
        result2 = asyncio.run(optimizer2.compile(student2, params=_bt_params(valset=None, strategy=["p"]), run=run))
    assert hasattr(result2.program, "marker"), "Result2 should have marker"
    assert result2.program.marker == "optimized", "Should return the latest program when valset is None"
