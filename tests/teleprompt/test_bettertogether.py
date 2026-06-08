import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from typing_extensions import override

from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.teleprompt.bettertogether import BetterTogether
from dspy.teleprompt.bootstrap_finetune import BootstrapFinetune
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.teleprompt import Teleprompter
from dspy.utils.dummies import DummyLM
from tests.task_spec.helpers import ts


def simple_metric(example, prediction, trace=None):
    return 1.0 if example.output == prediction.output else 0.0


examples = [
    Example(
        input="What is the oldest known human-made monument?",
        output="Göbekli Tepe in southeastern Turkiye, dating back to around 9600 BCE",
    ).with_inputs("input"),
    Example(input="Why can't fish fall in love?", output="Because love is in the air").with_inputs("input"),
    Example(
        input="What would bring world peace?", output="8 billion people meeting for a tea party in my backyard"
    ).with_inputs("input"),
]
trainset = examples[:2]
valset = [examples[2]]


class SimpleModule(Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    async def aforward(self, **kwargs: object):
        return await self.predictor(**kwargs)


class SimpleOptimizer(Teleprompter):
    @override
    async def compile(self, student, **kwargs: object):
        return student


class MarkedOptimizer(Teleprompter):
    def __init__(self, marker):
        self.marker = marker

    @override
    async def compile(self, student, **kwargs: object):
        prog = SimpleModule(ts("input -> output"))
        prog.marker = self.marker
        return prog


class CapturingOptimizer(Teleprompter):
    def __init__(self):
        self.received_kwargs = {}

    @override
    async def compile(
        self, student, trainset=None, valset=None, teacher=None, num_trials=None, max_bootstrapped_demos=None, **kwargs
    ):
        self.received_kwargs = {
            "trainset": trainset,
            "valset": valset,
            "teacher": teacher,
            "num_trials": num_trials,
            "max_bootstrapped_demos": max_bootstrapped_demos,
            **kwargs,
        }
        return student


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
        BetterTogether(metric=simple_metric, p="not_a_teleprompter")
        raise AssertionError("Should have raised TypeError for invalid optimizer")
    except TypeError as e:
        assert "must be a Teleprompter" in str(e)


def test_strategy_validation(make_run):
    optimizer = BetterTogether(metric=simple_metric)
    valid_strategies = ["p", "w", "p -> w", "w -> p", "p -> w -> p"]
    for strategy in valid_strategies:
        parsed = optimizer._prepare_strategy(strategy)
        assert parsed is not None, f"Failed to parse valid strategy: {strategy}"
    with pytest.raises(ValueError, match="invalid optimizer keys"):
        optimizer._prepare_strategy("p -> x -> w")
    with pytest.raises(ValueError, match="cannot be empty"):
        optimizer._prepare_strategy("")


def test_compile_basic(make_run):
    from dspy.teleprompt.teleprompt import Teleprompter

    student = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "blue"}, {"output": "4"}])
    run = make_run(lm=lm)
    student.set_lm(lm)

    class MockTeleprompter(Teleprompter):
        def __init__(self):
            self.compile_called = False

        @override
        async def compile(self, student, **kwargs: object):
            self.compile_called = True
            return student

    mock_p = MockTeleprompter()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p)
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program", new_callable=AsyncMock) as mock_eval:
        mock_eval.return_value = Mock(score=0.8)
        with patch("dspy.teleprompt.bettertogether.launch_lms"):
            with patch("dspy.teleprompt.bettertogether.kill_lms"):
                compiled = asyncio.run(
                    optimizer.compile(student, trainset=trainset, valset=valset, strategy="p", run=run)
                )
    assert compiled is not None, "Compilation returned None"
    assert hasattr(compiled, "candidate_programs"), "Missing candidate_programs attribute"
    assert hasattr(compiled, "flag_compilation_error_occurred"), "Missing flag_compilation_error_occurred attribute"
    assert mock_p.compile_called, "Mock optimizer compile was not called"


def test_trainset_validation(make_run):
    optimizer = BetterTogether(metric=simple_metric)
    student = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    student.set_lm(lm)
    try:
        asyncio.run(optimizer.compile(student, trainset=[], valset=valset, run=run))
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
        asyncio.run(optimizer.compile(student, trainset=trainset, valset_ratio=1.0, run=run))
        raise AssertionError("Should have raised ValueError for valset_ratio >= 1")
    except ValueError as e:
        assert "must be in range [0, 1)" in str(e)
    try:
        asyncio.run(optimizer.compile(student, trainset=trainset, valset_ratio=-0.1, run=run))
        raise AssertionError("Should have raised ValueError for valset_ratio < 0")
    except ValueError as e:
        assert "must be in range [0, 1)" in str(e)


def test_optimizer_compile_args_validation():
    optimizer = BetterTogether(metric=simple_metric)
    try:
        optimizer._prepare_optimizer_compile_args({"invalid_key": {"num_trials": 10}}, teacher=None)
        raise AssertionError("Should have raised ValueError for invalid optimizer key")
    except ValueError as e:
        assert "invalid optimizer key" in str(e).lower()


def test_student_in_optimizer_compile_args():
    optimizer = BetterTogether(metric=simple_metric)
    try:
        optimizer._validate_compile_args(
            optimizer.optimizers["p"], "p", {"student": SimpleModule(ts("input -> output"))}
        )
        raise AssertionError("Should have raised ValueError for 'student' in compile_args")
    except ValueError as e:
        assert "student" in str(e).lower()
        assert "not allowed" in str(e).lower()


def test_compile_args_passed_to_optimizer(student_with_lm, mock_bt_dependencies, make_run):
    run = make_run(lm=student_with_lm.get_lm())
    mock_eval, _, _ = mock_bt_dependencies
    mock_eval.return_value = Mock(score=0.9)
    mock_p = CapturingOptimizer()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p)
    custom_args = {"num_trials": 20, "max_bootstrapped_demos": 8}
    asyncio.run(
        optimizer.compile(
            student_with_lm,
            trainset=trainset,
            valset=valset,
            strategy="p",
            optimizer_compile_args={"p": custom_args},
            run=run,
        )
    )
    assert mock_p.received_kwargs is not None, "Optimizer compile was not called"
    assert "num_trials" in mock_p.received_kwargs, "num_trials not passed to optimizer"
    assert mock_p.received_kwargs["num_trials"] == 20, "num_trials value incorrect"
    assert "max_bootstrapped_demos" in mock_p.received_kwargs, "max_bootstrapped_demos not passed"
    assert mock_p.received_kwargs["max_bootstrapped_demos"] == 8, "max_bootstrapped_demos value incorrect"


def test_compile_args_multi_optimizer_strategy(make_run):
    from dspy.teleprompt.teleprompt import Teleprompter

    student = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    student.set_lm(lm)

    class PromptOptimizer(Teleprompter):
        def __init__(self):
            self.received_kwargs = {}

        @override
        async def compile(self, student, trainset=None, num_trials=None, **kwargs: object):
            self.received_kwargs = {"trainset": trainset, "num_trials": num_trials, **kwargs}
            return student

    class WeightOptimizer(Teleprompter):
        def __init__(self):
            self.received_kwargs = {}

        @override
        async def compile(self, student, trainset=None, num_batches=None, **kwargs: object):
            self.received_kwargs = {"trainset": trainset, "num_batches": num_batches, **kwargs}
            return student

    mock_p = PromptOptimizer()
    mock_w = WeightOptimizer()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p, w=mock_w)
    compile_args = {"p": {"num_trials": 10}, "w": {"num_batches": 5}}
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program", new_callable=AsyncMock) as mock_eval:
        mock_eval.return_value = Mock(score=0.85)
        with patch("dspy.teleprompt.bettertogether.launch_lms"):
            with patch("dspy.teleprompt.bettertogether.kill_lms"):
                with patch.object(optimizer, "_models_changed", return_value=False):
                    asyncio.run(
                        optimizer.compile(
                            student,
                            trainset=trainset,
                            valset=valset,
                            strategy="p -> w",
                            optimizer_compile_args=compile_args,
                            run=run,
                        )
                    )
    assert mock_p.received_kwargs is not None, "Optimizer 'p' compile was not called"
    assert "num_trials" in mock_p.received_kwargs, "num_trials not passed to optimizer 'p'"
    assert mock_p.received_kwargs["num_trials"] == 10, "num_trials value incorrect for 'p'"
    assert mock_p.received_kwargs.get("num_batches") is None, "Optimizer 'p' should not receive 'w' args"
    assert mock_w.received_kwargs is not None, "Optimizer 'w' compile was not called"
    assert "num_batches" in mock_w.received_kwargs, "num_batches not passed to optimizer 'w'"
    assert mock_w.received_kwargs["num_batches"] == 5, "num_batches value incorrect for 'w'"
    assert mock_w.received_kwargs.get("num_trials") is None, "Optimizer 'w' should not receive 'p' args"


def test_compile_args_override_global_params(make_run):
    from dspy.teleprompt.teleprompt import Teleprompter

    student = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    student.set_lm(lm)

    class CapturingTeleprompter(Teleprompter):
        def __init__(self):
            self.received_kwargs = {}

        @override
        async def compile(self, student, trainset=None, valset=None, teacher=None, **kwargs: object):
            self.received_kwargs = {"trainset": trainset, "valset": valset, "teacher": teacher, **kwargs}
            return student

    mock_p = CapturingTeleprompter()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p)
    override_trainset = [examples[2]]
    override_valset = [examples[0]]
    override_teacher = SimpleModule(ts("input -> output"))
    compile_args = {"p": {"trainset": override_trainset, "valset": override_valset, "teacher": override_teacher}}
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program", new_callable=AsyncMock) as mock_eval:
        mock_eval.return_value = Mock(score=0.9)
        with patch("dspy.teleprompt.bettertogether.launch_lms"):
            with patch("dspy.teleprompt.bettertogether.kill_lms"):
                asyncio.run(
                    optimizer.compile(
                        student,
                        trainset=trainset,
                        valset=valset,
                        teacher=None,
                        strategy="p",
                        optimizer_compile_args=compile_args,
                        run=run,
                    )
                )
    assert mock_p.received_kwargs["trainset"] == override_trainset, (
        "Optimizer should receive override trainset from compile_args"
    )
    assert mock_p.received_kwargs["valset"] == override_valset, (
        "Optimizer should receive override valset from compile_args"
    )
    assert mock_p.received_kwargs["teacher"] is override_teacher, (
        "Optimizer should receive override teacher from compile_args"
    )
    assert mock_p.received_kwargs["trainset"] != trainset, "Override trainset should differ from global trainset"
    assert mock_p.received_kwargs["valset"] != valset, "Override valset should differ from global valset"


def test_trainset_shuffling_between_steps(make_run):
    from dspy.teleprompt.teleprompt import Teleprompter

    student = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    student.set_lm(lm)
    trainsets_received = []

    class TrainsetCapturingOptimizer(Teleprompter):
        @override
        async def compile(self, student, trainset=None, **kwargs: object):
            trainsets_received.append(trainset)
            return student

    mock_p = TrainsetCapturingOptimizer()
    mock_w = TrainsetCapturingOptimizer()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p, w=mock_w)
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program", new_callable=AsyncMock) as mock_eval:
        mock_eval.return_value = Mock(score=0.8)
        with patch("dspy.teleprompt.bettertogether.launch_lms"):
            with patch("dspy.teleprompt.bettertogether.kill_lms"):
                with patch.object(optimizer, "_models_changed", return_value=False):
                    asyncio.run(
                        optimizer.compile(
                            student,
                            trainset=trainset,
                            valset=valset,
                            strategy="p -> w",
                            shuffle_trainset_between_steps=True,
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
    from dspy.teleprompt.teleprompt import Teleprompter

    student = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    student.set_lm(lm)
    execution_log = []

    class LoggingOptimizer(Teleprompter):
        def __init__(self, name):
            self.name = name

        @override
        async def compile(self, student, **kwargs: object):
            optimized = SimpleModule(ts("input -> output"))
            if not hasattr(student, "optimization_path"):
                optimized.optimization_path = [self.name]
            else:
                optimized.optimization_path = student.optimization_path + [self.name]
            execution_log.append((self.name, optimized.optimization_path.copy()))
            return optimized

    mock_p = LoggingOptimizer("p")
    mock_w = LoggingOptimizer("w")
    optimizer = BetterTogether(metric=simple_metric, p=mock_p, w=mock_w)
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program", new_callable=AsyncMock) as mock_eval:
        mock_eval.return_value = Mock(score=0.85)
        with patch("dspy.teleprompt.bettertogether.launch_lms"):
            with patch("dspy.teleprompt.bettertogether.kill_lms"):
                with patch.object(optimizer, "_models_changed", return_value=False):
                    asyncio.run(
                        optimizer.compile(student, trainset=trainset, valset=valset, strategy="p -> w -> p", run=run)
                    )
    assert len(execution_log) == 3, "Should have executed 3 optimization steps"
    assert execution_log[0] == ("p", ["p"]), "First step should be 'p'"
    assert execution_log[1] == ("w", ["p", "w"]), "Second step should be 'w' receiving output from 'p'"
    assert execution_log[2] == ("p", ["p", "w", "p"]), "Third step should be 'p' receiving output from 'w'"


def test_lm_lifecycle_management(make_run):
    from dspy.teleprompt.teleprompt import Teleprompter

    student = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    student.set_lm(lm)

    class SimpleOptimizer(Teleprompter):
        @override
        async def compile(self, student, **kwargs: object):
            return student

    mock_p = SimpleOptimizer()
    mock_w = SimpleOptimizer()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p, w=mock_w)
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program", new_callable=AsyncMock) as mock_eval:
        mock_eval.return_value = Mock(score=0.8)
        with patch("dspy.teleprompt.bettertogether.launch_lms") as mock_launch:
            with patch("dspy.teleprompt.bettertogether.kill_lms") as mock_kill:
                with patch.object(optimizer, "_models_changed", return_value=True):
                    asyncio.run(
                        optimizer.compile(student, trainset=trainset, valset=valset, strategy="p -> w", run=run)
                    )
    assert mock_launch.called, "launch_lms should be called when models change"
    assert mock_kill.called, "kill_lms should be called when models change"


def test_error_handling_returns_best_program(make_run):
    from dspy.teleprompt.teleprompt import Teleprompter

    student = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    student.set_lm(lm)

    class SuccessfulOptimizer(Teleprompter):
        @override
        async def compile(self, student, **kwargs: object):
            optimized = SimpleModule(ts("input -> output"))
            optimized.step_name = "p_success"
            return optimized

    class FailingOptimizer(Teleprompter):
        @override
        async def compile(self, student, **kwargs: object):
            raise RuntimeError("Intentional failure for testing")

    mock_p = SuccessfulOptimizer()
    mock_w = FailingOptimizer()
    optimizer = BetterTogether(metric=simple_metric, p=mock_p, w=mock_w)
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program", new_callable=AsyncMock) as mock_eval:
        mock_eval.side_effect = [Mock(score=0.5), Mock(score=0.7)]
        with patch("dspy.teleprompt.bettertogether.launch_lms"):
            with patch("dspy.teleprompt.bettertogether.kill_lms"):
                with patch.object(optimizer, "_models_changed", return_value=False):
                    result = asyncio.run(
                        optimizer.compile(student, trainset=trainset, valset=valset, strategy="p -> w", run=run)
                    )
    assert result is not None, "Should return a program even if a step fails"
    assert hasattr(result, "flag_compilation_error_occurred"), "Should have error flag"
    assert result.flag_compilation_error_occurred is True, "Error flag should be True"
    assert hasattr(result, "candidate_programs"), "Should have candidate_programs"
    assert len(result.candidate_programs) > 0, "Should have at least one candidate program"


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
        with patch("dspy.teleprompt.bettertogether.launch_lms"):
            with patch("dspy.teleprompt.bettertogether.kill_lms"):
                with patch.object(optimizer, "_models_changed", return_value=False):
                    result = asyncio.run(
                        optimizer.compile(
                            student_with_lm, trainset=trainset, valset=test_valset, strategy="p -> w", run=run
                        )
                    )
    assert hasattr(result, "marker"), "Result should have marker"
    assert result.marker == expected_marker, test_description


def test_candidate_programs_structure(student_with_lm, make_run):
    run = make_run(lm=student_with_lm.get_lm())
    mock_p = MarkedOptimizer("p")
    mock_w = MarkedOptimizer("w")
    optimizer = BetterTogether(metric=simple_metric, p=mock_p, w=mock_w)
    with patch("dspy.teleprompt.bettertogether.eval_candidate_program", new_callable=AsyncMock) as mock_eval:
        mock_eval.side_effect = [Mock(score=0.5), Mock(score=0.8), Mock(score=0.9)]
        with patch("dspy.teleprompt.bettertogether.launch_lms"):
            with patch("dspy.teleprompt.bettertogether.kill_lms"):
                with patch.object(optimizer, "_models_changed", return_value=False):
                    result = asyncio.run(
                        optimizer.compile(student_with_lm, trainset=trainset, valset=valset, strategy="p -> w", run=run)
                    )
    assert hasattr(result, "candidate_programs"), "Result should have candidate_programs attribute"
    candidates = result.candidate_programs
    assert len(candidates) == 3, f"Should have 3 candidates, got {len(candidates)}"
    for i, candidate in enumerate(candidates):
        assert "score" in candidate, f"Candidate {i} missing 'score' key"
        assert "program" in candidate, f"Candidate {i} missing 'program' key"
        assert "strategy" in candidate, f"Candidate {i} missing 'strategy' key"
        assert isinstance(candidate["score"], (int, float)), f"Candidate {i} score should be numeric"
        assert isinstance(candidate["program"], Module), f"Candidate {i} program should be a Module"
        assert isinstance(candidate["strategy"], (str, type(None))), f"Candidate {i} strategy should be str or None"
    scores = [c["score"] for c in candidates]
    assert scores == sorted(scores, reverse=True), "Candidates should be sorted by score (descending)"
    assert candidates[0]["score"] == 0.9, "Best candidate should have score 0.9"
    assert candidates[0]["program"].marker == "w", "Best candidate should be from optimizer 'w'"
    baseline = [c for c in candidates if c["strategy"] is None or c["strategy"] == ""]
    assert len(baseline) == 1, "Should have exactly one baseline candidate"
    assert baseline[0]["score"] == 0.5, "Baseline should have score 0.5"


def test_empty_valset_handling(student_with_lm, make_run):
    run = make_run(lm=student_with_lm.get_lm())
    mock_p = MarkedOptimizer("optimized")
    optimizer = BetterTogether(metric=simple_metric, p=mock_p)
    with patch("dspy.teleprompt.bettertogether.launch_lms"), patch("dspy.teleprompt.bettertogether.kill_lms"):
        with patch.object(optimizer, "_models_changed", return_value=False):
            result = asyncio.run(
                optimizer.compile(student_with_lm, trainset=trainset, valset=[], strategy="p", run=run)
            )
    assert hasattr(result, "marker"), "Result should have marker"
    assert result.marker == "optimized", "Should return the latest program when valset is empty list"
    assert hasattr(result, "candidate_programs"), "Should have candidate_programs"
    student2 = SimpleModule(ts("input -> output"))
    lm = DummyLM([{"output": "test"}])
    run = make_run(lm=lm)
    student2.set_lm(lm)
    mock_p2 = MarkedOptimizer("optimized")
    optimizer2 = BetterTogether(metric=simple_metric, p=mock_p2)
    with patch("dspy.teleprompt.bettertogether.launch_lms"), patch("dspy.teleprompt.bettertogether.kill_lms"):
        with patch.object(optimizer2, "_models_changed", return_value=False):
            result2 = asyncio.run(optimizer2.compile(student2, trainset=trainset, valset=None, strategy="p", run=run))
    assert hasattr(result2, "marker"), "Result2 should have marker"
    assert result2.marker == "optimized", "Should return the latest program when valset is None"
