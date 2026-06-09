import asyncio

from dspy.predict.chain_of_thought import ChainOfThought
from dspy.primitives import Example, Module
from dspy.teleprompt.compile_params import COPROCompileParams, EvaluateCompileParams
from dspy.teleprompt.copro_optimizer import COPRO
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts


def simple_metric(example, prediction, trace=None):
    return example.output == prediction.output


trainset = [
    Example.from_record({"input": "Question: What is the color of the sky?", "output": "blue"}, input_keys=("input",)),
    Example.from_record(
        {"input": "Question: What does the fox say?", "output": "Ring-ding-ding-ding-dingeringeding!"},
        input_keys=("input",),
    ),
]


def test_signature_optimizer_initialization(make_run):
    optimizer = COPRO(metric=simple_metric, breadth=2, depth=1, init_temperature=1.4)
    assert optimizer.metric == simple_metric, "Metric not correctly initialized"
    assert optimizer.breadth == 2, "Breadth not correctly initialized"
    assert optimizer.depth == 1, "Depth not correctly initialized"
    assert optimizer.init_temperature == 1.4, "Initial temperature not correctly initialized"


class SimpleModule(Module):
    def __init__(self, task_spec):
        super().__init__()
        self.predictor = ChainOfThought(task_spec)

    async def _aforward_impl(self, *, run, options=None, **inputs):
        return await self.predictor(run=run, options=options, **inputs)


def test_signature_optimizer_optimization_process(make_run):
    optimizer = COPRO(metric=simple_metric, breadth=2, depth=1, init_temperature=1.4)
    run = make_run(
        lm=DummyLM(
            [
                {
                    "proposed_instruction": "Optimized instruction 1",
                    "proposed_prefix_for_output_field": "Optimized instruction 2",
                }
            ]
        )
    )
    student = SimpleModule(ts("input -> output"))
    result = asyncio.run(
        optimizer.compile(
            student,
            params=COPROCompileParams(
                trainset=trainset,
                evaluate=EvaluateCompileParams(max_concurrency=1, display_progress=False),
            ),
            run=run,
        )
    )
    assert result.program is not student, "Optimization did not modify the student"


def test_signature_optimizer_statistics_tracking(make_run):
    optimizer = COPRO(metric=simple_metric, breadth=2, depth=1, init_temperature=1.4)
    optimizer.track_stats = True
    run = make_run(
        lm=DummyLM(
            [
                {
                    "proposed_instruction": "Optimized instruction 1",
                    "proposed_prefix_for_output_field": "Optimized instruction 2",
                }
            ]
        )
    )
    student = SimpleModule(ts("input -> output"))
    result = asyncio.run(
        optimizer.compile(
            student,
            params=COPROCompileParams(
                trainset=trainset,
                evaluate=EvaluateCompileParams(max_concurrency=1, display_progress=False),
            ),
            run=run,
        )
    )
    assert result.stats.metric_calls > 0, "Total calls statistic not tracked"
    assert result.stats.copro_depth_stats is not None, "Best results statistics not tracked"


def test_optimization_and_output_verification(make_run):
    lm = DummyLM(
        [
            {"proposed_instruction": "Optimized Prompt", "proposed_prefix_for_output_field": "Optimized Prefix"},
            {"reasoning": "france", "output": "Paris"},
            {"reasoning": "france", "output": "Paris"},
            {"reasoning": "france", "output": "Paris"},
            {"reasoning": "france", "output": "Paris"},
            {"reasoning": "france", "output": "Paris"},
            {"reasoning": "france", "output": "Paris"},
            {"reasoning": "france", "output": "Paris"},
        ]
    )
    run = make_run(lm=lm)
    optimizer = COPRO(metric=simple_metric, breadth=2, depth=1, init_temperature=1.4)
    student = SimpleModule(ts("input -> output"))
    result = asyncio.run(
        optimizer.compile(
            student,
            params=COPROCompileParams(
                trainset=trainset,
                evaluate=EvaluateCompileParams(max_concurrency=1, display_progress=False),
            ),
            run=run,
        )
    )
    test_input = "What is the capital of France?"
    prediction = asyncio.run(result.program(input=test_input, run=run))
    assert prediction.output == "Paris"


def test_statistics_tracking_during_optimization(make_run):
    run = make_run(
        lm=DummyLM(
            [{"proposed_instruction": "Optimized Prompt", "proposed_prefix_for_output_field": "Optimized Prefix"}]
        )
    )
    optimizer = COPRO(metric=simple_metric, breadth=2, depth=1, init_temperature=1.4)
    optimizer.track_stats = True
    student = SimpleModule(ts("input -> output"))
    result = asyncio.run(
        optimizer.compile(
            student,
            params=COPROCompileParams(
                trainset=trainset,
                evaluate=EvaluateCompileParams(max_concurrency=1, display_progress=False),
            ),
            run=run,
        )
    )
    assert result.stats.metric_calls > 0, "Optimizer did not track total metric calls"
    assert result.stats.copro_depth_stats is not None, "Optimizer did not track depth statistics"
    depth_stats = result.stats.copro_depth_stats
    assert "results_best" in depth_stats, "Optimizer did not track the best results"
    assert "results_latest" in depth_stats, "Optimizer did not track the latest results"
    assert len(depth_stats["results_best"]) > 0, "Optimizer did not properly populate the best results statistics"
    assert len(depth_stats["results_latest"]) > 0, "Optimizer did not properly populate the latest results statistics"
