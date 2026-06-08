import asyncio

from dspy.predict.parallel import Parallel
from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.primitives.module import Module
from dspy.utils.dummies import DummyLM
from tests.task_spec.helpers import ts


def test_parallel_module(make_run):
    lm = DummyLM(
        [
            {"output": "test output 1"},
            {"output": "test output 2"},
            {"output": "test output 3"},
            {"output": "test output 4"},
            {"output": "test output 5"},
        ]
    )
    run = make_run(lm=lm)

    class MyModule(Module):
        def __init__(self):
            super().__init__()
            self.predictor = Predict(ts("input -> output"))
            self.predictor2 = Predict(ts("input -> output"))
            self.parallel = Parallel(num_threads=2)

        async def aforward(self, input, *, run):
            return await self.parallel(
                [
                    (self.predictor, input),
                    (self.predictor2, input),
                    (self.predictor, input),
                    (self.predictor2, input),
                    (self.predictor, input),
                ],
                run=run,
            )

    output = asyncio.run(MyModule()(Example(input="test input").with_inputs("input"), run=run))
    expected_outputs = {f"test output {i}" for i in range(1, 6)}
    assert {r.output for r in output} == expected_outputs


def test_batch_module(make_run):
    lm = DummyLM(
        [
            {"output": "test output 1"},
            {"output": "test output 2"},
            {"output": "test output 3"},
            {"output": "test output 4"},
            {"output": "test output 5"},
        ]
    )
    res_lm = DummyLM(
        [
            {"output": "test output 1", "reasoning": "test reasoning 1"},
            {"output": "test output 2", "reasoning": "test reasoning 2"},
            {"output": "test output 3", "reasoning": "test reasoning 3"},
            {"output": "test output 4", "reasoning": "test reasoning 4"},
            {"output": "test output 5", "reasoning": "test reasoning 5"},
        ]
    )
    run = make_run(lm=lm)

    class MyModule(Module):
        def __init__(self):
            super().__init__()
            self.predictor = Predict(ts("input -> output"))
            self.predictor2 = Predict(ts("input -> output, reasoning"))
            self.parallel = Parallel(num_threads=2)

        async def aforward(self, input, *, run):
            res1 = await self.predictor.batch([input] * 5, run=run)
            reasoning_run = make_run(lm=res_lm)
            res2 = await self.predictor2.batch([input] * 5, run=reasoning_run)
            return (res1, res2)

    result, reason_result = asyncio.run(MyModule()(Example(input="test input").with_inputs("input"), run=run))
    expected_outputs = {f"test output {i}" for i in range(1, 6)}
    assert {r.output for r in result} == expected_outputs
    assert {r.output for r in reason_result} == expected_outputs
    for r in reason_result:
        num = r.output.split()[-1]
        assert r.reasoning == f"test reasoning {num}"


def test_nested_parallel_module(make_run):
    lm = DummyLM(
        [
            {"output": "test output 1"},
            {"output": "test output 2"},
            {"output": "test output 3"},
            {"output": "test output 4"},
            {"output": "test output 5"},
        ]
    )
    run = make_run(lm=lm)

    class MyModule(Module):
        def __init__(self):
            super().__init__()
            self.predictor = Predict(ts("input -> output"))
            self.predictor2 = Predict(ts("input -> output"))
            self.parallel = Parallel(num_threads=2)

        async def aforward(self, input, *, run):
            return await self.parallel(
                [
                    (self.predictor, input),
                    (self.predictor2, input),
                    (self.parallel, [(self.predictor2, input), (self.predictor, input)]),
                ],
                run=run,
            )

    output = asyncio.run(MyModule()(Example(input="test input").with_inputs("input"), run=run))
    assert {output[0].output, output[1].output} <= {f"test output {i}" for i in range(1, 5)}
    assert {output[2][0].output, output[2][1].output} <= {f"test output {i}" for i in range(1, 5)}
    all_outputs = {output[0].output, output[1].output, output[2][0].output, output[2][1].output}
    assert len(all_outputs) == 4


def test_nested_batch_method(make_run):
    lm = DummyLM(
        [
            {"output": "test output 1"},
            {"output": "test output 2"},
            {"output": "test output 3"},
            {"output": "test output 4"},
            {"output": "test output 5"},
        ]
    )
    run = make_run(lm=lm)

    class MyModule(Module):
        def __init__(self):
            super().__init__()
            self.predictor = Predict(ts("input -> output"))

        async def aforward(self, input, *, run):
            return await self.predictor.batch([Example(input=input).with_inputs("input")] * 2, run=run)

    result = asyncio.run(MyModule().batch([Example(input="test input").with_inputs("input")] * 2, run=run))
    assert {result[0][0].output, result[0][1].output, result[1][0].output, result[1][1].output} == {
        "test output 1",
        "test output 2",
        "test output 3",
        "test output 4",
    }


def test_batch_with_failed_examples(make_run):
    run = make_run(lm=DummyLM([{}]))

    class FailingModule(Module):
        async def aforward(self, value: int, *, run) -> str:
            if value == 42:
                raise ValueError("test error")
            return f"success-{value}"

    module = FailingModule()
    examples = [
        Example(value=1).with_inputs("value"),
        Example(value=42).with_inputs("value"),
        Example(value=3).with_inputs("value"),
    ]
    results, failed_examples, exceptions = asyncio.run(
        module.batch(examples, return_failed_examples=True, provide_traceback=True, run=run)
    )
    assert results == ["success-1", None, "success-3"]
    assert len(failed_examples) == 1
    assert failed_examples[0].inputs()["value"] == 42
    assert len(exceptions) == 1
    assert isinstance(exceptions[0], ValueError)
    assert str(exceptions[0]) == "test error"


def test_parallel_timeout_and_straggler_limit_params(make_run):
    parallel_default = Parallel()
    assert parallel_default.timeout == 120
    assert parallel_default.straggler_limit == 3
    parallel_custom = Parallel(timeout=0, straggler_limit=5)
    assert parallel_custom.timeout == 0
    assert parallel_custom.straggler_limit == 5


def test_batch_timeout_and_straggler_limit_params(make_run):

    run = make_run(lm=DummyLM([{}]))

    class SimpleModule(Module):
        async def aforward(self, value: int, *, run) -> int:
            return value * 2

    module = SimpleModule()
    examples = [
        Example(value=1).with_inputs("value"),
        Example(value=2).with_inputs("value"),
        Example(value=3).with_inputs("value"),
    ]
    results = asyncio.run(module.batch(examples, timeout=0, straggler_limit=5, run=run))
    assert results == [2, 4, 6]
