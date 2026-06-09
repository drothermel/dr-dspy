import asyncio

from dspy.predict.predict import Predict
from dspy.primitives import Example, Module
from dspy.runtime.batch import Parallel
from dspy.testing import DummyLM
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
            self.parallel = Parallel(max_concurrency=2)

        async def _aforward_impl(self, *, run, options=None, **inputs):
            example = {"input": inputs["input"]}
            return (
                await self.parallel(
                    [
                        (self.predictor, example),
                        (self.predictor2, example),
                        (self.predictor, example),
                        (self.predictor2, example),
                        (self.predictor, example),
                    ],
                    run=run,
                )
            ).results

    output = asyncio.run(MyModule()(input="test input", run=run))
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
            self.parallel = Parallel(max_concurrency=2)

        async def _aforward_impl(self, *, run, options=None, **inputs):
            examples = [Example.from_record({"input": inputs["input"]}, input_keys=("input",))] * 5
            res1 = (await self.predictor.batch(examples, run=run)).results
            reasoning_run = make_run(lm=res_lm)
            res2 = (await self.predictor2.batch(examples, run=reasoning_run)).results
            return (res1, res2)

    result, reason_result = asyncio.run(MyModule()(input="test input", run=run))
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
            self.parallel = Parallel(max_concurrency=2)

        async def _aforward_impl(self, *, run, options=None, **inputs):
            example = {"input": inputs["input"]}
            return (
                await self.parallel(
                    [
                        (self.predictor, example),
                        (self.predictor2, example),
                        (self.parallel, [(self.predictor2, example), (self.predictor, example)]),
                    ],
                    run=run,
                )
            ).results

    output = asyncio.run(MyModule()(input="test input", run=run))
    assert {output[0].output, output[1].output} <= {f"test output {i}" for i in range(1, 5)}
    assert {output[2].results[0].output, output[2].results[1].output} <= {f"test output {i}" for i in range(1, 5)}
    all_outputs = {output[0].output, output[1].output, output[2].results[0].output, output[2].results[1].output}
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

        async def _aforward_impl(self, *, run, options=None, **inputs):
            input_value = inputs["input"]
            return (
                await self.predictor.batch(
                    [Example.from_record({"input": input_value}, input_keys=("input",))] * 2,
                    run=run,
                )
            ).results

    result = asyncio.run(
        MyModule().batch([Example.from_record({"input": "test input"}, input_keys=("input",))] * 2, run=run)
    ).results
    assert {result[0][0].output, result[0][1].output, result[1][0].output, result[1][1].output} == {
        "test output 1",
        "test output 2",
        "test output 3",
        "test output 4",
    }


def test_batch_with_failed_examples(make_run):
    run = make_run(lm=DummyLM([{}]))

    class FailingModule(Module):
        async def _aforward_impl(self, *, run, options=None, **inputs) -> str:
            value = inputs["value"]
            if value == 42:
                raise ValueError("test error")
            return f"success-{value}"

    module = FailingModule()
    examples = [
        Example.from_record({"value": 1}, input_keys=("value",)),
        Example.from_record({"value": 42}, input_keys=("value",)),
        Example.from_record({"value": 3}, input_keys=("value",)),
    ]
    batch_result = asyncio.run(module.batch(examples, provide_traceback=True, run=run))
    assert batch_result.results == ("success-1", None, "success-3")
    assert len(batch_result.failures) == 1
    assert batch_result.failures[0].input["value"] == 42
    assert isinstance(batch_result.failures[0].exception, ValueError)
    assert str(batch_result.failures[0].exception) == "test error"


def test_parallel_timeout_disabled_when_zero(make_run):
    parallel_default = Parallel()
    assert parallel_default.timeout == 120
    parallel_custom = Parallel(timeout=0)
    assert parallel_custom.timeout == 0


def test_batch_timeout_disabled_when_zero(make_run):
    run = make_run(lm=DummyLM([{}]))

    class SimpleModule(Module):
        async def _aforward_impl(self, *, run, options=None, **inputs) -> int:
            return inputs["value"] * 2

    module = SimpleModule()
    examples = [
        Example.from_record({"value": 1}, input_keys=("value",)),
        Example.from_record({"value": 2}, input_keys=("value",)),
        Example.from_record({"value": 3}, input_keys=("value",)),
    ]
    results = asyncio.run(module.batch(examples, timeout=0, run=run)).results
    assert results == (2, 4, 6)


def test_parallel_timeout_records_slow_item_failure(make_run):
    run = make_run(lm=DummyLM([{}]))

    class SlowModule(Module):
        async def _aforward_impl(self, *, run, options=None, **inputs) -> str:
            if inputs["value"] == "slow":
                await asyncio.sleep(2)
            return inputs["value"]

    module = SlowModule()
    parallel = Parallel(run=run, timeout=1, max_concurrency=2)
    batch_result = asyncio.run(
        parallel(
            [
                (module, {"value": "fast"}),
                (module, {"value": "slow"}),
            ],
            run=run,
        )
    )
    assert batch_result.results[0] == "fast"
    assert batch_result.results[1] is None
    assert len(batch_result.failures) == 1
    assert batch_result.failures[0].input == {"value": "slow"}
    assert isinstance(batch_result.failures[0].exception, TimeoutError)
