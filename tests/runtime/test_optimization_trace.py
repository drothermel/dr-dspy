import asyncio

from dspy.adapters.json_adapter import JSONAdapter
from dspy.errors import AdapterParseError
from dspy.predict.predict import Predict
from dspy.primitives import Example, Module, Prediction
from dspy.runtime import run_with_trace
from dspy.runtime.optimization_trace import FailedPrediction, TraceCapturingModule
from dspy.task_spec import input_field, make_task_spec, output_field
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts


class _TraceProgram(Module):
    def __init__(self, predictor: Predict) -> None:
        super().__init__()
        self.predictor = predictor

    async def _aforward_impl(self, *, run, options=None, **inputs):
        return await self.predictor(**inputs, run=run, options=options)


def test_run_with_trace_isolates_parent_trace(make_run):
    predictor = Predict(ts("input -> output"))
    program = _TraceProgram(predictor)
    run = make_run(lm=DummyLM([{"output": "ok"}]))
    run.optimization_trace.append(("parent",))

    async def _run():
        _prediction, trace = await run_with_trace(
            program,
            Example.from_record({"input": "x"}, input_keys=("input",)),
            run,
        )
        return trace

    trace = asyncio.run(_run())
    assert len(trace) == 1
    assert run.optimization_trace == [("parent",)]


def test_trace_capturing_module_delegates_predictors(make_run):
    predictor = Predict(ts("input -> output"))
    inner = _TraceProgram(predictor)
    wrapper = TraceCapturingModule(inner)
    assert wrapper.predictors() == inner.predictors()
    assert wrapper.named_predictors() == inner.named_predictors()


def test_trace_capturing_module_does_not_mutate_inner_forward(make_run):
    run = make_run(lm=DummyLM([{}]))

    class DummyProgram(Module):
        async def _aforward_impl(self, *, run, options=None, **inputs):
            return Prediction()

    inner = DummyProgram()
    original_impl = object.__getattribute__(inner, "_aforward_impl")
    wrapper = TraceCapturingModule(inner)

    async def _run():
        await run_with_trace(wrapper, {}, run)

    asyncio.run(_run())
    restored_impl = object.__getattribute__(inner, "_aforward_impl")
    assert restored_impl.__func__ is original_impl.__func__
    assert restored_impl.__self__ is original_impl.__self__


def test_run_with_trace_capture_parse_failures(make_run, monkeypatch):
    string_to_int_task_spec = make_task_spec(
        {"text": input_field("text", str, desc="The text."), "number": output_field("number", int, desc="The number.")},
        instructions="Convert a string number to integer",
    )
    program = Predict(string_to_int_task_spec)
    run = make_run(lm=DummyLM([{}]), adapter=JSONAdapter())

    async def _raise_parse(*_args, **_kwargs):
        raise AdapterParseError(
            adapter_name="JSONAdapter",
            lm_response="invalid json",
            parsed_result={"number": 1},
            task_spec=string_to_int_task_spec,
        )

    monkeypatch.setattr(program, "aforward", _raise_parse)

    async def _run():
        return await run_with_trace(
            program,
            Example.from_record({"text": "one"}, input_keys=("text",)),
            run,
            capture_parse_failures=True,
            format_failure_score=-1.0,
            failure_score=0.0,
        )

    prediction, trace = asyncio.run(_run())
    assert isinstance(prediction, FailedPrediction)
    assert prediction.completion_text == "invalid json"
    assert len(trace) == 1
    assert trace[0][2] is prediction
