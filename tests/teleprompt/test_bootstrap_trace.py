import asyncio
from typing import Any, ClassVar
from unittest import mock

import pytest

try:
    from litellm import Choices, Message, ModelResponse
except ImportError:
    pytest.skip(reason="litellm is not installed", allow_module_level=True)
from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.lm import LM
from dspy.predict.predict import Predict
from dspy.primitives import Example, Module, Prediction
from dspy.task_spec import input_field, make_task_spec, output_field
from dspy.teleprompt.bootstrap_trace import FailedPrediction, bootstrap_trace_data


def test_bootstrap_trace_data(make_run):
    string_to_int_task_spec = make_task_spec(
        {"text": input_field("text", str, desc="The text."), "number": output_field("number", int, desc="The number.")},
        instructions="Convert a string number to integer",
    )
    program = Predict(string_to_int_task_spec)
    dataset = [
        Example.from_record({"text": "one", "number": 1}, input_keys=("text",)),
        Example.from_record({"text": "two", "number": 2}, input_keys=("text",)),
        Example.from_record({"text": "three", "number": 3}, input_keys=("text",)),
        Example.from_record({"text": "four", "number": 4}, input_keys=("text",)),
        Example.from_record({"text": "five", "number": 5}, input_keys=("text",)),
    ]

    def exact_match_metric(example, prediction, trace=None):
        return example.number == prediction.number

    run = make_run(lm=LM(model="openai/gpt-4o-mini"), adapter=JSONAdapter())
    successful_responses = [
        ModelResponse(
            choices=[Choices(message=Message(content='```json\n{"number": 1}\n```'))], model="openai/gpt-4o-mini"
        ),
        ModelResponse(
            choices=[Choices(message=Message(content='```json\n{"number": 2}\n```'))], model="openai/gpt-4o-mini"
        ),
        ModelResponse(
            choices=[Choices(message=Message(content='```json\n{"number": 3}\n```'))], model="openai/gpt-4o-mini"
        ),
        ModelResponse(
            choices=[Choices(message=Message(content='```json\n{"number": 4}\n```'))], model="openai/gpt-4o-mini"
        ),
    ]

    call_state = {"count": 0}

    def completion_side_effect(*args: object, **kwargs: object):
        call_count = call_state["count"]
        call_state["count"] += 1
        if call_count == 2:
            return ModelResponse(
                choices=[Choices(message=Message(content="This is an invalid JSON!"))], model="openai/gpt-4o-mini"
            )
        return successful_responses[call_count if call_count < 2 else call_count - 1]

    with mock.patch("litellm.acompletion", new=mock.AsyncMock(side_effect=completion_side_effect)):
        results = asyncio.run(
            bootstrap_trace_data(
                program=program,
                dataset=dataset,
                metric=exact_match_metric,
                max_concurrency=1,
                raise_on_error=False,
                capture_failed_parses=True,
                run=run,
            )
        )
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"
    successful_count = 0
    failed_count = 0
    for result in results:
        assert "example" in result
        assert "prediction" in result
        assert "trace" in result
        assert "example_ind" in result
        assert "score" in result
        if isinstance(result["prediction"], FailedPrediction):
            failed_count += 1
            assert hasattr(result["prediction"], "completion_text")
            assert hasattr(result["prediction"], "format_reward")
            assert result["prediction"].completion_text == "This is an invalid JSON!"
        else:
            successful_count += 1
            assert hasattr(result["prediction"], "number")
    assert successful_count == 4, f"Expected 4 successful predictions, got {successful_count}"
    assert failed_count == 1, f"Expected 1 failed prediction, got {failed_count}"
    for result in results:
        assert len(result["trace"]) > 0, "Trace should not be empty"
        for trace_entry in result["trace"]:
            assert len(trace_entry) == 3, "Trace entry should have 3 elements"


def test_bootstrap_trace_data_passes_callback_metadata(monkeypatch, make_run):
    from dspy.teleprompt import bootstrap_trace as bootstrap_trace_module
    from dspy.testing import DummyLM

    run = make_run(lm=DummyLM([{}]))

    class DummyProgram(Module):
        async def _aforward_impl(self, *, run, options=None, **inputs):
            return Prediction()

    captured_metadata: dict[str, Any] = {}

    class DummyEvaluate:
        def __init__(self, *args: object, **kwargs: object):
            pass

        async def __call__(self, *args: object, callback_metadata=None, **kwargs: object):
            captured_metadata["value"] = callback_metadata

            class _Result:
                results: ClassVar[list[Any]] = []

            return _Result()

    def fake_make_optimizer_evaluator(*_args, **_kwargs):
        return DummyEvaluate()

    monkeypatch.setattr(bootstrap_trace_module, "make_optimizer_evaluator", fake_make_optimizer_evaluator)
    asyncio.run(
        bootstrap_trace_module.bootstrap_trace_data(
            program=DummyProgram(), dataset=[], callback_metadata={"disable_logging": True}, run=run
        )
    )
    assert captured_metadata["value"] == {"disable_logging": True}


def test_bootstrap_trace_respects_capture_failed_parses_false(monkeypatch, make_run):
    from dspy.teleprompt import bootstrap_trace as bootstrap_trace_module
    from dspy.testing import DummyLM

    class DummyProgram(Module):
        async def _aforward_impl(self, *, run, options=None, **inputs):
            return Prediction()

    class DummyEvaluate:
        async def __call__(self, *args: object, **kwargs: object):
            class _Result:
                results: ClassVar[list[Any]] = []

            return _Result()

    monkeypatch.setattr(bootstrap_trace_module, "make_optimizer_evaluator", lambda *_args, **_kwargs: DummyEvaluate())

    run = make_run(lm=DummyLM([{}]))
    program = DummyProgram()
    original_impl = object.__getattribute__(program, "_aforward_impl")
    asyncio.run(
        bootstrap_trace_data(
            program=program,
            dataset=[],
            capture_failed_parses=False,
            run=run,
        )
    )
    restored_impl = object.__getattribute__(program, "_aforward_impl")
    assert restored_impl.__func__ is original_impl.__func__
    assert restored_impl.__self__ is original_impl.__self__
