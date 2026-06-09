import asyncio
import logging
import os
import threading
from typing import Any, cast
from unittest.mock import patch

import pytest

from dspy.runtime import CallLogMode, TelemetryConfig

try:
    from litellm import Choices, Message, ModelResponse
    from litellm.types.utils import Usage
except ImportError:
    pytest.skip(reason="litellm is not installed", allow_module_level=True)
from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.lm import LM
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.predict import Predict
from dspy.primitives import Module, Prediction
from dspy.runtime.batch import Parallel
from dspy.task_spec import default_task_instructions
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts

QUESTION_ANSWER_TASK_SPEC = ts(
    "question -> answer", instructions=default_task_instructions(inputs=("question",), outputs=("answer",))
)


def _module_attrs(module: Module) -> Any:
    return module


class HopModule(Module):
    def __init__(self):
        super().__init__()
        self.predict1 = Predict(
            ts("question -> query", instructions=default_task_instructions(inputs=("question",), outputs=("query",)))
        )
        self.predict2 = Predict(
            ts("query -> answer", instructions=default_task_instructions(inputs=("query",), outputs=("answer",)))
        )

    async def _aforward_impl(self, question, **kwargs):
        run = kwargs.get("run")
        query = (await self.predict1(question=question, run=run)).query
        return await self.predict2(query=query, run=run)


def test_module_initialization(make_run):
    module = Module()
    assert module._compiled is False, "Module _compiled attribute should be False upon initialization"


def test_named_predictors(make_run):
    module = HopModule()
    named_preds = module.named_predictors()
    assert len(named_preds) == 2, "Should identify correct number of Predict instances"
    names, _preds = zip(*named_preds, strict=False)
    assert "self.predict1" in names and "self.predict2" in names, (
        "Named predictors should include 'self.predict1' and 'self.predict2'"
    )


def test_predictors(make_run):
    module = HopModule()
    preds = module.predictors()
    assert len(preds) == 2, "Should return correct number of Predict instances"
    assert all(isinstance(p, Predict) for p in preds), "All returned items should be instances of Predict"


def test_forward(make_run):
    program = HopModule()
    run = make_run(lm=DummyLM({"What is 1+1?": {"query": "let me check"}, "let me check": {"answer": "2"}}))
    result = asyncio.run(program(question="What is 1+1?", run=run)).answer
    assert result == "2"


def test_nested_named_predictors():

    class Hop2Module(Module):
        def __init__(self):
            super().__init__()
            self.hop = HopModule()

    module = Hop2Module()
    named_preds = module.named_predictors()
    assert len(named_preds) == 2
    names, _preds = zip(*named_preds, strict=False)
    assert "self.hop.predict1" in names
    assert "self.hop.predict2" in names


def test_empty_module():
    module = Module()
    assert list(module.named_sub_modules()) == [("self", module)]


def test_single_level():
    module = Module()
    _module_attrs(module).sub = Module()
    expected = [("self", module), ("self.sub", _module_attrs(module).sub)]
    assert list(module.named_sub_modules()) == expected


def test_multiple_levels():
    module = Module()
    _module_attrs(module).sub = Module()
    _module_attrs(module).sub.subsub = Module()
    expected = [
        ("self", module),
        ("self.sub", _module_attrs(module).sub),
        ("self.sub.subsub", _module_attrs(module).sub.subsub),
    ]
    assert list(module.named_sub_modules()) == expected


def test_multiple_sub_modules():
    module = Module()
    _module_attrs(module).sub1 = Module()
    _module_attrs(module).sub2 = Module()
    expected = [
        ("self", module),
        ("self.sub1", _module_attrs(module).sub1),
        ("self.sub2", _module_attrs(module).sub2),
    ]
    assert sorted(module.named_sub_modules()) == sorted(expected)


def test_named_sub_modules_skips_non_modules():
    module = Module()
    _module_attrs(module).sub = Module()
    _module_attrs(module).not_a_sub = "Not a self"
    expected = [("self", module), ("self.sub", _module_attrs(module).sub)]
    assert list(module.named_sub_modules()) == expected


def test_complex_module_traversal():
    root = Module()
    root_attrs = _module_attrs(root)
    sub_attrs = _module_attrs(Module())
    root_attrs.sub_module = sub_attrs
    sub_attrs.nested_list = [Module(), {"key": Module()}]
    sub_attrs.nested_tuple = (Module(), [Module(), Module()])
    expected_names = {
        "self",
        "self.sub_module",
        "self.sub_module.nested_list[0]",
        "self.sub_module.nested_list[1][key]",
        "self.sub_module.nested_tuple[0]",
        "self.sub_module.nested_tuple[1][0]",
        "self.sub_module.nested_tuple[1][1]",
    }
    found_names = {name for name, _ in root.named_sub_modules()}
    assert found_names == expected_names, (
        f"Missing or extra modules found. Missing: {expected_names - found_names}, Extra: {found_names - expected_names}"
    )


def test_complex_module_traversal_with_same_module():
    root = Module()
    root_attrs = _module_attrs(root)
    sub_attrs = _module_attrs(Module())
    root_attrs.sub_module = sub_attrs
    sub_attrs.nested_list = [Module(), {"key": Module()}]
    same_module = Module()
    sub_attrs.nested_tuple = (Module(), [same_module, same_module])
    expected_names = {
        "self",
        "self.sub_module",
        "self.sub_module.nested_list[0]",
        "self.sub_module.nested_list[1][key]",
        "self.sub_module.nested_tuple[0]",
        "self.sub_module.nested_tuple[1][0]",
    }
    found_names = {name for name, _ in root.named_sub_modules()}
    assert found_names == expected_names, (
        f"Missing or extra modules found. Missing: {expected_names - found_names}, Extra: {found_names - expected_names}"
    )


def test_named_predictors_traverses_nested_containers():
    root = Module()
    root_attrs = _module_attrs(root)
    sub_attrs = _module_attrs(Module())
    root_attrs.sub_module = sub_attrs
    sub_attrs.nested_predict = Predict(ts("question -> answer", instructions="Answer the question."))
    sub_attrs.nested_list = [Predict(ts("question -> answer", instructions="Answer the question."))]
    sub_attrs.nested_dict = {"key": Predict(ts("question -> answer", instructions="Answer the question."))}
    found_names = {name for name, _ in root.named_predictors()}
    assert "self.sub_module.nested_predict" in found_names
    assert "self.sub_module.nested_list[0]" in found_names
    assert "self.sub_module.nested_dict[key]" in found_names


class DuplicateModule(Module):
    def __init__(self):
        super().__init__()
        self.p0 = Predict(ts("question -> answer", instructions="Answer the question."))
        self.p1 = self.p0


def test_named_predictors_duplicate_references():
    module = DuplicateModule()
    named = module.named_predictors()
    assert len(named) == 1
    assert len(module.predictors()) == 1
    assert named[0][1] is module.p0
    assert named[0][1] is module.p1
    assert named[0][0] in {"self.p0", "self.p1"}


def test_compiled_subgraph_is_opaque():
    root = Module()
    root_attrs = _module_attrs(root)
    compiled_child = Module()
    compiled_child._compiled = True
    inner_predict = Predict(ts("question -> answer", instructions="Answer the question."))
    _module_attrs(compiled_child).inner = inner_predict
    root_attrs.compiled_child = compiled_child
    root_attrs.top_predict = Predict(ts("question -> answer", instructions="Answer the question."))

    predictor_names = {name for name, _ in root.named_predictors()}
    submodule_names = {name for name, _ in root.named_sub_modules()}

    assert "self.top_predict" in predictor_names
    assert "self.compiled_child.inner" not in predictor_names
    assert "self.compiled_child" in submodule_names
    assert "self.compiled_child.inner" not in submodule_names


# --- deepcopy / reset_copy ---


def test_deepcopy_basic():
    cot = ChainOfThought(ts("q -> a"))
    cot_copy = cot.deepcopy()
    assert len(cot.predictors()) == len(cot_copy.predictors())
    assert id(cot.predictors()[0]) != id(cot_copy.predictors()[0])
    assert cot.predictors()[0].__dict__ == cot_copy.predictors()[0].__dict__


def test_deepcopy_with_uncopyable_modules(make_run):

    class CustomClass(Module):
        def __init__(self):
            self.lock = threading.Lock()
            self.cot = ChainOfThought(ts("q -> a"))

    model = CustomClass()
    model_copy = model.deepcopy()
    assert len(model.predictors()) == len(model_copy.predictors())
    assert id(model.lock) == id(model_copy.lock)
    assert id(model.predictors()[0]) != id(model_copy.predictors()[0])
    assert model.predictors()[0].__dict__ == model_copy.predictors()[0].__dict__


def test_deepcopy_with_nested_modules(make_run):

    class CustomClass1(Module):
        def __init__(self):
            self.lock = threading.Lock()
            self.cot = ChainOfThought(ts("q -> a"))

    class CustomClass2(Module):
        def __init__(self):
            self.submodel = CustomClass1()

    model = CustomClass2()
    model_copy = model.deepcopy()
    assert len(model.predictors()) == len(model_copy.predictors())
    assert id(model.submodel.lock) == id(model_copy.submodel.lock)
    assert id(model.predictors()[0]) != id(model_copy.predictors()[0])
    assert model.predictors()[0].__dict__ == model_copy.predictors()[0].__dict__


# --- usage tracking ---


@pytest.mark.llm_call
def test_single_module_call_with_usage_tracker(lm_for_test, make_run):
    run = make_run(lm=LM(lm_for_test, temperature=0.0), telemetry=TelemetryConfig(track_usage=True))
    predict = ChainOfThought(ts("question -> answer"))
    output = asyncio.run(predict(question="What is the capital of France?", run=run))
    lm_usage = output.get_lm_usage()
    assert lm_usage is not None
    assert len(lm_usage) == 1
    assert lm_usage[lm_for_test]["prompt_tokens"] > 0
    assert lm_usage[lm_for_test]["completion_tokens"] > 0
    assert lm_usage[lm_for_test]["total_tokens"] > 0


@pytest.mark.llm_call
def test_multi_module_call_with_usage_tracker(lm_for_test, make_run):
    run = make_run(lm=LM(lm_for_test, temperature=0.0), telemetry=TelemetryConfig(track_usage=True))

    class MyProgram(Module):
        def __init__(self):
            super().__init__()
            self.predict1 = ChainOfThought(ts("question -> answer"))
            self.predict2 = ChainOfThought(ts("question, answer -> score"))

        async def _aforward_impl(self, question: str, **kwargs: object) -> Prediction:
            call_run = kwargs["run"]
            answer = await self.predict1(question=question, run=call_run)
            return await self.predict2(question=question, answer=answer, run=call_run)

    program = MyProgram()
    output = asyncio.run(program(question="What is the capital of France?", run=run))
    lm_usage = output.get_lm_usage()
    assert lm_usage is not None
    assert len(lm_usage) == 1
    assert lm_usage[lm_for_test]["prompt_tokens"] > 0
    assert lm_usage[lm_for_test]["completion_tokens"] > 0
    assert lm_usage[lm_for_test]["total_tokens"] > 0


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Skip the test if OPENAI_API_KEY is not set.")
def test_usage_tracker_in_parallel(make_run):

    class MyProgram(Module):
        def __init__(self, lm):
            self.lm = lm
            self.predict1 = ChainOfThought(ts("question -> answer"))
            self.predict2 = ChainOfThought(ts("question, answer -> score"))

        async def _aforward_impl(self, question: str, *, run) -> Prediction:
            answer = await self.predict1(question=question, run=run)
            return await self.predict2(question=question, answer=answer, run=run)

    program1 = MyProgram(lm=LM("openai/gpt-4o-mini"))
    program2 = MyProgram(lm=LM("openai/gpt-3.5-turbo"))
    parallelizer = Parallel()
    run = make_run(lm=LM("openai/gpt-4o-mini"))
    results = asyncio.run(
        parallelizer(
            [
                (program1, {"question": "What is the meaning of life?"}),
                (program2, {"question": "why did a chicken cross the kitchen?"}),
            ],
            run=run,
        )
    ).results
    typed_results = cast("list[Prediction]", results)
    usage0 = typed_results[0].get_lm_usage()
    usage1 = typed_results[1].get_lm_usage()
    assert usage0 is not None
    assert usage1 is not None
    assert usage0.keys() == {"openai/gpt-4o-mini"}
    assert usage1.keys() == {"openai/gpt-3.5-turbo"}


@pytest.mark.asyncio
async def test_usage_tracker_async_parallel(make_run):
    program = Predict(QUESTION_ANSWER_TASK_SPEC)
    with patch("litellm.acompletion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'answer': 'Paris'}"))],
            usage=Usage(
                prompt_tokens=1117,
                completion_tokens=46,
                total_tokens=1163,
                prompt_tokens_details={"cached_tokens": 0, "audio_tokens": 0},
                completion_tokens_details={
                    "reasoning_tokens": 0,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
            ),
            model="openai/gpt-4o-mini",
        )
        run = make_run(lm=LM("openai/gpt-4o-mini"), adapter=JSONAdapter(), telemetry=TelemetryConfig(track_usage=True))
        coroutines = [
            program(question="What is the capital of France?", run=run),
            program(question="What is the capital of France?", run=run),
            program(question="What is the capital of France?", run=run),
            program(question="What is the capital of France?", run=run),
        ]
        results = await asyncio.gather(*coroutines)
        assert results[0].get_lm_usage() is not None
        assert results[1].get_lm_usage() is not None
        lm_usage0 = results[0].get_lm_usage()["openai/gpt-4o-mini"]
        lm_usage1 = results[1].get_lm_usage()["openai/gpt-4o-mini"]
        assert lm_usage0["prompt_tokens"] == 1117
        assert lm_usage1["prompt_tokens"] == 1117
        assert lm_usage0["completion_tokens"] == 46
        assert lm_usage1["completion_tokens"] == 46
        assert lm_usage0["total_tokens"] == 1163
        assert lm_usage1["total_tokens"] == 1163


def test_usage_tracker_no_side_effect(make_run):

    class MyProgram(Module):
        def __init__(self):
            super().__init__()
            self.predict = Predict(QUESTION_ANSWER_TASK_SPEC)

        async def _aforward_impl(self, question: str, **kwargs: object) -> str:
            run = kwargs["run"]
            return (await self.predict(question=question, run=run)).answer

    program = MyProgram()
    run = make_run(lm=DummyLM([{"answer": "Paris"}]), telemetry=TelemetryConfig(track_usage=True))
    result = asyncio.run(program(question="What is the capital of France?", run=run))
    assert result == "Paris"


def test_module_history(make_run):

    class MyProgram(Module):
        def __init__(self):
            super().__init__()
            self.cot = ChainOfThought(ts("question -> answer"))

        async def _aforward_impl(self, question: str, **kwargs: object) -> Prediction:
            run = kwargs["run"]
            return await self.cot(question=question, run=run)

    with patch("litellm.acompletion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(message=Message(content="{'reasoning': 'Paris is the capital of France', 'answer': 'Paris'}"))
            ],
            model="openai/gpt-4o-mini",
        )
        run = make_run(lm=LM("openai/gpt-4o-mini"), adapter=JSONAdapter())
        program = MyProgram()
        asyncio.run(program(question="What is the capital of France?", run=run))
        asyncio.run(program.cot(question="What is the capital of France?", run=run))
        assert len(program.call_log) == 1
        assert len(program.cot.call_log) == 2
        assert len(program.cot.predict.call_log) == 2
        assert id(program.call_log[0]) == id(program.cot.call_log[0])
        assert program.call_log[0].outputs == ["{'reasoning': 'Paris is the capital of France', 'answer': 'Paris'}"]
        asyncio.run(program(question="What is the capital of France?", run=run))
        assert len(program.call_log) == 2
        assert len(program.cot.call_log) == 3
        assert len(program.cot.predict.call_log) == 3
        asyncio.run(program(question="What is the capital of France?", run=run))
        assert len(program.call_log) == 3
        assert len(program.cot.call_log) == 4
        assert len(program.cot.predict.call_log) == 4


def test_module_history_with_concurrency(make_run):

    class MyProgram(Module):
        def __init__(self):
            super().__init__()
            self.cot = ChainOfThought(ts("question -> answer"))

        async def _aforward_impl(self, question: str, **kwargs: object) -> Prediction:
            run = kwargs["run"]
            return await self.cot(question=question, run=run)

    with patch("litellm.acompletion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'reasoning': 'N/A', 'answer': 'Holy crab!'}"))],
            model="openai/gpt-4o-mini",
        )
        run = make_run(lm=LM("openai/gpt-4o-mini"), adapter=JSONAdapter())
        program = MyProgram()

        async def run_concurrent():
            await asyncio.gather(
                program(question="What is the meaning of life?", run=run),
                program(question="why did a chicken cross the kitchen?", run=run),
            )

        asyncio.run(run_concurrent())
        assert len(program.call_log) == 2
        assert len(program.cot.call_log) == 2
        assert len(program.cot.predict.call_log) == 2


@pytest.mark.asyncio
async def test_module_history_async(make_run):

    class MyProgram(Module):
        def __init__(self):
            super().__init__()
            self.cot = ChainOfThought(ts("question -> answer"))

        async def _aforward_impl(self, question: str, **kwargs: object) -> Prediction:
            run = kwargs["run"]
            return await self.cot(question=question, run=run)

    with patch("litellm.acompletion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(message=Message(content="{'reasoning': 'Paris is the capital of France', 'answer': 'Paris'}"))
            ],
            model="openai/gpt-4o-mini",
        )
        program = MyProgram()
        run = make_run(lm=LM("openai/gpt-4o-mini"), adapter=JSONAdapter())
        await program(question="What is the capital of France?", run=run)
        await program.cot(question="What is the capital of France?", run=run)
        assert len(program.call_log) == 1
        assert len(program.cot.call_log) == 2
        assert len(program.cot.predict.call_log) == 2
        assert id(program.call_log[0]) == id(program.cot.call_log[0])
        assert program.call_log[0].outputs == ["{'reasoning': 'Paris is the capital of France', 'answer': 'Paris'}"]
        run = make_run(
            lm=LM("openai/gpt-4o-mini"), adapter=JSONAdapter(), telemetry=TelemetryConfig(call_log=CallLogMode.off)
        )
        await program(question="What is the capital of France?", run=run)
        assert len(program.call_log) == 1
        assert len(program.cot.call_log) == 2
        assert len(program.cot.predict.call_log) == 2
        run = make_run(
            lm=LM("openai/gpt-4o-mini"),
            adapter=JSONAdapter(),
            telemetry=TelemetryConfig(call_log=CallLogMode.memory),
        )
        fresh_program = MyProgram()
        await fresh_program(question="What is the capital of France?", run=run)
        assert len(fresh_program.call_log) == 1
        assert len(fresh_program.cot.call_log) == 1
        assert len(fresh_program.cot.predict.call_log) == 1


# --- aforward warning ---


def test_forward_direct_call_warning(caplog, make_run):

    class TestModule(Module):
        async def _aforward_impl(self, x, **kwargs: object):
            return x

    module = TestModule()
    run = make_run(lm=DummyLM([{}]))
    with caplog.at_level(logging.WARNING, logger="dspy.primitives.module"):
        asyncio.run(module.aforward(x="test", run=run))
        asyncio.run(module.aforward(x="test", run=run))
    assert caplog.text.count("directly is discouraged") == 1


def test_forward_through_call_no_warning(capsys, make_run):

    class TestModule(Module):
        async def _aforward_impl(self, x, **kwargs: object):
            return x

    module = TestModule()
    run = make_run(lm=DummyLM([{}]))
    asyncio.run(module(x="test", run=run))
    captured = capsys.readouterr()
    assert "directly is discouraged" not in captured.err
