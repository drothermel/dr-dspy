import asyncio
import time
from typing import Any

import pytest
from typing_extensions import override

from dspy.adapters.types.tool import Tool
from dspy.core.types import LMConfig
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.primitives.module import Module
from dspy.runtime.callback import ACTIVE_CALL_ID, NoOpCallback, with_callbacks
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts


class MyCallback(NoOpCallback):
    def __init__(self):
        self.calls = []

    @override
    def on_module_start(self, call_id, instance, inputs):
        self.calls.append({"handler": "on_module_start", "instance": instance, "inputs": inputs})

    @override
    def on_module_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None):
        self.calls.append({"handler": "on_module_end", "outputs": outputs, "exception": exception})

    @override
    def on_lm_start(self, call_id, instance, inputs):
        self.calls.append({"handler": "on_lm_start", "instance": instance, "inputs": inputs})

    @override
    def on_lm_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None):
        self.calls.append({"handler": "on_lm_end", "outputs": outputs, "exception": exception})

    @override
    def on_adapter_format_start(self, call_id, instance, inputs):
        self.calls.append({"handler": "on_adapter_format_start", "instance": instance, "inputs": inputs})

    @override
    def on_adapter_format_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None):
        self.calls.append({"handler": "on_adapter_format_end", "outputs": outputs, "exception": exception})

    @override
    def on_adapter_parse_start(self, call_id, instance, inputs):
        self.calls.append({"handler": "on_adapter_parse_start", "instance": instance, "inputs": inputs})

    @override
    def on_adapter_parse_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None):
        self.calls.append({"handler": "on_adapter_parse_end", "outputs": outputs, "exception": exception})

    @override
    def on_tool_start(self, call_id, instance, inputs):
        self.calls.append({"handler": "on_tool_start", "instance": instance, "inputs": inputs})

    @override
    def on_tool_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None):
        self.calls.append({"handler": "on_tool_end", "outputs": outputs, "exception": exception})


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [([1, "2", 3.0], {}), ([1, "2"], {"z": 3.0}), ([1], {"y": "2", "z": 3.0}), ([], {"x": 1, "y": "2", "z": 3.0})],
)
def test_callback_injection(args, kwargs, make_run):

    class Target(Module):
        @with_callbacks(kind="module")
        def forward(self, x: int, y: str, z: float, run=None) -> int:
            time.sleep(0.1)
            return x + int(y) + int(z)

    callback = MyCallback()
    run = make_run(lm=DummyLM([]), callbacks=[callback])
    target = Target()
    result = target.forward(*args, **kwargs, run=run)
    assert result == 6
    assert len(callback.calls) == 2
    assert callback.calls[0]["handler"] == "on_module_start"
    inputs = dict(callback.calls[0]["inputs"])
    inputs.pop("run", None)
    assert inputs == {"x": 1, "y": "2", "z": 3.0}
    assert callback.calls[1]["handler"] == "on_module_end"
    assert callback.calls[1]["outputs"] == 6


def test_callback_injection_local(make_run):

    class Target(Module):
        @with_callbacks(kind="module")
        def forward(self, x: int, y: str, z: float) -> int:
            time.sleep(0.1)
            return x + int(y) + int(z)

    callback = MyCallback()
    target_1 = Target(callbacks=[callback])
    result = target_1.forward(1, "2", 3.0)
    assert result == 6
    assert len(callback.calls) == 2
    assert callback.calls[0]["handler"] == "on_module_start"
    assert callback.calls[0]["inputs"] == {"x": 1, "y": "2", "z": 3.0}
    assert callback.calls[1]["handler"] == "on_module_end"
    assert callback.calls[1]["outputs"] == 6
    callback.calls = []
    target_2 = Target()
    result = target_2.forward(1, "2", 3.0)
    assert not callback.calls


def test_callback_error_handling(make_run):

    class Target(Module):
        @with_callbacks(kind="module")
        def forward(self, x: int, y: str, z: float, run=None) -> int:
            time.sleep(0.1)
            raise ValueError("Error")

    callback = MyCallback()
    run = make_run(lm=DummyLM([]), callbacks=[callback])
    target = Target()
    with pytest.raises(ValueError, match="Error"):
        target.forward(1, "2", 3.0, run=run)
    assert len(callback.calls) == 2
    assert callback.calls[0]["handler"] == "on_module_start"
    assert callback.calls[1]["handler"] == "on_module_end"
    assert isinstance(callback.calls[1]["exception"], ValueError)


def test_multiple_callbacks(make_run):

    class Target(Module):
        @with_callbacks(kind="module")
        def forward(self, x: int, y: str, z: float, run=None) -> int:
            time.sleep(0.1)
            return x + int(y) + int(z)

    callback_1 = MyCallback()
    callback_2 = MyCallback()
    run = make_run(lm=DummyLM([]), callbacks=[callback_1, callback_2])
    target = Target()
    result = target.forward(1, "2", 3.0, run=run)
    assert result == 6
    assert len(callback_1.calls) == 2
    assert len(callback_2.calls) == 2


def test_callback_complex_module(make_run):
    callback = MyCallback()
    run = make_run(
        lm=DummyLM({"How are you?": {"answer": "test output", "reasoning": "No more responses"}}), callbacks=[callback]
    )
    cot = ChainOfThought(ts("question -> answer"), config=LMConfig(n=3))
    result = asyncio.run(cot(question="How are you?", run=run))
    assert result["answer"] == "test output"
    assert result["reasoning"] == "No more responses"
    assert len(callback.calls) == 6
    assert [call["handler"] for call in callback.calls] == [
        "on_module_start",
        "on_module_start",
        "on_lm_start",
        "on_lm_end",
        "on_module_end",
        "on_module_end",
    ]


@pytest.mark.asyncio
async def test_callback_async_module(make_run):
    callback = MyCallback()
    run = make_run(
        lm=DummyLM({"How are you?": {"answer": "test output", "reasoning": "No more responses"}}), callbacks=[callback]
    )
    cot = ChainOfThought(ts("question -> answer"), config=LMConfig(n=3))
    result = await cot(question="How are you?", run=run)
    assert result["answer"] == "test output"
    assert result["reasoning"] == "No more responses"
    assert len(callback.calls) == 6
    assert [call["handler"] for call in callback.calls] == [
        "on_module_start",
        "on_module_start",
        "on_lm_start",
        "on_lm_end",
        "on_module_end",
        "on_module_end",
    ]


def test_tool_calls(make_run):
    callback = MyCallback()
    run = make_run(lm=DummyLM([]), callbacks=[callback])

    def tool_1(query: str) -> str:
        return "result 1"

    def tool_2(query: str) -> str:
        return "result 2"

    class MyModule(Module):
        def __init__(self):
            self.tools = [Tool(tool_1, description="Tool one."), Tool(tool_2, description="Tool two.")]

        async def aforward(self, *, query: str, run, options=None, **kwargs: object) -> str:
            query = self.tools[0](query=query)
            return self.tools[1](query=query)

    module = MyModule()
    result = asyncio.run(module(query="query", run=run))
    assert result == "result 2"
    assert len(callback.calls) == 6
    assert [call["handler"] for call in callback.calls] == [
        "on_module_start",
        "on_tool_start",
        "on_tool_end",
        "on_tool_start",
        "on_tool_end",
        "on_module_end",
    ]


def test_active_id(make_run):

    class CustomCallback(NoOpCallback):
        def __init__(self):
            self.parent_call_ids = []
            self.call_ids = []

        @override
        def on_module_start(self, call_id, instance, inputs):
            parent_call_id = ACTIVE_CALL_ID.get()
            self.parent_call_ids.append(parent_call_id)
            self.call_ids.append(call_id)

    class Parent(Module):
        def __init__(self):
            self.child_1 = Child()
            self.child_2 = Child()

        async def aforward(self, *, run, options=None, **inputs):
            await self.child_1(run=run, options=options, **inputs)
            await self.child_2(run=run, options=options, **inputs)

    class Child(Module):
        async def aforward(self, *, run, options=None, **inputs):
            pass

    callback = CustomCallback()
    run = make_run(lm=DummyLM([]), callbacks=[callback])
    parent = Parent()
    asyncio.run(parent(run=run))
    assert len(callback.call_ids) == 3
    assert len(set(callback.call_ids)) == 3
    parent_call_id = callback.call_ids[0]
    assert callback.parent_call_ids == [None, parent_call_id, parent_call_id]
