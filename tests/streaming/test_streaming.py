import asyncio
import time
from dataclasses import dataclass
from typing import Any, cast
from unittest import mock
from unittest.mock import AsyncMock

import anyio.from_thread
import pydantic
import pytest
from typing_extensions import override

from dspy.streaming.streamify import apply_sync_streaming
from dspy.streaming.streaming_listener import StreamListener
from dspy.utils.dummies import DummyLM

try:
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices
except ImportError:
    pytest.skip("litellm is not installed", allow_module_level=True)  # ty: ignore[too-many-positional-arguments]

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types.base_type import Type
from dspy.adapters.types.citation import Citations
from dspy.adapters.types.document import Document
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool
from dspy.adapters.xml_adapter import XMLAdapter
from dspy.clients.lm import LM
from dspy.core.types import LMOutput
from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature
from dspy.streaming.messages import StatusMessage, StatusMessageProvider, StreamResponse
from dspy.streaming.streamify import streamify, streaming_response


@pytest.mark.anyio
async def test_streamify_yields_expected_response_chunks(litellm_test_server):
    api_base, _ = litellm_test_server
    lm = LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        cache=True,
    )
    with settings.context(lm=lm, adapter=JSONAdapter()):

        class TestSignature(Signature):
            input_text: str = InputField()
            output_text: str = OutputField()

        program: Any = streamify(Predict(TestSignature))
        output_stream1 = program(input_text="Test")
        output_chunks1 = [chunk async for chunk in output_stream1]
        last_chunk1 = output_chunks1[-1]
        assert isinstance(last_chunk1, Prediction)
        assert last_chunk1.output_text == "Hello!"

        output_stream2 = program(input_text="Test")
        output_chunks2 = [chunk async for chunk in output_stream2]
        # Since the input is cached, only one chunk should be
        # yielded containing the prediction
        assert len(output_chunks2) == 1
        last_chunk2 = output_chunks2[-1]
        assert isinstance(last_chunk2, Prediction)
        assert last_chunk2.output_text == "Hello!"


@pytest.mark.anyio
async def test_streaming_response_yields_expected_response_chunks(litellm_test_server):
    api_base, _ = litellm_test_server
    lm = LM(
        model="openai/dspy-test-model",
        api_base=api_base,
        api_key="fakekey",
        cache=False,
    )
    with settings.context(lm=lm):

        class TestSignature(Signature):
            input_text: str = InputField()
            output_text: str = OutputField()

        program: Any = streamify(Predict(TestSignature))
        output_stream_from_program = streaming_response(program(input_text="Test"))
        output_stream_for_server_response = streaming_response(output_stream_from_program)
        output_chunks = [chunk async for chunk in output_stream_for_server_response]
        assert all(chunk.startswith("data: ") for chunk in output_chunks)
        assert 'data: {"prediction":{"output_text":"Hello!"}}\n\n' in output_chunks
        assert output_chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.anyio
async def test_default_status_streaming():
    class MyProgram(Module):
        def __init__(self):
            self.generate_question = Tool(lambda x: f"What color is the {x}?", name="generate_question")
            self.predict = Predict("question->answer")

        @override
        def __call__(self, x: str):
            question = self.generate_question(x=x)
            return self.predict(question=question)

    lm = DummyLM([{"answer": "red"}, {"answer": "blue"}])
    with settings.context(lm=lm):
        program: Any = streamify(MyProgram())
        output = program("sky")

        status_messages = []
        async for value in output:
            if isinstance(value, StatusMessage):
                status_messages.append(value)  # noqa: PERF401

    assert len(status_messages) == 2
    assert status_messages[0].message == "Calling tool generate_question..."
    assert status_messages[1].message == "Tool calling finished! Querying the LLM with tool calling results..."


@pytest.mark.anyio
async def test_custom_status_streaming():
    class MyProgram(Module):
        def __init__(self):
            self.generate_question = Tool(lambda x: f"What color is the {x}?", name="generate_question")
            self.predict = Predict("question->answer")

        @override
        def __call__(self, x: str):
            question = self.generate_question(x=x)
            return self.predict(question=question)

    class MyStatusMessageProvider(StatusMessageProvider):
        @override
        def tool_start_status_message(self, instance, inputs):
            return "Tool starting!"

        @override
        def tool_end_status_message(self, outputs):
            return "Tool finished!"

        @override
        def module_start_status_message(self, instance, inputs):
            if isinstance(instance, Predict):
                return "Predict starting!"
            return None

    lm = DummyLM([{"answer": "red"}, {"answer": "blue"}])
    with settings.context(lm=lm):
        program: Any = streamify(MyProgram(), status_message_provider=MyStatusMessageProvider())
        output = program("sky")

        status_messages = []
        async for value in output:
            if isinstance(value, StatusMessage):
                status_messages.append(value)  # noqa: PERF401

        assert len(status_messages) == 3
        assert status_messages[0].message == "Tool starting!"
        assert status_messages[1].message == "Tool finished!"
        assert status_messages[2].message == "Predict starting!"


@pytest.mark.anyio
async def test_concurrent_status_message_providers():
    class MyProgram(Module):
        def __init__(self):
            self.generate_question = Tool(lambda x: f"What color is the {x}?", name="generate_question")
            self.predict = Predict("question->answer")

        @override
        def __call__(self, x: str):
            question = self.generate_question(x=x)
            return self.predict(question=question)

    class MyStatusMessageProvider1(StatusMessageProvider):
        @override
        def tool_start_status_message(self, instance, inputs):
            return "Provider1: Tool starting!"

        @override
        def tool_end_status_message(self, outputs):
            return "Provider1: Tool finished!"

        @override
        def module_start_status_message(self, instance, inputs):
            if isinstance(instance, Predict):
                return "Provider1: Predict starting!"
            return None

    class MyStatusMessageProvider2(StatusMessageProvider):
        @override
        def tool_start_status_message(self, instance, inputs):
            return "Provider2: Tool starting!"

        @override
        def tool_end_status_message(self, outputs):
            return "Provider2: Tool finished!"

        @override
        def module_start_status_message(self, instance, inputs):
            if isinstance(instance, Predict):
                return "Provider2: Predict starting!"
            return None

    # Store the original callbacks to verify they're not modified
    original_callbacks = list(settings.callbacks)

    lm = DummyLM([{"answer": "red"}, {"answer": "blue"}, {"answer": "green"}, {"answer": "yellow"}])

    # Results storage for each thread
    results = {}

    async def run_with_provider1():
        with settings.context(lm=lm):
            program: Any = streamify(MyProgram(), status_message_provider=MyStatusMessageProvider1())
            output = program("sky")

            status_messages = []
            async for value in output:
                if isinstance(value, StatusMessage):
                    status_messages.append(value.message)  # noqa: PERF401

            results["provider1"] = status_messages

    async def run_with_provider2():
        with settings.context(lm=lm):
            program: Any = streamify(MyProgram(), status_message_provider=MyStatusMessageProvider2())
            output = program("ocean")

            status_messages = []
            async for value in output:
                if isinstance(value, StatusMessage):
                    status_messages.append(value.message)  # noqa: PERF401

            results["provider2"] = status_messages

    # Run both tasks concurrently
    await asyncio.gather(run_with_provider1(), run_with_provider2())

    # Verify provider1 got its expected messages
    assert len(results["provider1"]) == 3
    assert results["provider1"][0] == "Provider1: Tool starting!"
    assert results["provider1"][1] == "Provider1: Tool finished!"
    assert results["provider1"][2] == "Provider1: Predict starting!"

    # Verify provider2 got its expected messages
    assert len(results["provider2"]) == 3
    assert results["provider2"][0] == "Provider2: Tool starting!"
    assert results["provider2"][1] == "Provider2: Tool finished!"
    assert results["provider2"][2] == "Provider2: Predict starting!"

    # Verify that the global callbacks were not modified
    assert settings.callbacks == original_callbacks


@pytest.mark.llm_call
@pytest.mark.anyio
async def test_stream_listener_chat_adapter(lm_for_test):
    class MyProgram(Module):
        def __init__(self):
            self.predict1 = Predict("question->answer")
            self.predict2 = Predict("question, answer->judgement")

        @override
        def __call__(self, x: str, **kwargs: object):
            answer = self.predict1(question=x, **kwargs)
            return self.predict2(question=x, answer=answer, **kwargs)

    my_program = MyProgram()
    program: Any = streamify(
        my_program,
        stream_listeners=[
            StreamListener(signature_field_name="answer"),
            StreamListener(signature_field_name="judgement"),
        ],
        include_final_prediction_in_output_stream=False,
    )
    # Turn off the cache to ensure the stream is produced.
    with settings.context(lm=LM(lm_for_test, cache=False, temperature=0.0)):
        output = program(x="why did a chicken cross the kitchen?")
        all_chunks = []
        async for value in output:
            if isinstance(value, StreamResponse):
                all_chunks.append(value)  # noqa: PERF401

    assert all_chunks[0].predict_name == "self.predict1"
    assert all_chunks[0].signature_field_name == "answer"
    # The last chunk can be from either predictor because sometimes small LMs miss the `[[ ## completed ## ]]` marker,
    # which results in an extra chunk that flushes out the buffer.
    assert all_chunks[-2].predict_name == "self.predict2"
    assert all_chunks[-2].signature_field_name == "judgement"


@pytest.mark.anyio
async def test_default_status_streaming_in_async_program():
    class MyProgram(Module):
        def __init__(self):
            self.generate_question = Tool(lambda x: f"What color is the {x}?", name="generate_question")
            self.predict = Predict("question->answer")

        @override
        async def acall(self, x: str):
            question = await cast("Any", self.generate_question).acall(x=x)
            return await self.predict.acall(question=question)

    lm = DummyLM([{"answer": "red"}, {"answer": "blue"}])
    with settings.context(lm=lm):
        program: Any = streamify(MyProgram(), is_async_program=True)
        output = program("sky")

        status_messages = []
        async for value in output:
            if isinstance(value, StatusMessage):
                status_messages.append(value)  # noqa: PERF401

    assert len(status_messages) == 2
    assert status_messages[0].message == "Calling tool generate_question..."
    assert status_messages[1].message == "Tool calling finished! Querying the LLM with tool calling results..."


@pytest.mark.llm_call
@pytest.mark.anyio
async def test_stream_listener_json_adapter(lm_for_test):
    class MyProgram(Module):
        def __init__(self):
            self.predict1 = Predict("question->answer")
            self.predict2 = Predict("question, answer->judgement")

        @override
        def __call__(self, x: str, **kwargs: object):
            answer = self.predict1(question=x, **kwargs)
            return self.predict2(question=x, answer=answer, **kwargs)

    my_program = MyProgram()
    program: Any = streamify(
        my_program,
        stream_listeners=[
            StreamListener(signature_field_name="answer"),
            StreamListener(signature_field_name="judgement"),
        ],
        include_final_prediction_in_output_stream=False,
    )
    # Turn off the cache to ensure the stream is produced.
    with settings.context(lm=LM(lm_for_test, cache=False, temperature=0.0), adapter=JSONAdapter()):
        output = program(x="why did a chicken cross the kitchen?")
        all_chunks = []
        async for value in output:
            if isinstance(value, StreamResponse):
                all_chunks.append(value)  # noqa: PERF401

    assert all_chunks[0].predict_name == "self.predict1"
    assert all_chunks[0].signature_field_name == "answer"
    assert all_chunks[0].is_last_chunk is False

    assert all_chunks[-1].predict_name == "self.predict2"
    assert all_chunks[-1].signature_field_name == "judgement"


@pytest.mark.anyio
async def test_streaming_handles_space_correctly():
    my_program = Predict("question->answer")
    program: Any = streamify(my_program, stream_listeners=[StreamListener(signature_field_name="answer")])

    async def gpt_4o_mini_stream(*args: object, **kwargs: object):
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ## answer ## ]]\n"))]
        )
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="How "))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="are "))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="you "))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="doing?"))])
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## completed ## ]]"))]
        )

    with mock.patch("litellm.acompletion", side_effect=gpt_4o_mini_stream):  # noqa: SIM117
        with settings.context(lm=LM("openai/gpt-4o-mini", cache=False), adapter=ChatAdapter()):
            output = program(question="What is the capital of France?")
            all_chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    all_chunks.append(value)  # noqa: PERF401

    assert "".join([chunk.chunk for chunk in all_chunks]) == "How are you doing?"


@pytest.mark.llm_call
def test_sync_streaming(lm_for_test):
    class MyProgram(Module):
        def __init__(self):
            self.predict1 = Predict("question->answer")
            self.predict2 = Predict("question, answer->judgement")

        @override
        def __call__(self, x: str, **kwargs: object):
            answer = self.predict1(question=x, **kwargs)
            return self.predict2(question=x, answer=answer, **kwargs)

    my_program = MyProgram()
    program: Any = streamify(
        my_program,
        stream_listeners=[
            StreamListener(signature_field_name="answer"),
            StreamListener(signature_field_name="judgement"),
        ],
        include_final_prediction_in_output_stream=False,
        async_streaming=False,
    )
    # Turn off the cache to ensure the stream is produced.
    with settings.context(lm=LM(lm_for_test, cache=False, temperature=0.0)):
        output = program(x="why did a chicken cross the kitchen?")
        all_chunks = []
        for value in output:
            if isinstance(value, StreamResponse):
                all_chunks.append(value)  # noqa: PERF401

    assert all_chunks[0].predict_name == "self.predict1"
    assert all_chunks[0].signature_field_name == "answer"
    assert all_chunks[0].is_last_chunk is False
    # The last chunk can be from either predictor because sometimes small LMs miss the `[[ ## completed ## ]]` marker,
    # which results in an extra chunk that flushes out the buffer.
    assert all_chunks[-2].predict_name == "self.predict2"
    assert all_chunks[-2].signature_field_name == "judgement"


def test_sync_status_streaming():
    class MyProgram(Module):
        def __init__(self):
            self.generate_question = Tool(lambda x: f"What color is the {x}?", name="generate_question")
            self.predict = Predict("question->answer")

        @override
        def __call__(self, x: str):
            question = self.generate_question(x=x)
            return self.predict(question=question)

    lm = DummyLM([{"answer": "red"}, {"answer": "blue"}])
    with settings.context(lm=lm):
        program: Any = streamify(MyProgram())
        output = program("sky")
        sync_output = apply_sync_streaming(output)
        status_messages = []
        for value in sync_output:
            if isinstance(value, StatusMessage):
                status_messages.append(value)  # noqa: PERF401

    assert len(status_messages) == 2
    assert status_messages[0].message == "Calling tool generate_question..."
    assert status_messages[1].message == "Tool calling finished! Querying the LLM with tool calling results..."


@pytest.mark.anyio
async def test_stream_listener_returns_correct_chunk_chat_adapter():
    class MyProgram(Module):
        def __init__(self):
            super().__init__()
            self.predict1 = Predict("question->answer")
            self.predict2 = Predict("question, answer->judgement")

        def forward(self, question, **kwargs: object):
            answer = self.predict1(question=question, **kwargs).answer
            return self.predict2(question=question, answer=answer, **kwargs)

    async def gpt_4o_mini_stream_1(*args: object, **kwargs: object):
        # Recorded streaming from openai/gpt-4o-mini
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[["))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" answer"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ]]\n\n"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="To"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" get"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" to"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" the"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" other"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" side"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" of"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" the"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" dinner"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" plate"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="!\n\n[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" completed"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ]]"))])

    async def gpt_4o_mini_stream_2():
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" judgement"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ]]\n\n"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="The"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" answer"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" is"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" humorous"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" and"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" plays"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" on"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" the"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" classic"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" joke"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" format"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=".\n\n[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" completed"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ]]"))])

    stream_generators = [gpt_4o_mini_stream_1, gpt_4o_mini_stream_2]

    async def completion_side_effect(*args: object, **kwargs: object):
        return stream_generators.pop(0)()  # return new async generator instance

    with mock.patch("litellm.acompletion", side_effect=completion_side_effect):
        program: Any = streamify(
            MyProgram(),
            stream_listeners=[
                StreamListener(signature_field_name="answer"),
                StreamListener(signature_field_name="judgement"),
            ],
        )
        with settings.context(lm=LM("openai/gpt-4o-mini", cache=False)):
            output = program(question="why did a chicken cross the kitchen?")
            all_chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    all_chunks.append(value)  # noqa: PERF401

        assert all_chunks[0].predict_name == "self.predict1"
        assert all_chunks[0].signature_field_name == "answer"
        assert all_chunks[0].chunk == "To"
        assert all_chunks[1].chunk == " get"
        assert all_chunks[2].chunk == " to"
        assert all_chunks[3].chunk == " the"
        assert all_chunks[4].chunk == " other"
        assert all_chunks[5].chunk == " side"
        assert all_chunks[6].chunk == " of"
        assert all_chunks[7].chunk == " the"
        assert all_chunks[8].chunk == " dinner"
        assert all_chunks[9].chunk == " plate"
        assert all_chunks[10].chunk == "!"
        assert all_chunks[10].is_last_chunk is True

        assert all_chunks[11].predict_name == "self.predict2"
        assert all_chunks[11].signature_field_name == "judgement"
        assert all_chunks[11].chunk == "The"
        assert all_chunks[12].chunk == " answer"
        assert all_chunks[13].chunk == " is"
        assert all_chunks[14].chunk == " humorous"
        assert all_chunks[15].chunk == " and"
        assert all_chunks[16].chunk == " plays"
        assert all_chunks[17].chunk == " on"
        assert all_chunks[18].chunk == " the"
        assert all_chunks[19].chunk == " classic"
        assert all_chunks[20].chunk == " joke"
        assert all_chunks[21].chunk == " format"
        assert all_chunks[22].chunk == "."
        assert all_chunks[22].is_last_chunk is True


@pytest.mark.anyio
async def test_stream_listener_returns_correct_chunk_json_adapter():
    class MyProgram(Module):
        def __init__(self):
            super().__init__()
            self.predict1 = Predict("question->answer")
            self.predict2 = Predict("question,answer->judgement")

        def forward(self, question, **kwargs: object):
            answer = self.predict1(question=question, **kwargs).answer
            return self.predict2(question=question, answer=answer, **kwargs)

    async def gpt_4o_mini_stream_1(*args: object, **kwargs: object):
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="answer"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='":'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='"To'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" get"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" to"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" the"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" other"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" side"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" of"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" the"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" frying"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" pan"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='!"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="}\n"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="None"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="None"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="None"))])

    async def gpt_4o_mini_stream_2(*args: object, **kwargs: object):
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="jud"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="gement"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='":'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='"The'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" answer"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" is"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" humorous"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" and"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" plays"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" on"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" the"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" very"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" funny"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" and"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" classic"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" joke"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" format"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='."'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="}"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="None"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="None"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="None"))])

    with mock.patch(
        "litellm.acompletion", new_callable=AsyncMock, side_effect=[gpt_4o_mini_stream_1(), gpt_4o_mini_stream_2()]
    ):
        program: Any = streamify(
            MyProgram(),
            stream_listeners=[
                StreamListener(signature_field_name="answer"),
                StreamListener(signature_field_name="judgement"),
            ],
        )
        with settings.context(lm=LM("openai/gpt-4o-mini", cache=False), adapter=JSONAdapter()):
            output = program(question="why did a chicken cross the kitchen?")
            all_chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    all_chunks.append(value)  # noqa: PERF401

        assert all_chunks[0].predict_name == "self.predict1"
        assert all_chunks[0].signature_field_name == "answer"
        assert all_chunks[0].chunk == '"To'
        assert all_chunks[1].chunk == " get"
        assert all_chunks[2].chunk == " to"
        assert all_chunks[3].chunk == " the"
        assert all_chunks[4].chunk == " other"
        assert all_chunks[5].chunk == " side"
        assert all_chunks[6].chunk == " of"
        assert all_chunks[7].chunk == " the"
        assert all_chunks[8].chunk == " frying"
        assert all_chunks[9].chunk == " pan"
        assert all_chunks[10].chunk == '!"'
        assert all_chunks[10].is_last_chunk is True

        assert all_chunks[11].predict_name == "self.predict2"
        assert all_chunks[11].signature_field_name == "judgement"
        assert all_chunks[11].chunk == '"The'
        assert all_chunks[12].chunk == " answer"
        assert all_chunks[13].chunk == " is"
        assert all_chunks[14].chunk == " humorous"
        assert all_chunks[15].chunk == " and"
        assert all_chunks[16].chunk == " plays"
        assert all_chunks[17].chunk == " on"
        assert all_chunks[18].chunk == " the"
        assert all_chunks[19].chunk == " very"
        assert all_chunks[20].chunk == " funny"
        assert all_chunks[21].chunk == " and"
        assert all_chunks[22].chunk == " classic"
        assert all_chunks[23].chunk == " joke"
        assert all_chunks[24].chunk == " format"
        assert all_chunks[25].chunk == '."'
        assert all_chunks[25].is_last_chunk is True


@pytest.mark.anyio
async def test_stream_listener_returns_correct_chunk_chat_adapter_untokenized_stream():
    class MyProgram(Module):
        def __init__(self):
            super().__init__()
            self.predict1 = Predict("question->answer")
            self.predict2 = Predict("question,answer->judgement")

        def forward(self, question, **kwargs: object):
            answer = self.predict1(question=question, **kwargs).answer
            return self.predict2(question=question, answer=answer, **kwargs)

    async def gemini_stream_1(*args: object, **kwargs: object):
        yield ModelResponseStream(model="gemini", choices=[StreamingChoices(delta=Delta(content="[[ ##"))])
        yield ModelResponseStream(model="gemini", choices=[StreamingChoices(delta=Delta(content=" answer ## ]]"))])
        yield ModelResponseStream(
            model="gemini", choices=[StreamingChoices(delta=Delta(content="To get to the other side."))]
        )
        yield ModelResponseStream(
            model="gemini", choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## completed ## ]]"))]
        )

    async def gemini_stream_2(*args: object, **kwargs: object):
        yield ModelResponseStream(
            model="gemini", choices=[StreamingChoices(delta=Delta(content="[[ ## judgement ## ]]\n\n"))]
        )
        yield ModelResponseStream(
            model="gemini",
            choices=[
                StreamingChoices(
                    delta=Delta(
                        content=(
                            "The answer provides the standard punchline for this classic joke format, adapted to the "
                            "specific location mentioned in the question. It is the expected and appropriate response."
                        )
                    )
                )
            ],
        )
        yield ModelResponseStream(
            model="gemini",
            choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## completed ## ]]"))],
        )
        yield ModelResponseStream(model="gemini", choices=[StreamingChoices(delta=Delta(content="}\n"))])

    with mock.patch("litellm.acompletion", new_callable=AsyncMock, side_effect=[gemini_stream_1(), gemini_stream_2()]):
        program: Any = streamify(
            MyProgram(),
            stream_listeners=[
                StreamListener(signature_field_name="answer"),
                StreamListener(signature_field_name="judgement"),
            ],
        )
        with settings.context(lm=LM("gemini/gemini-2.5-flash", cache=False), adapter=ChatAdapter()):
            output = program(question="why did a chicken cross the kitchen?")
            all_chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    all_chunks.append(value)  # noqa: PERF401

        assert all_chunks[0].predict_name == "self.predict1"
        assert all_chunks[0].signature_field_name == "answer"
        assert all_chunks[0].chunk == "To get to the other side."
        assert all_chunks[1].is_last_chunk is True

        assert all_chunks[2].predict_name == "self.predict2"
        assert all_chunks[2].signature_field_name == "judgement"
        assert all_chunks[2].chunk == (
            "The answer provides the standard punchline for this classic joke format, adapted to the specific location "
            "mentioned in the question. It is the expected and appropriate response."
        )


@pytest.mark.anyio
async def test_stream_listener_missing_completion_marker_chat_adapter():
    """Test that streaming works correctly when LLM response omits a final completion marker.

    This test verifies that:
    1. All tokens are yielded including those in the buffer
    2. The last chunk is properly marked with is_last_chunk=True
    3. No tokens are lost when the completion marker is missing
    """

    class MyProgram(Module):
        def __init__(self):
            super().__init__()
            self.predict = Predict("question->answer")

        def forward(self, question, **kwargs: object):
            return self.predict(question=question, **kwargs)

    async def incomplete_stream(*args: object, **kwargs: object):
        """Stream that includes start marker but MISSING completion marker"""
        # Start marker
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" answer"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ## ]]\n\n"))])

        # Content tokens - more than 10 to ensure buffering happens
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="This"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" is"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" a"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" test"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" response"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" with"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" many"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" tokens"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" to"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ensure"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" buffering"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" works"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" correctly"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="."))])
        # NO COMPLETION MARKER

    with mock.patch("litellm.acompletion", side_effect=incomplete_stream):
        program: Any = streamify(
            MyProgram(),
            stream_listeners=[
                StreamListener(signature_field_name="answer"),
            ],
        )
        with settings.context(lm=LM("openai/gpt-4o-mini", cache=False), adapter=ChatAdapter()):
            output = program(question="Test question")
            all_chunks = []
            final_prediction = None
            async for value in output:
                if isinstance(value, StreamResponse):
                    all_chunks.append(value)
                elif isinstance(value, Prediction):
                    final_prediction = value

    full_content = "".join([chunk.chunk for chunk in all_chunks])
    expected_content = "This is a test response with many tokens to ensure buffering works correctly."
    assert full_content == expected_content
    assert final_prediction is not None
    assert final_prediction.answer == expected_content


@pytest.mark.anyio
async def test_stream_listener_returns_correct_chunk_json_adapter_untokenized_stream():
    class MyProgram(Module):
        def __init__(self):
            super().__init__()
            self.predict1 = Predict("question->answer")
            self.predict2 = Predict("question,answer->judgement")

        def forward(self, question, **kwargs: object):
            answer = self.predict1(question=question, **kwargs).answer
            return self.predict2(question=question, answer=answer, **kwargs)

    async def gemini_stream_1(*args: object, **kwargs: object):
        yield ModelResponseStream(model="gemini", choices=[StreamingChoices(delta=Delta(content="{\n"))])
        yield ModelResponseStream(
            model="gemini", choices=[StreamingChoices(delta=Delta(content='  "answer": "To get to'))]
        )
        yield ModelResponseStream(
            model="gemini", choices=[StreamingChoices(delta=Delta(content=' the other side... of the cutting board!"'))]
        )
        yield ModelResponseStream(model="gemini", choices=[StreamingChoices(delta=Delta(content="}\n"))])

    async def gemini_stream_2(*args: object, **kwargs: object):
        yield ModelResponseStream(model="gemini", choices=[StreamingChoices(delta=Delta(content="{\n"))])
        yield ModelResponseStream(
            model="gemini", choices=[StreamingChoices(delta=Delta(content='  "judgement": "The'))]
        )
        yield ModelResponseStream(
            model="gemini",
            choices=[
                StreamingChoices(
                    delta=Delta(
                        content=' answer provides a humorous and relevant punchline to the classic joke setup."'
                    )
                )
            ],
        )
        yield ModelResponseStream(model="gemini", choices=[StreamingChoices(delta=Delta(content="}\n"))])

    with mock.patch("litellm.acompletion", new_callable=AsyncMock, side_effect=[gemini_stream_1(), gemini_stream_2()]):
        program: Any = streamify(
            MyProgram(),
            stream_listeners=[
                StreamListener(signature_field_name="answer"),
                StreamListener(signature_field_name="judgement"),
            ],
        )
        with settings.context(lm=LM("gemini/gemini-2.5-flash", cache=False), adapter=JSONAdapter()):
            output = program(question="why did a chicken cross the kitchen?")
            all_chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    all_chunks.append(value)  # noqa: PERF401

        assert all_chunks[0].predict_name == "self.predict1"
        assert all_chunks[0].signature_field_name == "answer"

        assert all_chunks[0].chunk == '"To get to the other side... of the cutting board!"'

        assert all_chunks[1].predict_name == "self.predict2"
        assert all_chunks[1].signature_field_name == "judgement"
        assert (
            all_chunks[1].chunk == '"The answer provides a humorous and relevant punchline to the classic joke setup."'
        )


@pytest.mark.anyio
async def test_status_message_non_blocking():
    def dummy_tool():
        time.sleep(1)
        return "dummy_tool_output"

    class MyProgram(Module):
        def forward(self, question, **kwargs: object):
            Tool(dummy_tool)()
            return Prediction(answer="dummy_tool_output")

    program: Any = streamify(MyProgram(), status_message_provider=StatusMessageProvider())

    with mock.patch("litellm.acompletion", new_callable=AsyncMock, side_effect=[dummy_tool]):  # noqa: SIM117
        with settings.context(lm=LM("openai/gpt-4o-mini", cache=False)):
            output = program(question="why did a chicken cross the kitchen?")
            timestamps = []
            async for value in output:
                if isinstance(value, StatusMessage):
                    timestamps.append(time.time())  # noqa: PERF401

    # timestamps[0]: tool start message
    # timestamps[1]: tool end message
    # There should be ~1 second delay between the tool start and end messages because we explicitly sleep for 1 second
    # in the tool.
    assert timestamps[1] - timestamps[0] >= 1


@pytest.mark.anyio
async def test_status_message_non_blocking_async_program():
    async def dummy_tool():
        await asyncio.sleep(1)
        return "dummy_tool_output"

    class MyProgram(Module):
        async def aforward(self, question, **kwargs: object):
            await cast("Any", Tool(dummy_tool)).acall()
            return Prediction(answer="dummy_tool_output")

    program: Any = streamify(MyProgram(), status_message_provider=StatusMessageProvider(), is_async_program=True)

    with mock.patch("litellm.acompletion", new_callable=AsyncMock, side_effect=[dummy_tool]):  # noqa: SIM117
        with settings.context(lm=LM("openai/gpt-4o-mini", cache=False)):
            output = program(question="why did a chicken cross the kitchen?")
            timestamps = []
            async for value in output:
                if isinstance(value, StatusMessage):
                    timestamps.append(time.time())  # noqa: PERF401

    # timestamps[0]: tool start message
    # timestamps[1]: tool end message
    # There should be ~1 second delay between the tool start and end messages because we explicitly sleep for 1 second
    # in the tool.
    assert timestamps[1] - timestamps[0] >= 1


@pytest.mark.anyio
async def test_stream_listener_allow_reuse():
    class MyProgram(Module):
        def __init__(self):
            super().__init__()
            self.predict = Predict("question->answer")

        def forward(self, question, **kwargs: object):
            self.predict(question=question, **kwargs)
            return self.predict(question=question, **kwargs)

    program: Any = streamify(
        MyProgram(),
        stream_listeners=[
            StreamListener(signature_field_name="answer", allow_reuse=True),
        ],
    )

    async def gpt_4o_mini_stream(*args: object, **kwargs: object):
        # Recorded streaming from openai/gpt-4o-mini
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[["))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" answer"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ]]\n\n"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="To"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" get"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" to"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" the"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" other"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" side"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="!\n\n[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" completed"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ]]"))])

    stream_generators = [gpt_4o_mini_stream, gpt_4o_mini_stream]

    async def completion_side_effect(*args: object, **kwargs: object):
        return stream_generators.pop(0)()  # return new async generator instance

    with mock.patch("litellm.acompletion", side_effect=completion_side_effect):  # noqa: SIM117
        with settings.context(lm=LM("openai/gpt-4o-mini", cache=False)):
            output = program(question="why did a chicken cross the kitchen?")
            all_chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    all_chunks.append(value)  # noqa: PERF401

    concat_message = "".join([chunk.chunk for chunk in all_chunks])
    # Both matching predicts stream the same answer text, so the listener emits it twice.
    assert concat_message == "To get to the other side!To get to the other side!"


@pytest.mark.anyio
async def test_stream_listener_returns_correct_chunk_xml_adapter():
    class MyProgram(Module):
        def __init__(self):
            super().__init__()
            self.predict1 = Predict("question->answer")
            self.predict2 = Predict("question,answer->judgement")

        def forward(self, question, **kwargs: object):
            answer = self.predict1(question=question, **kwargs).answer
            return self.predict2(question=question, answer=answer, **kwargs)

    async def xml_stream_1(*args: object, **kwargs: object):
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="<"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="answer"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=">"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="To"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" get"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" to"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" the"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" other"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" side"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="!"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="<"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="/answer"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=">"))])

    async def xml_stream_2(*args: object, **kwargs: object):
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="<"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="judgement"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=">"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="The"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" answer"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" is"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" humorous"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="."))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="<"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="/judgement"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=">"))])

    stream_generators = [xml_stream_1, xml_stream_2]

    async def completion_side_effect(*args: object, **kwargs: object):
        return stream_generators.pop(0)()

    with mock.patch("litellm.acompletion", side_effect=completion_side_effect):
        program: Any = streamify(
            MyProgram(),
            stream_listeners=[
                StreamListener(signature_field_name="answer"),
                StreamListener(signature_field_name="judgement"),
            ],
        )
        with settings.context(lm=LM("openai/gpt-4o-mini", cache=False), adapter=XMLAdapter()):
            output = program(question="why did a chicken cross the kitchen?")
            all_chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    all_chunks.append(value)  # noqa: PERF401

    # Verify answer chunks
    answer_chunks = [chunk for chunk in all_chunks if chunk.signature_field_name == "answer"]
    assert len(answer_chunks) > 0
    assert answer_chunks[0].predict_name == "self.predict1"
    assert "".join([chunk.chunk for chunk in answer_chunks]) == "To get to the other side!"

    # Verify judgement chunks
    judgement_chunks = [chunk for chunk in all_chunks if chunk.signature_field_name == "judgement"]
    assert len(judgement_chunks) > 0
    assert judgement_chunks[0].predict_name == "self.predict2"
    assert "".join([chunk.chunk for chunk in judgement_chunks]) == "The answer is humorous."


@pytest.mark.anyio
async def test_streaming_allows_custom_chunk_types():
    @dataclass
    class CustomChunk:
        text: str

    class MyProgram(Module):
        def forward(self, question, **kwargs: object):
            async def send_to_stream():
                chunk = CustomChunk(text="hello")
                await settings.send_stream.send(chunk)

            anyio.from_thread.run(send_to_stream)
            return Prediction(answer="dummy output")

    program: Any = streamify(MyProgram())

    output = program(question="why did a chicken cross the kitchen?")
    all_chunks = []
    async for value in output:
        all_chunks.append(value)  # noqa: PERF401

    assert isinstance(all_chunks[0], CustomChunk)
    assert isinstance(all_chunks[1], Prediction)


@pytest.mark.anyio
async def test_streaming_allows_custom_streamable_type():
    class CustomType(Type):
        message: str

        @classmethod
        @override
        def is_streamable(cls) -> bool:
            return True

        @classmethod
        @override
        def parse_stream_chunk(cls, chunk):
            return CustomType(message=chunk.choices[0].delta.content)

        @classmethod
        @override
        def parse_lm_output(cls, output: LMOutput) -> "CustomType | None":
            if output.text is None:
                return None
            return CustomType(message=output.text.split("\n\n")[0])

    class CustomSignature(Signature):
        question: str = InputField()
        answer: CustomType = OutputField()

    program: Any = streamify(
        Predict(CustomSignature),
        stream_listeners=[
            StreamListener(signature_field_name="answer"),
        ],
    )

    async def stream(*args: object, **kwargs: object):
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="Hello"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="World"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" completed"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ]]"))])

    with (
        mock.patch("litellm.acompletion", side_effect=stream),
        settings.context(
            lm=LM("openai/gpt-4o-mini", cache=False), adapter=ChatAdapter(native_response_types=[CustomType])
        ),
    ):
        output = program(question="why did a chicken cross the kitchen?")
        all_chunks = []
        async for value in output:
            if isinstance(value, StreamResponse):
                all_chunks.append(value)
            elif isinstance(value, Prediction):
                assert isinstance(value.answer, CustomType)
                assert value.answer.message == "HelloWorld"

    assert all(isinstance(chunk.chunk, CustomType) for chunk in all_chunks)


@pytest.mark.anyio
async def test_streaming_with_citations():
    class AnswerWithSources(Signature):
        """Answer questions using provided documents with citations."""

        documents: list["Document"] = InputField()
        question: str = InputField()
        answer: str = OutputField()
        citations: "Citations" = OutputField()

    class MyProgram(Module):
        def __init__(self):
            super().__init__()
            self.predict = Predict(AnswerWithSources)

        def forward(self, documents, question, **kwargs: object):
            return self.predict(documents=documents, question=question, **kwargs)

    async def citation_stream(*args: object, **kwargs: object):
        # Stream chunks with citation data in provider_specific_fields
        # To verify the realistic scenario with more than 10 chunks in the stream, include more than 10 chunks before the citation.
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="[[ ##"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" answer"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" ## ]]\n\n"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="A"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="c"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="c"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="o"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="r"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="d"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="i"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="n"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="g"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" to "))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content="the references,"))])
        yield ModelResponseStream(
            model="claude",
            choices=[
                StreamingChoices(
                    delta=Delta(
                        content="",
                        provider_specific_fields={
                            "citation": {
                                "type": "char_location",
                                "cited_text": "water boils at 100°C",
                                "document_index": 0,
                                "document_title": "Physics Facts",
                                "start_char_index": 0,
                                "end_char_index": 19,
                            }
                        },
                    )
                )
            ],
        )
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" water"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" boils"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" at"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" 100°C"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=".\n\n[[ ##"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" completed"))])
        yield ModelResponseStream(model="claude", choices=[StreamingChoices(delta=Delta(content=" ## ]]"))])

    # Mock the final response choice to include provider_specific_fields with citations
    with mock.patch("litellm.acompletion", return_value=citation_stream()):
        program: Any = streamify(
            MyProgram(),
            stream_listeners=[
                StreamListener(signature_field_name="answer"),
                StreamListener(signature_field_name="citations"),
            ],
        )

        # Create test documents
        docs = [Document(data="Water boils at 100°C at standard pressure.", title="Physics Facts")]

        with settings.context(
            lm=LM("anthropic/claude-3-5-sonnet-20241022", cache=False),
            adapter=ChatAdapter(native_response_types=[cast("Any", Citations)]),
        ):
            output = program(documents=docs, question="What temperature does water boil?")
            citation_chunks = []
            answer_chunks = []
            final_prediction = None
            async for value in output:
                if isinstance(value, StreamResponse) and value.signature_field_name == "citations":
                    citation_chunks.append(value)
                elif isinstance(value, StreamResponse) and value.signature_field_name == "answer":
                    answer_chunks.append(value.chunk)
                elif isinstance(value, Prediction):
                    final_prediction = value

            # Test that we received citation chunks from streaming
            assert len(citation_chunks) > 0
            citation_chunk = citation_chunks[0]
            assert isinstance(citation_chunk.chunk, cast("Any", Citations))
            assert len(citation_chunk.chunk) == 1
            assert citation_chunk.chunk[0].cited_text == "water boils at 100°C"
            assert citation_chunk.chunk[0].document_title == "Physics Facts"

            # Verify the answer chunks are correct
            assert "".join(answer_chunks) == "According to the references, water boils at 100°C."

            # Test that prediction contains the expected fields
            assert final_prediction is not None
            assert hasattr(final_prediction, "answer")
            assert hasattr(final_prediction, "citations")


# Test Pydantic Models
class SimpleResponse(pydantic.BaseModel):
    message: str
    status: str


class NestedResponse(pydantic.BaseModel):
    title: str
    content: dict
    metadata: SimpleResponse


class ComplexResponse(pydantic.BaseModel):
    items: list[str]
    settings: dict[str, str]
    active: bool


@pytest.mark.anyio
async def test_chat_adapter_simple_pydantic_streaming():
    """Test ChatAdapter streaming with a simple pydantic model."""

    class TestSignature(Signature):
        question: str = InputField()
        response: SimpleResponse = OutputField()

    class MyProgram(Module):
        def __init__(self):
            self.predict = Predict(TestSignature)

        def forward(self, question, **kwargs: object):
            return self.predict(question=question, **kwargs)

    async def chat_stream(*args: object, **kwargs: object):
        # Simulate streaming of a pydantic model via ChatAdapter format
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" response"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ## ]]\n\n"))])
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"message": "Hello'))]
        )
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=' world!"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "status":'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=' "success"}'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" completed"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ## ]]"))])

    program: Any = streamify(
        MyProgram(),
        stream_listeners=[
            StreamListener(signature_field_name="response"),
        ],
    )

    with mock.patch("litellm.acompletion", side_effect=chat_stream):  # noqa: SIM117
        with settings.context(lm=LM("openai/gpt-4o-mini", cache=False), adapter=ChatAdapter()):
            output = program(question="Say hello")
            chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    chunks.append(value)  # noqa: PERF401

    # Verify we got chunks for the pydantic field
    assert len(chunks) > 0
    assert chunks[0].signature_field_name == "response"

    # Combine all chunks to verify the content
    full_content = "".join(chunk.chunk for chunk in chunks)
    assert "Hello world!" in full_content
    assert "success" in full_content


@pytest.mark.anyio
async def test_chat_adapter_with_generic_type_annotation():
    class TestSignature(Signature):
        question: str = InputField()
        response: list[str] | int = OutputField()

    class MyProgram(Module):
        def __init__(self):
            self.predict = Predict(TestSignature)

        def forward(self, question, **kwargs: object):
            return self.predict(question=question, **kwargs)

    async def chat_stream(*args: object, **kwargs: object):
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" response"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ## ]]\n\n"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="1"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n[[ ##"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" completed"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" ## ]]"))])

    program: Any = streamify(
        MyProgram(),
        stream_listeners=[
            StreamListener(signature_field_name="response"),
        ],
    )

    with mock.patch("litellm.acompletion", side_effect=chat_stream):  # noqa: SIM117
        with settings.context(lm=LM("openai/gpt-4o-mini", cache=False), adapter=ChatAdapter()):
            output = program(question="Say hello")
            chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    chunks.append(value)  # noqa: PERF401

    assert len(chunks) > 0
    assert chunks[0].signature_field_name == "response"

    full_content = "".join(chunk.chunk for chunk in chunks)
    assert "1" in full_content


@pytest.mark.anyio
async def test_chat_adapter_nested_pydantic_streaming():
    """Test ChatAdapter streaming with nested pydantic model."""

    class TestSignature(Signature):
        question: str = InputField()
        response: NestedResponse = OutputField()

    async def nested_stream(*args: object, **kwargs: object):
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ## response ## ]]\n\n"))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"title": "Test"'))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "content": {"key": "value"}'))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "metadata": {"message": "nested"'))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "status": "ok"}}'))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## completed ## ]]"))]
        )

    program: Any = streamify(
        Predict(TestSignature),
        stream_listeners=[
            StreamListener(signature_field_name="response"),
        ],
    )

    with mock.patch("litellm.acompletion", side_effect=nested_stream):  # noqa: SIM117
        with settings.context(lm=LM("openai/gpt-4o-mini", cache=False), adapter=ChatAdapter()):
            output = program(question="Generate nested response")
            chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    chunks.append(value)  # noqa: PERF401

    assert len(chunks) > 0
    full_content = "".join(chunk.chunk for chunk in chunks)
    assert "nested" in full_content
    assert "Test" in full_content


@pytest.mark.anyio
async def test_chat_adapter_mixed_fields_streaming():
    """Test ChatAdapter streaming with both pydantic and string fields."""

    class TestSignature(Signature):
        question: str = InputField()
        summary: str = OutputField()
        details: SimpleResponse = OutputField()

    async def mixed_stream(*args: object, **kwargs: object):
        # First output field (summary - string)
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ## summary ## ]]\n\n"))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="This is a summary"))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=" of the response"))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## details ## ]]\n\n"))]
        )
        # Second output field (details - pydantic)
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"message": "Detailed info"'))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "status": "complete"}'))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## completed ## ]]"))]
        )

    program: Any = streamify(
        Predict(TestSignature),
        stream_listeners=[
            StreamListener(signature_field_name="summary"),
            StreamListener(signature_field_name="details"),
        ],
    )

    with mock.patch("litellm.acompletion", side_effect=mixed_stream):  # noqa: SIM117
        with settings.context(lm=LM("openai/gpt-4o-mini", cache=False), adapter=ChatAdapter()):
            output = program(question="Generate mixed response")
            summary_chunks = []
            details_chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    if value.signature_field_name == "summary":
                        summary_chunks.append(value)
                    elif value.signature_field_name == "details":
                        details_chunks.append(value)

    # Verify both field types were streamed
    assert len(summary_chunks) > 0
    assert len(details_chunks) > 0

    summary_content = "".join(chunk.chunk for chunk in summary_chunks)
    details_content = "".join(chunk.chunk for chunk in details_chunks)

    assert "summary" in summary_content
    assert "Detailed info" in details_content


@pytest.mark.anyio
async def test_json_adapter_simple_pydantic_streaming():
    """Test JSONAdapter streaming with a simple pydantic model."""

    class TestSignature(Signature):
        question: str = InputField()
        response: SimpleResponse = OutputField()

    async def json_stream(*args: object, **kwargs: object):
        # Simulate JSON streaming with proper bracket balance tracking
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='response"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=":"))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"message"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=': "Hello'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=' JSON!"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "status"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=': "ok"}'))])
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="}"))]
        )  # Close main object

    program: Any = streamify(
        Predict(TestSignature),
        stream_listeners=[
            StreamListener(signature_field_name="response"),
        ],
    )

    with mock.patch("litellm.acompletion", side_effect=json_stream):  # noqa: SIM117
        with settings.context(lm=LM("openai/gpt-4o-mini", cache=False), adapter=JSONAdapter()):
            output = program(question="Say hello in JSON")
            chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    chunks.append(value)  # noqa: PERF401

    assert len(chunks) > 0
    assert chunks[0].signature_field_name == "response"

    full_content = "".join(chunk.chunk for chunk in chunks)
    assert "Hello JSON!" in full_content


@pytest.mark.anyio
async def test_json_adapter_bracket_balance_detection():
    """Test JSONAdapter correctly detects field completion using bracket balance."""

    class TestSignature(Signature):
        question: str = InputField()
        response: ComplexResponse = OutputField()

    async def complex_json_stream(*args: object, **kwargs: object):
        # Test nested objects and arrays for bracket counting
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"'))])
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='response": {'))]
        )  # +1 bracket
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='"items": ["a"'))])
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "b"], '))])
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='"settings": {"key"'))]
        )  # +1 bracket
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=': "value"}, '))]
        )  # -1 bracket
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='"active": true}'))]
        )  # -1 bracket (should end field)
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="}"))]
        )  # Close main object

    program: Any = streamify(
        Predict(TestSignature),
        stream_listeners=[
            StreamListener(signature_field_name="response"),
        ],
    )

    with mock.patch("litellm.acompletion", side_effect=complex_json_stream):  # noqa: SIM117
        with settings.context(lm=LM("openai/gpt-4o-mini", cache=False), adapter=JSONAdapter()):
            output = program(question="Generate complex JSON")
            chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    chunks.append(value)  # noqa: PERF401

    assert len(chunks) > 0
    # Check that the last chunk is marked as the last
    assert chunks[-1].is_last_chunk is True

    full_content = "".join(chunk.chunk for chunk in chunks)

    assert "items" in full_content
    assert "settings" in full_content


@pytest.mark.anyio
async def test_json_adapter_multiple_fields_detection():
    """Test JSONAdapter correctly detects when next field starts."""

    class TestSignature(Signature):
        question: str = InputField()
        first: SimpleResponse = OutputField()
        second: SimpleResponse = OutputField()

    async def multi_field_stream(*args: object, **kwargs: object):
        # First field
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"first": {'))])
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='"message": "first response"'))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "status": "ok"}'))]
        )
        # Second field starts
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "second": {'))])
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='"message": "second response"'))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=', "status": "done"}}'))]
        )

    program: Any = streamify(
        Predict(TestSignature),
        stream_listeners=[
            StreamListener(signature_field_name="first"),
            StreamListener(signature_field_name="second"),
        ],
    )

    with mock.patch("litellm.acompletion", side_effect=multi_field_stream):  # noqa: SIM117
        with settings.context(lm=LM("openai/gpt-4o-mini", cache=False), adapter=JSONAdapter()):
            output = program(question="Generate two responses")
            first_chunks = []
            second_chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    if value.signature_field_name == "first":
                        first_chunks.append(value)
                    elif value.signature_field_name == "second":
                        second_chunks.append(value)

    assert len(first_chunks) > 0
    assert len(second_chunks) > 0

    first_content = "".join(chunk.chunk for chunk in first_chunks)
    second_content = "".join(chunk.chunk for chunk in second_chunks)

    assert "first response" in first_content
    assert "second response" in second_content


def test_stream_listener_could_form_end_identifier_chat_adapter():
    listener = StreamListener(signature_field_name="answer")

    assert listener._could_form_end_identifier("some text [", "ChatAdapter") is True
    assert listener._could_form_end_identifier("some text [[", "ChatAdapter") is True
    assert listener._could_form_end_identifier("some text [[ ", "ChatAdapter") is True
    assert listener._could_form_end_identifier("some text [[ #", "ChatAdapter") is True
    assert listener._could_form_end_identifier("some text [[ ##", "ChatAdapter") is True

    assert listener._could_form_end_identifier("some text [[ ## com", "ChatAdapter") is True
    assert listener._could_form_end_identifier("some text [[ ## completed", "ChatAdapter") is True

    assert listener._could_form_end_identifier("hello world", "ChatAdapter") is False
    assert listener._could_form_end_identifier("some text", "ChatAdapter") is False
    assert listener._could_form_end_identifier("answer: hello", "ChatAdapter") is False


def test_stream_listener_could_form_end_identifier_json_adapter():
    listener = StreamListener(signature_field_name="output")

    assert listener._could_form_end_identifier('some text "', "JSONAdapter") is True
    assert listener._could_form_end_identifier('some text ",', "JSONAdapter") is True
    assert listener._could_form_end_identifier('some text " ', "JSONAdapter") is True
    assert listener._could_form_end_identifier('some text "}', "JSONAdapter") is True

    assert listener._could_form_end_identifier("hello world", "JSONAdapter") is False
    assert listener._could_form_end_identifier("some text", "JSONAdapter") is False


def test_stream_listener_could_form_end_identifier_xml_adapter():
    listener = StreamListener(signature_field_name="result")

    assert listener._could_form_end_identifier("some text <", "XMLAdapter") is True
    assert listener._could_form_end_identifier("some text </", "XMLAdapter") is True
    assert listener._could_form_end_identifier("some text </result", "XMLAdapter") is True

    assert listener._could_form_end_identifier("hello world", "XMLAdapter") is False
    assert listener._could_form_end_identifier("some text", "XMLAdapter") is False


@pytest.mark.anyio
async def test_streaming_reasoning_model():
    """Test streaming behavior for reasoning-capable models using Reasoning.

    This test verifies that:
    1. Reasoning content is extracted from delta.reasoning_content in stream chunks
    2. Reasoning chunks are streamed independently from regular content
    3. The final prediction contains a Reasoning object with the full reasoning content
    """

    class ReasoningSignature(Signature):
        question: str = InputField()
        reasoning: Reasoning = OutputField()
        answer: str = OutputField()

    class MyProgram(Module):
        def __init__(self):
            super().__init__()
            self.predict = Predict(ReasoningSignature)

        def forward(self, question, **kwargs: object):
            return self.predict(question=question, **kwargs)

    async def reasoning_stream(*args: object, **kwargs: object):
        """Simulate streaming from a reasoning model like Claude 3.7 Sonnet"""
        # Reasoning content comes through delta.reasoning_content
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[
                StreamingChoices(delta=Delta(reasoning_content="First, let's think about this problem step by step. "))
            ],
        )
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[StreamingChoices(delta=Delta(reasoning_content="We need to consider the context of a kitchen. "))],
        )
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[
                StreamingChoices(
                    delta=Delta(reasoning_content="The chicken likely wants to reach something on the other side.")
                )
            ],
        )
        # Regular answer content comes through delta.content
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[StreamingChoices(delta=Delta(content="[[ ## answer ## ]]\n"))],
        )
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[StreamingChoices(delta=Delta(content="To"))],
        )
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[StreamingChoices(delta=Delta(content=" get"))],
        )
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[StreamingChoices(delta=Delta(content=" to"))],
        )
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[StreamingChoices(delta=Delta(content=" the"))],
        )
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[StreamingChoices(delta=Delta(content=" other"))],
        )
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[StreamingChoices(delta=Delta(content=" side"))],
        )
        yield ModelResponseStream(
            model="anthropic/claude-3-7-sonnet-20250219",
            choices=[StreamingChoices(delta=Delta(content="!\n\n[[ ## completed ## ]]"))],
        )

    with mock.patch("litellm.acompletion", side_effect=reasoning_stream):  # noqa: SIM117
        with mock.patch("litellm.supports_reasoning", return_value=True):
            program: Any = streamify(
                MyProgram(),
                stream_listeners=[
                    StreamListener(signature_field_name="reasoning"),
                    StreamListener(signature_field_name="answer"),
                ],
            )
            with settings.context(
                lm=LM("anthropic/claude-3-7-sonnet-20250219", cache=False),
                adapter=ChatAdapter(native_response_types=[Reasoning]),
            ):
                output = program(question="Why did a chicken cross the kitchen?")
                reasoning_chunks = []
                answer_chunks = []
                final_prediction = None
                async for value in output:
                    if isinstance(value, StreamResponse):
                        if value.signature_field_name == "reasoning":
                            reasoning_chunks.append(value)
                        elif value.signature_field_name == "answer":
                            answer_chunks.append(value)
                    elif isinstance(value, Prediction):
                        final_prediction = value

                # Verify reasoning chunks were streamed
                assert len(reasoning_chunks) == 3
                assert reasoning_chunks[0].chunk == "First, let's think about this problem step by step. "
                assert reasoning_chunks[1].chunk == "We need to consider the context of a kitchen. "
                assert reasoning_chunks[2].chunk == "The chicken likely wants to reach something on the other side."

                # Verify answer chunks were streamed
                assert len(answer_chunks) > 0
                assert answer_chunks[0].chunk == "To"
                full_answer = "".join([chunk.chunk for chunk in answer_chunks])
                assert full_answer == "To get to the other side!"

                # Verify final prediction has Reasoning object
                assert final_prediction is not None
                assert hasattr(final_prediction, "reasoning")
                assert isinstance(final_prediction.reasoning, Reasoning)
                expected_reasoning = (
                    "First, let's think about this problem step by step. "
                    "We need to consider the context of a kitchen. "
                    "The chicken likely wants to reach something on the other side."
                )
                assert final_prediction.reasoning.content == expected_reasoning


@pytest.mark.anyio
async def test_stream_listener_empty_last_chunk_chat_adapter():
    """Test that StreamListener emits an empty chunk marking field end.

    This test covers the scenario where:
    1. Tokens that cannot form the end identifier are immediately yielded
    2. The last chunk received contains only the marker for the next field (or completion marker)
    3. An empty chunk with is_last_chunk=True is emitted to properly mark field end
    """

    predict = Predict("question->reasoning, answer")

    async def mock_stream(*args: object, **kwargs: object):
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="[[ ## reasoning ## ]]\n"))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="Let's think about this problem step by step. "))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="We need to consider the context of a kitchen. "))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[
                StreamingChoices(delta=Delta(content="The chicken likely wants to reach something on the other side. "))
            ],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## answer ## ]]\n"))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="To get to the other side!"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## completed ## ]]"))],
        )

    with mock.patch("litellm.acompletion", side_effect=mock_stream):
        program: Any = streamify(
            predict,
            stream_listeners=[
                StreamListener(signature_field_name="reasoning"),
                StreamListener(signature_field_name="answer"),
            ],
        )
        with settings.context(lm=LM("openai/gpt-4o-mini", cache=False), adapter=ChatAdapter()):
            output = program(question="Why did the chicken cross the kitchen?")
            all_chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    all_chunks.append(value)  # noqa: PERF401

            # Find answer and judgement chunks
            reasoning_chunks = [c for c in all_chunks if c.signature_field_name == "reasoning"]
            answer_chunks = [c for c in all_chunks if c.signature_field_name == "answer"]

            # The last chunk should be marked as last chunk for both fields.
            assert answer_chunks[-1].is_last_chunk is True
            assert reasoning_chunks[-1].is_last_chunk is True


@pytest.mark.anyio
async def test_stream_listener_empty_last_chunk_json_adapter():
    predict = Predict("question->reasoning, answer")

    async def mock_stream(*args: object, **kwargs: object):
        yield ModelResponseStream(
            model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content='{"reasoning": "'))]
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="Let's think about this problem step by step. "))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="We need to consider the context of a kitchen. "))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[
                StreamingChoices(
                    delta=Delta(content='The chicken likely wants to reach something on the other side. "')
                )
            ],
        )
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=',"answer": "'))])
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content='To get to the other side!"'))],
        )
        yield ModelResponseStream(model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content="\n}"))])

    with mock.patch("litellm.acompletion", side_effect=mock_stream):
        program: Any = streamify(
            predict,
            stream_listeners=[
                StreamListener(signature_field_name="reasoning"),
                StreamListener(signature_field_name="answer"),
            ],
        )
        with settings.context(lm=LM("openai/gpt-4o-mini", cache=False), adapter=JSONAdapter()):
            output = program(question="Why did the chicken cross the kitchen?")
            all_chunks = []
            async for value in output:
                if isinstance(value, StreamResponse):
                    all_chunks.append(value)  # noqa: PERF401

            # Find answer and judgement chunks
            reasoning_chunks = [c for c in all_chunks if c.signature_field_name == "reasoning"]
            answer_chunks = [c for c in all_chunks if c.signature_field_name == "answer"]

            # The last chunk should be marked as last chunk for both fields.
            assert answer_chunks[-1].is_last_chunk is True
            assert reasoning_chunks[-1].is_last_chunk is True


@pytest.mark.anyio
async def test_streaming_reasoning_fallback():
    """Test fallback behavior for non-reasoning models using Reasoning.

    This test verifies that:
    1. For non-reasoning models, reasoning is treated as a regular string field
    2. Reasoning content is streamed through regular adapter parsing (not reasoning_content)
    3. The Reasoning object is created from the parsed string content
    4. Streaming behavior is identical to regular string fields
    """

    class ReasoningSignature(Signature):
        question: str = InputField()
        reasoning: Reasoning = OutputField()
        answer: str = OutputField()

    class MyProgram(Module):
        def __init__(self):
            super().__init__()
            self.predict = Predict(ReasoningSignature)

        def forward(self, question, **kwargs: object):
            return self.predict(question=question, **kwargs)

    async def non_reasoning_stream(*args: object, **kwargs: object):
        """Simulate streaming from a non-reasoning model like GPT-4o-mini.

        The reasoning field is formatted by the adapter as a regular field,
        and content comes through delta.content (not reasoning_content).
        """
        # Reasoning field marker (ChatAdapter format)
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="[[ ## reasoning ## ]]\n"))],
        )
        # Reasoning content as regular text
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="Let"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="'s"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" think"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" step"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" by"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" step"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" about"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" this"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" question"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="."))],
        )
        # Answer field marker
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## answer ## ]]\n"))],
        )
        # Answer content
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="To"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" get"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" to"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" the"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" other"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content=" side"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="!"))],
        )
        yield ModelResponseStream(
            model="gpt-4o-mini",
            choices=[StreamingChoices(delta=Delta(content="\n\n[[ ## completed ## ]]"))],
        )

    with mock.patch("litellm.acompletion", side_effect=non_reasoning_stream):  # noqa: SIM117
        with mock.patch("litellm.supports_reasoning", return_value=False):
            program: Any = streamify(
                MyProgram(),
                stream_listeners=[
                    StreamListener(signature_field_name="reasoning"),
                    StreamListener(signature_field_name="answer"),
                ],
            )
            with settings.context(
                lm=LM("openai/gpt-4o-mini", cache=False),
                adapter=ChatAdapter(),
            ):
                output = program(question="Why did a chicken cross the kitchen?")
                reasoning_chunks = []
                answer_chunks = []
                final_prediction = None
                async for value in output:
                    if isinstance(value, StreamResponse):
                        if value.signature_field_name == "reasoning":
                            reasoning_chunks.append(value)
                        elif value.signature_field_name == "answer":
                            answer_chunks.append(value)
                    elif isinstance(value, Prediction):
                        final_prediction = value

                # Verify reasoning was streamed as regular text
                assert len(reasoning_chunks) > 0
                assert reasoning_chunks[0].chunk == "Let"
                assert reasoning_chunks[1].chunk == "'s"
                full_reasoning = "".join([chunk.chunk for chunk in reasoning_chunks])
                assert full_reasoning == "Let's think step by step about this question."

                # Verify answer chunks were streamed
                assert len(answer_chunks) > 0
                assert answer_chunks[0].chunk == "To"
                full_answer = "".join([chunk.chunk for chunk in answer_chunks])
                assert full_answer == "To get to the other side!"

                # Verify final prediction has Reasoning object created from string
                assert final_prediction is not None
                assert hasattr(final_prediction, "reasoning")
                assert isinstance(final_prediction.reasoning, Reasoning)
                assert final_prediction.reasoning.content == "Let's think step by step about this question."
                # Verify Reasoning object is str-like
                assert str(final_prediction.reasoning) == "Let's think step by step about this question."
