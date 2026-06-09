import asyncio
from typing import TYPE_CHECKING, Any, cast
from unittest import mock

import pydantic
import pytest

from dspy.errors import AdapterParseError
from tests.test_utils import DummyLM

if TYPE_CHECKING:
    from litellm.utils import ModelResponse
try:
    from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse
    from litellm.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse
except ImportError:
    pytest.skip(reason="litellm is not installed", allow_module_level=True)
from openai.types.responses import ResponseOutputMessage

from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types.code import Code
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.clients.lm import LM
from dspy.errors import LMUnexpectedError
from dspy.history import TurnLog
from dspy.predict.predict import Predict
from dspy.task_spec import input_field, make_task_spec, output_field
from tests.adapters.conftest import adapter_format_as_openai, make_adapter_run
from tests.history.turn_fixtures import task_io_turn
from tests.task_spec.helpers import ts


def _structured_output_model_response() -> ModelResponse:
    return ModelResponse(
        choices=[
            Choices(
                message=Message(
                    content='{"output1":"x","output2":true,"output3":{"subfield1":1,"subfield2":1.0},"output4_unannotated":"y"}'
                )
            )
        ],
        model="openai/gpt-4o",
    )


def test_json_adapter_passes_structured_output_when_supported_by_model(make_run):

    class OutputField3(pydantic.BaseModel):
        subfield1: int = pydantic.Field(description="Int subfield 1", ge=0, le=10)
        subfield2: float = pydantic.Field(description="Float subfield 2")

    TestSignature = make_task_spec(
        {
            "input1": input_field("input1", desc="The input 1."),
            "output1": output_field("output1", desc="The output 1."),
            "output2": output_field("output2", type_=bool, desc="Boolean output field"),
            "output3": output_field("output3", type_=OutputField3, desc="Nested output field"),
            "output4_unannotated": output_field("output4_unannotated", desc="Unannotated output field"),
        },
        instructions="Given the fields `input1`, produce the fields `output1`, `output2`, `output3`, `output4_unannotated`.",
    )
    program = Predict(TestSignature)
    run = make_run(lm=LM(model="openai/gpt-4o"), adapter=JSONAdapter())
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = _structured_output_model_response()
        asyncio.run(program(input1="Test input", run=run))

    def clean_schema_extra(field_name, field_info):
        attrs = dict(field_info.__repr_args__())
        if "json_schema_extra" in attrs:
            attrs["json_schema_extra"] = {
                k: v
                for k, v in attrs["json_schema_extra"].items()
                if k != "__dspy_field_type" and (not (k == "desc" and v == f"${{{field_name}}}"))
            }
        return attrs

    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args
    response_format = call_kwargs.get("response_format")
    assert response_format is not None
    assert response_format["type"] == "json_schema"
    schema_props = response_format["json_schema"]["schema"]["properties"]
    assert set(schema_props.keys()) == {"output1", "output2", "output3", "output4_unannotated"}


def test_json_adapter_not_using_structured_outputs_when_not_supported_by_model(make_run):
    TestSignature = make_task_spec(
        {
            "input1": input_field("input1", desc="The input 1."),
            "output1": output_field("output1", desc="The output 1."),
            "output2": output_field("output2", type_=bool, desc="The output 2."),
        },
        instructions="Given the fields `input1`, produce the fields `output1`, `output2`.",
    )
    program = Predict(TestSignature)
    run = make_run(lm=LM(model="fakeprovider/fakemodel"), adapter=JSONAdapter())
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'output1': 'Test output', 'output2': True}"))],
            model="openai/gpt-4o",
        )
        asyncio.run(program(input1="Test input", run=run))
    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args
    assert "response_format" not in call_kwargs


def test_json_adapter_with_structured_outputs_does_not_mutate_original_signature(make_run):

    class OutputField3(pydantic.BaseModel):
        subfield1: int = pydantic.Field(description="Int subfield 1")
        subfield2: float = pydantic.Field(description="Float subfield 2")

    TestSignature = make_task_spec(
        {
            "input1": input_field("input1", desc="The input 1."),
            "output1": output_field("output1", desc="The output 1."),
            "output2": output_field("output2", type_=bool, desc="Boolean output field"),
            "output3": output_field("output3", type_=OutputField3, desc="Nested output field"),
            "output4_unannotated": output_field("output4_unannotated", desc="Unannotated output field"),
        },
        instructions="Given the fields `input1`, produce the fields `output1`, `output2`, `output3`, `output4_unannotated`.",
    )
    run = make_run(lm=LM(model="openai/gpt-4o"), adapter=JSONAdapter())
    program = Predict(TestSignature)
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = _structured_output_model_response()
        asyncio.run(program(input1="Test input", run=run))
    assert program.task_spec == TestSignature


def test_json_adapter_sync_call():
    signature = ts("question -> answer", instructions="Given the fields, produce the outputs.")
    adapter = JSONAdapter()
    lm = DummyLM([{"answer": "Paris"}], adapter=JSONAdapter())
    result = asyncio.run(
        adapter(
            lm=lm,
            config={},
            task_spec=signature,
            demos=[],
            inputs={"question": "What is the capital of France?"},
            run=make_adapter_run(lm=lm, adapter=adapter),
        )
    )
    assert result == [{"answer": "Paris"}]


@pytest.mark.asyncio
async def test_json_adapter_async_call():
    signature = ts("question -> answer", instructions="Given the fields, produce the outputs.")
    adapter = JSONAdapter()
    lm = DummyLM([{"answer": "Paris"}], adapter=JSONAdapter())
    result = await adapter(
        lm=lm,
        config={},
        task_spec=signature,
        demos=[],
        inputs={"question": "What is the capital of France?"},
        run=make_adapter_run(lm=lm, adapter=adapter),
    )
    assert result == [{"answer": "Paris"}]


def test_json_adapter_on_pydantic_model(make_run):
    from litellm.utils import Choices, Message, ModelResponse

    class User(pydantic.BaseModel):
        id: int
        name: str
        email: str

    class Answer(pydantic.BaseModel):
        analysis: str
        result: str

    TestSignature = make_task_spec(
        {
            "user": input_field("user", type_=User, desc="The user who asks the question"),
            "question": input_field("question", desc="Question the user asks"),
            "answer": output_field("answer", type_=Answer, desc="Answer to this question"),
        },
        instructions="Given the fields `user`, `question`, produce the fields `answer`.",
    )
    program = Predict(TestSignature)
    run = make_run(lm=LM(model="openai/gpt-4o"), adapter=JSONAdapter())
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    message=Message(
                        content="{'answer': {'analysis': 'Paris is the capital of France', 'result': 'Paris'}}"
                    )
                )
            ],
            model="openai/gpt-4o",
        )
        result = asyncio.run(
            program(
                user=User(id=5, name="name_test", email="email_test"),
                question="What is the capital of France?",
                run=run,
            )
        )
        mock_completion.assert_called_once()
        _, call_kwargs = mock_completion.call_args
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        content = call_kwargs["messages"][0]["content"]
        assert content is not None
        expected_input_fields = (
            "1. `user` (User): The user who asks the question\n2. `question` (str): Question the user asks\n"
        )
        assert expected_input_fields in content
        expected_output_fields = "1. `answer` (Answer): Answer to this question\n"
        assert expected_output_fields in content
        expected_input_structure = "[[ ## user ## ]]\n{user}\n\n[[ ## question ## ]]\n{question}\n\n"
        assert expected_input_structure in content
        expected_output_structure = 'Outputs will be a JSON object with the following fields.\n\n{\n  "answer": "{answer}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"object\\", \\"properties\\": {\\"analysis\\": {\\"type\\": \\"string\\", \\"title\\": \\"Analysis\\"}, \\"result\\": {\\"type\\": \\"string\\", \\"title\\": \\"Result\\"}}, \\"required\\": [\\"analysis\\", \\"result\\"], \\"title\\": \\"Answer\\"}"\n}'
        assert expected_output_structure in content
        assert call_kwargs["messages"][1]["role"] == "user"
        user_message_content = call_kwargs["messages"][1]["content"]
        assert user_message_content is not None
        expected_input_data = '[[ ## user ## ]]\n{"id": 5, "name": "name_test", "email": "email_test"}\n\n[[ ## question ## ]]\nWhat is the capital of France?\n\n'
        assert expected_input_data in user_message_content
        assert result.answer.analysis == "Paris is the capital of France"
        assert result.answer.result == "Paris"


def test_json_adapter_parse_raise_error_on_mismatch_fields():
    signature = ts("question -> answer", instructions="Given the fields, produce the outputs.")
    adapter = JSONAdapter()
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'answer1': 'Paris'}"))], model="openai/gpt-4o"
        )
        lm = LM(model="openai/gpt-4o-mini")
        with pytest.raises(AdapterParseError) as e:
            asyncio.run(
                adapter(
                    lm=lm,
                    config={},
                    task_spec=signature,
                    demos=[],
                    inputs={"question": "What is the capital of France?"},
                    run=make_adapter_run(lm=lm, adapter=adapter),
                )
            )
    assert e.value.adapter_name == "JSONAdapter"
    assert e.value.task_spec == signature
    assert e.value.lm_response == "{'answer1': 'Paris'}"
    assert e.value.parsed_result == {"answer1": "Paris"}
    assert "unexpected field(s): ['answer1']" in str(e.value)


def test_json_adapter_with_tool():
    MySignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "answer": output_field("answer", desc="The answer."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Answer question with the help of the tools",
    )

    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny"

    def get_population(country: str, year: int) -> str:
        return f"The population of {country} in {year} is 1000000"

    tools = [
        Tool(get_weather, description="Get the weather for a city"),
        Tool(get_population, description="Get the population for a country"),
    ]
    adapter = JSONAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter,
        task_spec=MySignature,
        demos=[],
        inputs={"question": "What is the weather in Tokyo?", "tools": tools},
    )
    assert len(messages) == 2
    assert ToolCalls.description() in messages[0]["content"]
    assert "What is the weather in Tokyo?" in messages[1]["content"]
    assert "get_weather" in messages[1]["content"]
    assert "get_population" in messages[1]["content"]
    assert "{'city': {'type': 'string'}}" in messages[1]["content"]
    assert "{'country': {'type': 'string'}, 'year': {'type': 'integer'}}" in messages[1]["content"]
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content='{"answer":"sunny"}'))],
            model="openai/gpt-4o-mini",
        )
        lm = LM(model="openai/gpt-4o-mini")
        asyncio.run(
            adapter(
                lm=lm,
                config={},
                task_spec=MySignature,
                demos=[],
                inputs={"question": "What is the weather in Tokyo?", "tools": tools},
                run=make_adapter_run(lm=lm, adapter=adapter),
            )
        )
    mock_completion.assert_called_once()
    _, call_kwargs = mock_completion.call_args
    assert len(call_kwargs["tools"]) > 0
    assert call_kwargs["tools"][0] == {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
    assert call_kwargs["tools"][1] == {
        "type": "function",
        "function": {
            "name": "get_population",
            "description": "Get the population for a country",
            "parameters": {
                "type": "object",
                "properties": {"country": {"type": "string"}, "year": {"type": "integer"}},
                "required": ["country", "year"],
            },
        },
    }


def test_json_adapter_with_code():
    CodeAnalysis = make_task_spec(
        {
            "code": input_field("code", type_=Code, desc="The code."),
            "result": output_field("result", desc="The result."),
        },
        instructions="Analyze the time complexity of the code",
    )
    adapter = JSONAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter, task_spec=CodeAnalysis, demos=[], inputs={"code": "print('Hello, world!')"}
    )
    assert len(messages) == 2
    assert Code.description() in messages[0]["content"]
    assert "print('Hello, world!')" in messages[1]["content"]
    CodeGeneration = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "code": output_field("code", type_=Code, desc="The code."),
        },
        instructions="Generate code to answer the question",
    )
    adapter = JSONAdapter()
    lm = LM(model="openai/gpt-4o-mini")
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'code': 'print(\"Hello, world!\")'}"))],
            model="openai/gpt-4o-mini",
        )
        result = asyncio.run(
            adapter(
                lm=lm,
                config={},
                task_spec=CodeGeneration,
                demos=[],
                inputs={"question": "Write a python program to print 'Hello, world!'"},
                run=make_adapter_run(lm=lm, adapter=adapter),
            )
        )
        assert result[0]["code"].code == 'print("Hello, world!")'


def test_json_adapter_formats_conversation_history():
    MySignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "turn_log": input_field("turn_log", type_=TurnLog, desc="The history."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `question`, `turn_log`, produce the fields `answer`.",
    )
    history = TurnLog.model_validate(
        {
            "turns": [
                task_io_turn(question="What is the capital of France?", answer="Paris"),
                task_io_turn(question="What is the capital of Germany?", answer="Berlin"),
            ],
        }
    )
    adapter = JSONAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter,
        task_spec=MySignature,
        demos=[],
        inputs={"question": "What is the capital of France?", "turn_log": history},
    )
    assert len(messages) == 6
    assert messages[1]["content"] == "[[ ## question ## ]]\nWhat is the capital of France?"
    assert messages[2]["content"] == '{\n  "answer": "Paris"\n}'
    assert messages[3]["content"] == "[[ ## question ## ]]\nWhat is the capital of Germany?"
    assert messages[4]["content"] == '{\n  "answer": "Berlin"\n}'


@pytest.mark.asyncio
async def test_json_adapter_on_pydantic_model_async(make_run):
    from litellm.utils import Choices, Message, ModelResponse

    class User(pydantic.BaseModel):
        id: int
        name: str
        email: str

    class Answer(pydantic.BaseModel):
        analysis: str
        result: str

    TestSignature = make_task_spec(
        {
            "user": input_field("user", type_=User, desc="The user who asks the question"),
            "question": input_field("question", desc="Question the user asks"),
            "answer": output_field("answer", type_=Answer, desc="Answer to this question"),
        },
        instructions="Given the fields `user`, `question`, produce the fields `answer`.",
    )
    program = Predict(TestSignature)
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    message=Message(
                        content="{'answer': {'analysis': 'Paris is the capital of France', 'result': 'Paris'}}"
                    )
                )
            ],
            model="openai/gpt-4o",
        )
        run = make_run(lm=LM(model="openai/gpt-4o"), adapter=JSONAdapter())
        result = await program(
            user=User(id=5, name="name_test", email="email_test"),
            question="What is the capital of France?",
            run=run,
        )
        mock_completion.assert_called_once()
        _, call_kwargs = mock_completion.call_args
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        content = call_kwargs["messages"][0]["content"]
        assert content is not None
        expected_input_fields = (
            "1. `user` (User): The user who asks the question\n2. `question` (str): Question the user asks\n"
        )
        assert expected_input_fields in content
        expected_output_fields = "1. `answer` (Answer): Answer to this question\n"
        assert expected_output_fields in content
        expected_input_structure = "[[ ## user ## ]]\n{user}\n\n[[ ## question ## ]]\n{question}\n\n"
        assert expected_input_structure in content
        expected_output_structure = 'Outputs will be a JSON object with the following fields.\n\n{\n  "answer": "{answer}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"object\\", \\"properties\\": {\\"analysis\\": {\\"type\\": \\"string\\", \\"title\\": \\"Analysis\\"}, \\"result\\": {\\"type\\": \\"string\\", \\"title\\": \\"Result\\"}}, \\"required\\": [\\"analysis\\", \\"result\\"], \\"title\\": \\"Answer\\"}"\n}'
        assert expected_output_structure in content
        assert call_kwargs["messages"][1]["role"] == "user"
        user_message_content = call_kwargs["messages"][1]["content"]
        assert user_message_content is not None
        expected_input_data = '[[ ## user ## ]]\n{"id": 5, "name": "name_test", "email": "email_test"}\n\n[[ ## question ## ]]\nWhat is the capital of France?\n\n'
        assert expected_input_data in user_message_content
        assert result.answer.analysis == "Paris is the capital of France"
        assert result.answer.result == "Paris"


def test_json_adapter_does_not_fallback_to_json_mode_on_structured_output_lm_error(make_run):
    TestSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", desc="String output field"),
        },
        instructions="Given the fields `question`, produce the fields `answer`.",
    )
    run = make_run(lm=LM(model="openai/gpt-4o-mini"), adapter=JSONAdapter())
    program = Predict(TestSignature)
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.side_effect = RuntimeError("Structured output failed!")
        with pytest.raises(LMUnexpectedError, match="Structured output failed"):
            asyncio.run(program(question="Dummy question!", run=run))
        assert mock_completion.call_count == 1
        _, first_call_kwargs = mock_completion.call_args_list[0]
        assert first_call_kwargs.get("response_format")["type"] == "json_schema"


def test_json_adapter_json_mode_no_structured_outputs(make_run):
    TestSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", desc="String output field"),
        },
        instructions="Given the fields `question`, produce the fields `answer`.",
    )
    run = make_run(lm=LM(model="openai/gpt-4o"), adapter=JSONAdapter())
    program = Predict(TestSignature)
    with (
        mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion,
        mock.patch("litellm.get_supported_openai_params") as mock_get_supported_openai_params,
        mock.patch("litellm.supports_response_schema") as mock_supports_response_schema,
    ):
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'answer': 'Test output'}"))]
        )
        mock_get_supported_openai_params.return_value = ["response_format"]
        mock_supports_response_schema.return_value = False
        result = asyncio.run(program(question="Dummy question!", run=run))
        assert mock_completion.call_count == 1
        assert result.answer == "Test output"
        _, call_kwargs = mock_completion.call_args_list[0]
        assert call_kwargs.get("response_format") == {"type": "json_object"}


@pytest.mark.asyncio
async def test_json_adapter_json_mode_no_structured_outputs_async(make_run):
    TestSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", desc="String output field"),
        },
        instructions="Given the fields `question`, produce the fields `answer`.",
    )
    program = Predict(TestSignature)
    with (
        mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_acompletion,
        mock.patch("litellm.get_supported_openai_params") as mock_get_supported_openai_params,
        mock.patch("litellm.supports_response_schema") as mock_supports_response_schema,
    ):
        mock_acompletion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'answer': 'Test output'}"))]
        )
        mock_get_supported_openai_params.return_value = ["response_format"]
        mock_supports_response_schema.return_value = False
        run = make_run(lm=LM(model="openai/gpt-4o"), adapter=JSONAdapter())
        result = await program(question="Dummy question!", run=run)
        assert mock_acompletion.call_count == 1
        assert result.answer == "Test output"
        _, call_kwargs = mock_acompletion.call_args_list[0]
        assert call_kwargs.get("response_format") == {"type": "json_object"}


@pytest.mark.asyncio
async def test_json_adapter_does_not_fallback_to_json_mode_on_structured_output_lm_error_async(make_run):
    TestSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", desc="String output field"),
        },
        instructions="Given the fields `question`, produce the fields `answer`.",
    )
    program = Predict(TestSignature)
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = RuntimeError("Structured output failed!")
        run = make_run(lm=LM(model="openai/gpt-4o-mini"), adapter=JSONAdapter())
        with pytest.raises(LMUnexpectedError, match="Structured output failed"):
            await program(question="Dummy question!", run=run)
        assert mock_acompletion.call_count == 1
        _, first_call_kwargs = mock_acompletion.call_args_list[0]
        assert first_call_kwargs.get("response_format")["type"] == "json_schema"


def test_error_message_on_json_adapter_failure(make_run):
    TestSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", desc="String output field"),
        },
        instructions="Given the fields `question`, produce the fields `answer`.",
    )
    program = Predict(TestSignature)
    run = make_run(lm=LM(model="openai/gpt-4o-mini"), adapter=JSONAdapter())
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.side_effect = RuntimeError("RuntimeError!")
        with pytest.raises(LMUnexpectedError) as error:
            asyncio.run(program(question="Dummy question!", run=run))
        assert "RuntimeError!" in str(error.value)
        mock_completion.side_effect = ValueError("ValueError!")
        with pytest.raises(LMUnexpectedError) as error:
            asyncio.run(program(question="Dummy question!", run=run))
        assert "ValueError!" in str(error.value)


@pytest.mark.asyncio
async def test_error_message_on_json_adapter_failure_async(make_run):
    TestSignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", desc="String output field"),
        },
        instructions="Given the fields `question`, produce the fields `answer`.",
    )
    program = Predict(TestSignature)
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_acompletion:
        run = make_run(lm=LM(model="openai/gpt-4o-mini"), adapter=JSONAdapter())
        mock_acompletion.side_effect = RuntimeError("RuntimeError!")
        with pytest.raises(LMUnexpectedError) as error:
            await program(question="Dummy question!", run=run)
        assert "RuntimeError!" in str(error.value)
        mock_acompletion.side_effect = ValueError("ValueError!")
        with pytest.raises(LMUnexpectedError) as error:
            await program(question="Dummy question!", run=run)
        assert "ValueError!" in str(error.value)


def test_json_adapter_toolcalls_native_function_calling():
    MySignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "answer": output_field("answer", desc="The answer."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `answer`, `tool_calls`.",
    )

    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny"

    tools = [Tool(get_weather, description="Get the weather for a city")]
    adapter = JSONAdapter(use_native_function_calling=True)
    lm = LM(model="openai/gpt-4o-mini")
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                function=Function(arguments='{"city":"Paris"}', name="get_weather"),
                                id="call_pQm8ajtSMxgA0nrzK2ivFmxG",
                                type="function",
                            )
                        ],
                    ),
                )
            ],
            model="openai/gpt-4o-mini",
        )
        result = asyncio.run(
            adapter(
                lm=lm,
                config={},
                task_spec=MySignature,
                demos=[],
                inputs={"question": "What is the weather in Paris?", "tools": tools},
                run=make_adapter_run(lm=lm, adapter=adapter),
            )
        )
        assert result[0]["tool_calls"] == ToolCalls(
            tool_calls=[
                ToolCalls.ToolCall(id="call_pQm8ajtSMxgA0nrzK2ivFmxG", name="get_weather", args={"city": "Paris"})
            ]
        )
        assert result[0]["answer"] is None
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'answer': 'Paris'}"))], model="openai/gpt-4o-mini"
        )
        result = asyncio.run(
            adapter(
                lm=lm,
                config={},
                task_spec=MySignature,
                demos=[],
                inputs={"question": "What is the weather in Paris?", "tools": tools},
                run=make_adapter_run(lm=lm, adapter=adapter),
            )
        )
        assert result[0]["answer"] == "Paris"
        assert result[0]["tool_calls"] is None


def test_json_adapter_toolcalls_no_native_function_calling():
    MySignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "tools": input_field("tools", type_=list[Tool], desc="The tools."),
            "answer": output_field("answer", desc="The answer."),
            "tool_calls": output_field("tool_calls", type_=ToolCalls, desc="The tool calls."),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `answer`, `tool_calls`.",
    )

    def get_weather(city: str) -> str:
        return f"The weather in {city} is sunny"

    tools = [Tool(get_weather, description="Get the weather for a city")]
    with mock.patch("dspy.adapters.structured_output.get_structured_outputs_response_format") as mock_structured:
        with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
            mock_completion.return_value = ModelResponse(
                choices=[Choices(message=Message(content="{'answer': 'sunny', 'tool_calls': {'tool_calls': []}}"))],
                model="openai/gpt-4o-mini",
            )
            adapter = JSONAdapter(use_native_function_calling=False)
            lm = LM(model="openai/gpt-4o-mini")
            asyncio.run(
                adapter(
                    lm=lm,
                    config={},
                    task_spec=MySignature,
                    demos=[],
                    inputs={"question": "What is the weather in Tokyo?", "tools": tools},
                    run=make_adapter_run(lm=lm, adapter=adapter),
                )
            )
        mock_structured.assert_not_called()
        mock_completion.assert_called_once()
        _, call_kwargs = mock_completion.call_args
        assert call_kwargs["response_format"] == {"type": "json_object"}


def test_json_adapter_native_reasoning():
    MySignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "reasoning": output_field("reasoning", type_=Reasoning, desc="The reasoning."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Given the fields `question`, produce the fields `reasoning`, `answer`.",
    )
    adapter = JSONAdapter()
    from dspy.core.types import LMProviderOptions

    lm = LM(
        model="anthropic/claude-3-7-sonnet-20250219",
        provider_options=LMProviderOptions(extensions={"reasoning": {"effort": "low"}}),
    )
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(
                    message=Message(
                        content="{'answer': 'Paris'}",
                        reasoning_content="Step-by-step thinking about the capital of France",
                    )
                )
            ],
            model="anthropic/claude-3-7-sonnet-20250219",
        )
        modified_signature, _, _ = adapter._call_preprocess(
            lm,
            {},
            MySignature,
            {"question": "What is the capital of France?"},
        )
        assert "reasoning" not in modified_signature.output_fields
        result = asyncio.run(
            adapter(
                lm=lm,
                config={},
                task_spec=MySignature,
                demos=[],
                inputs={"question": "What is the capital of France?"},
                run=make_adapter_run(lm=lm, adapter=adapter),
            )
        )
        assert result[0]["reasoning"] == Reasoning(content="Step-by-step thinking about the capital of France")


def test_json_adapter_with_responses_api(make_run):
    TestSignature = ts("question -> answer", instructions="Given the fields `question`, produce the fields `answer`.")
    api_response = ResponsesAPIResponse(
        id="resp_1",
        created_at=0,
        error=None,
        incomplete_details=None,
        instructions=None,
        model="openai/gpt-4o",
        object="response",
        output=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="completed",
                content=cast(
                    "Any",
                    [{"type": "output_text", "text": '{"answer": "Washington, D.C."}', "annotations": []}],
                ),
            )
        ],
        metadata={},
        parallel_tool_calls=False,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        max_output_tokens=None,
        previous_response_id=None,
        reasoning=None,
        status="completed",
        text=None,
        truncation="disabled",
        usage=ResponseAPIUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        user=None,
    )
    lm = LM(model="openai/gpt-4o", model_type="responses")
    run = make_run(lm=lm, adapter=JSONAdapter())
    program = Predict(TestSignature)
    with mock.patch("litellm.aresponses", new_callable=mock.AsyncMock, return_value=api_response) as mock_responses:
        result = asyncio.run(program(question="What is the capital of the USA?", run=run))
    assert result.answer == "Washington, D.C."
    mock_responses.assert_called_once()
    call_kwargs = mock_responses.call_args.kwargs
    assert "response_format" not in call_kwargs
    assert "text" in call_kwargs
    assert isinstance(call_kwargs["text"]["format"], dict)
    assert isinstance(call_kwargs["text"]["format"]["name"], str)
    assert call_kwargs["text"]["format"]["type"] == "json_schema"
    assert isinstance(call_kwargs["text"]["format"]["schema"], dict)


def test_format_system_message():
    MySignature = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "answers": output_field("answers", type_=list[str], desc="The answers."),
            "scores": output_field("scores", type_=list[float], desc="The scores."),
        },
        instructions="Answer the question with multiple answers and scores",
    )
    adapter = JSONAdapter()
    system_message = adapter.format_system_message(MySignature)
    expected_system_message = 'Your input fields are:\n1. `question` (str): The question.\nYour output fields are:\n1. `answers` (list[str]): The answers.\n2. `scores` (list[float]): The scores.\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\nInputs will have the following structure:\n\n[[ ## question ## ]]\n{question}\n\nOutputs will be a JSON object with the following fields.\n\n{\n  "answers": "{answers}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"array\\", \\"items\\": {\\"type\\": \\"string\\"}}",\n  "scores": "{scores}        # note: the value you produce must adhere to the JSON schema: {\\"type\\": \\"array\\", \\"items\\": {\\"type\\": \\"number\\"}}"\n}\nIn adhering to this structure, your objective is: \n        Answer the question with multiple answers and scores'
    assert system_message == expected_system_message


def test_json_adapter_parse_raises_on_unexpected_fields():
    signature = ts("question -> answer", instructions="Given the fields, produce the outputs.")
    adapter = JSONAdapter()
    with pytest.raises(AdapterParseError, match="unexpected field\\(s\\): \\['extra'\\]"):
        adapter.parse(task_spec=signature, completion='{"answer": "Paris", "extra": "noise"}')
