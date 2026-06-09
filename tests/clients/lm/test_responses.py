import json
from typing import Any, cast
from unittest import mock

import pydantic
import pytest

try:
    from litellm.types.llms.openai import ResponsesAPIResponse
except ImportError:
    pytest.skip(reason="litellm is not installed", allow_module_level=True)
from openai.types.responses import ResponseOutputMessage, ResponseReasoningItem
from openai.types.responses.response_reasoning_item import Summary

from dspy.clients.lm import LM
from dspy.core.types import LMConfig, LMProviderOptions
from dspy.runtime.usage_tracker import track_usage
from tests.clients.lm.conftest import _request, make_response, run_async


def test_responses_api(make_run):
    api_response = make_response(
        output_blocks=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="completed",
                content=cast(
                    "Any",
                    [{"type": "output_text", "text": "This is a test answer from responses API.", "annotations": []}],
                ),
            ),
            ResponseReasoningItem(
                id="reasoning_1",
                type="reasoning",
                summary=[Summary(type="summary_text", text="This is a dummy reasoning.")],
            ),
        ]
    )
    with mock.patch("litellm.aresponses", autospec=True, return_value=api_response) as dspy_responses:
        lm = LM(model="openai/gpt-5-mini", model_type="responses", temperature=1.0, max_tokens=16000)
        lm_result = run_async(lm(_request(lm, prompt="openai query"), run=make_run(lm=lm)))
        assert lm_result.text == "This is a test answer from responses API."
        assert lm_result.reasoning_content == "This is a dummy reasoning."
        dspy_responses.assert_called_once()
        assert dspy_responses.call_args.kwargs["model"] == "openai/gpt-5-mini"


def test_lm_replaces_system_with_developer_role(make_run):
    with mock.patch(
        "dspy.clients.lm.transport.alitellm_responses_completion", return_value={"choices": []}
    ) as mock_completion:
        lm = LM("openai/gpt-4o-mini", model_type="responses", use_developer_role=True)
        run_async(lm(_request(lm, messages=[{"role": "system", "content": "hi"}]), run=make_run(lm=lm)))
        assert mock_completion.call_args.kwargs["request"]["input"][0]["role"] == "developer"


def test_responses_api_tool_calls(litellm_test_server, make_run):
    api_base, _ = litellm_test_server
    expected_tool_call = {
        "type": "function_call",
        "name": "get_weather",
        "arguments": json.dumps({"city": "Paris"}),
        "call_id": "call_1",
        "status": "completed",
        "id": "call_1",
    }
    api_response = make_response(output_blocks=[expected_tool_call])
    with mock.patch("litellm.aresponses", autospec=True, return_value=api_response) as dspy_responses:
        lm = LM(
            model="openai/dspy-test-model",
            provider_options=LMProviderOptions(api_base=api_base, api_key="fakekey"),
            model_type="responses",
        )
        lm_result = run_async(lm(_request(lm, prompt="openai query"), run=make_run(lm=lm)))
        tool_call = lm_result.outputs[0].tool_calls[0]
        assert tool_call.name == expected_tool_call["name"]
        assert tool_call.args == {"city": "Paris"}
        assert tool_call.id == expected_tool_call["call_id"]
        dspy_responses.assert_called_once()
        assert dspy_responses.call_args.kwargs["model"] == "openai/dspy-test-model"


def test_reasoning_effort_responses_api(make_run):
    with mock.patch("litellm.aresponses", mock.AsyncMock(return_value={"choices": []})) as mock_responses:
        lm = LM(
            model="openai/gpt-5",
            model_type="responses",
            max_tokens=16000,
            temperature=1.0,
            provider_options=LMProviderOptions(extensions={"reasoning": {"effort": "low"}}),
        )
        run_async(lm(_request(lm, prompt="openai query"), run=make_run(lm=lm)))
        call_kwargs = mock_responses.call_args.kwargs
        assert "reasoning_effort" not in call_kwargs
        assert call_kwargs["reasoning"]["effort"] == "low"


def test_responses_api_converts_images_correctly():
    from dspy.clients.openai_format.responses_request import to_openai_responses_request
    from dspy.core.types import LMConfig, LMImagePart, LMRequest, LMTextPart, User

    base64_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    request = LMRequest(
        model="openai/gpt-5-mini",
        messages=[
            User(
                LMTextPart(text="What's in this image?"),
                LMImagePart(url=base64_url, media_type="image/png"),
            )
        ],
        config=LMConfig(),
    )
    result = to_openai_responses_request(request)
    assert "input" in result
    assert len(result["input"]) == 1
    assert result["input"][0]["role"] == "user"
    content = result["input"][0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "input_text"
    assert content[0]["text"] == "What's in this image?"
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"] == base64_url

    url_request = LMRequest(
        model="openai/gpt-5-mini",
        messages=[User(LMImagePart(url="https://example.com/image.jpg", media_type="image/jpeg"))],
        config=LMConfig(),
    )
    result = to_openai_responses_request(url_request)
    content = result["input"][0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "input_image"
    assert content[0]["image_url"] == "https://example.com/image.jpg"


def test_responses_api_converts_files_correctly():
    from dspy.clients.openai_format.responses_request import to_openai_responses_request
    from dspy.core.types import LMBinaryPart, LMConfig, LMRequest, LMTextPart, User

    request = LMRequest(
        model="openai/gpt-5-mini",
        messages=[
            User(
                LMTextPart(text="Analyze this file"),
                LMBinaryPart(
                    data="SGVsbG8gV29ybGQ=",
                    media_type="text/plain",
                    filename="test.txt",
                ),
            )
        ],
        config=LMConfig(),
    )
    result = to_openai_responses_request(request)
    assert "input" in result
    assert len(result["input"]) == 1
    assert result["input"][0]["role"] == "user"
    content = result["input"][0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "input_text"
    assert content[0]["text"] == "Analyze this file"
    assert content[1]["type"] == "input_file"
    assert content[1]["file_data"] == "data:text/plain;base64,SGVsbG8gV29ybGQ="
    assert content[1]["filename"] == "test.txt"

    file_id_request = LMRequest(
        model="openai/gpt-5-mini",
        messages=[User(LMBinaryPart(file_id="file-abc123", filename="document.pdf", media_type="application/pdf"))],
        config=LMConfig(),
    )
    result = to_openai_responses_request(file_id_request)
    content = result["input"][0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "input_file"
    assert content[0]["file_id"] == "file-abc123"
    assert content[0]["filename"] == "document.pdf"

    all_fields_request = LMRequest(
        model="openai/gpt-5-mini",
        messages=[
            User(
                LMBinaryPart(
                    data="JVBERi0xLjQ=",
                    filename="report.pdf",
                    media_type="application/pdf",
                )
            )
        ],
        config=LMConfig(),
    )
    result = to_openai_responses_request(all_fields_request)
    content = result["input"][0]["content"]
    assert content[0]["type"] == "input_file"
    assert content[0]["file_data"] == "data:application/pdf;base64,JVBERi0xLjQ="
    assert content[0]["filename"] == "report.pdf"


def test_responses_api_preserves_multi_message_structure(make_run):
    from dspy.clients.openai_format.responses_request import to_openai_responses_request
    from dspy.core.types import Assistant, LMConfig, LMRequest, System, User

    request = LMRequest(
        model="openai/gpt-5-mini",
        messages=[
            System("You are a helpful assistant."),
            User("What is 2+2?"),
            Assistant("4"),
            User("And 3+3?"),
        ],
        config=LMConfig(),
    )
    result = to_openai_responses_request(request)
    assert "input" in result
    assert len(result["input"]) == 4
    assert result["input"][0]["role"] == "system"
    assert result["input"][0]["content"] == [{"type": "input_text", "text": "You are a helpful assistant."}]
    assert result["input"][1]["role"] == "user"
    assert result["input"][1]["content"] == [{"type": "input_text", "text": "What is 2+2?"}]
    assert result["input"][2]["role"] == "assistant"
    assert result["input"][2]["content"] == [{"type": "input_text", "text": "4"}]
    assert result["input"][3]["role"] == "user"
    assert result["input"][3]["content"] == [{"type": "input_text", "text": "And 3+3?"}]


def test_responses_api_with_image_input(make_run):
    api_response = make_response(
        output_blocks=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="completed",
                content=cast(
                    "Any",
                    [{"type": "output_text", "text": "This is a test answer with image input.", "annotations": []}],
                ),
            )
        ]
    )
    with mock.patch("litellm.aresponses", autospec=True, return_value=api_response) as dspy_responses:
        lm = LM(model="openai/gpt-5-mini", model_type="responses", temperature=1.0, max_tokens=16000)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                        },
                    },
                ],
            }
        ]
        lm_result = run_async(lm(_request(lm, messages=messages), run=make_run(lm=lm)))
        assert lm_result.text == "This is a test answer with image input."
        dspy_responses.assert_called_once()
        call_args = dspy_responses.call_args.kwargs
        assert "input" in call_args
        content = call_args["input"][0]["content"]
        image_content = [c for c in content if c.get("type") == "input_image"]
        assert len(image_content) == 1
        assert (
            image_content[0]["image_url"]
            == "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )


def test_responses_api_with_pydantic_model_input(make_run):
    api_response = make_response(
        output_blocks=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="completed",
                content=cast(
                    "Any",
                    [
                        {
                            "type": "output_text",
                            "text": '{"answer" : "This is a good test answer", "number" : 42}',
                            "annotations": [],
                        }
                    ],
                ),
            )
        ]
    )
    lm = LM(model="openai/gpt-5-mini", model_type="responses", temperature=1.0, max_tokens=16000)

    class TestModel(pydantic.BaseModel):
        answer: str
        number: int

    with mock.patch("litellm.aresponses", autospec=True, return_value=api_response) as dspy_responses:
        lm_result = run_async(
            lm(
                _request(lm, prompt="What is a good test answer?", config=LMConfig(response_format=TestModel)),
                run=make_run(lm=lm),
            )
        )
    TestModel.model_validate_json(lm_result.text)
    dspy_responses.assert_called_once()
    call_args = dspy_responses.call_args.kwargs
    assert "text" in call_args
    response_format = call_args["text"]["format"]
    assert response_format == {
        "name": TestModel.__name__,
        "type": "json_schema",
        "schema": TestModel.model_json_schema(),
    }


def test_responses_api_with_none_usage(make_run):

    api_response = ResponsesAPIResponse(
        id="resp_1",
        created_at=0,
        error=None,
        incomplete_details=cast("Any", {"reason": "max_output_tokens"}),
        instructions=None,
        model="openai/gpt-5-mini",
        object="response",
        output=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="incomplete",
                content=cast(
                    "Any",
                    [{"type": "output_text", "text": "Partial response that was truncated", "annotations": []}],
                ),
            )
        ],
        metadata={},
        parallel_tool_calls=False,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        max_output_tokens=100,
        previous_response_id=None,
        reasoning=None,
        status="incomplete",
        text=None,
        truncation="disabled",
        usage=None,
        user=None,
    )
    with mock.patch("litellm.aresponses", autospec=True, return_value=api_response):
        lm = LM(model="openai/gpt-5-mini", model_type="responses", temperature=1.0, max_tokens=16000)
        run = make_run(lm=lm)
        with track_usage(run) as tracker:
            result = run_async(lm(_request(lm, prompt="test query"), run=run))
        assert result.text == "Partial response that was truncated"
        assert lm.call_log[-1].usage == {}
        assert tracker.get_total_tokens() == {}


@pytest.mark.asyncio
async def test_responses_api_with_none_usage_async(make_run):
    api_response = ResponsesAPIResponse(
        id="resp_1",
        created_at=0,
        error=None,
        incomplete_details=cast("Any", {"reason": "max_output_tokens"}),
        instructions=None,
        model="openai/gpt-5-mini",
        object="response",
        output=[
            ResponseOutputMessage(
                id="msg_1",
                type="message",
                role="assistant",
                status="incomplete",
                content=cast(
                    "Any",
                    [{"type": "output_text", "text": "Partial async response", "annotations": []}],
                ),
            )
        ],
        metadata={},
        parallel_tool_calls=False,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        max_output_tokens=100,
        previous_response_id=None,
        reasoning=None,
        status="incomplete",
        text=None,
        truncation="disabled",
        usage=None,
        user=None,
    )
    with mock.patch("litellm.aresponses", autospec=True, return_value=api_response):
        lm = LM(model="openai/gpt-5-mini", model_type="responses", temperature=1.0, max_tokens=16000)
        run = make_run(lm=lm)
        with track_usage(run) as tracker:
            result = await lm(_request(lm, prompt="test query"), run=run)
        assert result.text == "Partial async response"
        assert lm.call_log[-1].usage == {}
        assert tracker.get_total_tokens() == {}
