from unittest.mock import patch

import httpx
import pytest
from openai import AuthenticationError, NotFoundError

from dspy.integrations.finetune.openai import OpenAIProvider


def _not_found_error() -> NotFoundError:
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(404, request=request)
    return NotFoundError("not found", response=response, body=None)


def _authentication_error() -> AuthenticationError:
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(401, request=request)
    return AuthenticationError("auth failed", response=response, body=None)


def test_does_job_exist_returns_false_for_not_found():
    with patch("dspy.integrations.finetune.openai._openai") as mock_openai:
        mock_openai.return_value.fine_tuning.jobs.retrieve.side_effect = _not_found_error()
        assert OpenAIProvider.does_job_exist("job-missing") is False


def test_does_job_exist_propagates_authentication_error():
    with patch("dspy.integrations.finetune.openai._openai") as mock_openai:
        mock_openai.return_value.fine_tuning.jobs.retrieve.side_effect = _authentication_error()
        with pytest.raises(AuthenticationError):
            OpenAIProvider.does_job_exist("job-missing")


def test_does_file_exist_returns_false_for_not_found():
    with patch("dspy.integrations.finetune.openai._openai") as mock_openai:
        mock_openai.return_value.files.retrieve.side_effect = _not_found_error()
        assert OpenAIProvider.does_file_exist("file-missing") is False


def test_get_training_status_raises_value_error_for_missing_job():
    with (
        patch.object(OpenAIProvider, "does_job_exist", return_value=False),
        pytest.raises(ValueError, match="Job with ID job-missing does not exist"),
    ):
        OpenAIProvider.get_training_status("job-missing")
