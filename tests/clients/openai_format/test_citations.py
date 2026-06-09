from unittest.mock import MagicMock

import pytest

from dspy.clients.openai_format import extract_citations_from_choice
from dspy.errors import LMInvalidRequestError


def test_citation_extraction_from_lm_response():
    mock_choice = MagicMock(
        message=MagicMock(
            provider_specific_fields={
                "citations": [
                    [
                        {
                            "type": "char_location",
                            "cited_text": "The sky is blue",
                            "document_index": 0,
                            "document_title": "Weather Guide",
                            "start_char_index": 10,
                            "end_char_index": 25,
                            "supported_text": "The sky is blue",
                        }
                    ]
                ]
            }
        )
    )
    citations = extract_citations_from_choice(mock_choice)
    assert citations is not None
    assert len(citations) == 1
    assert citations[0].text == "The sky is blue"
    assert citations[0].title == "Weather Guide"
    assert citations[0].metadata["document_index"] == 0
    assert citations[0].metadata["start_char_index"] == 10
    assert citations[0].metadata["end_char_index"] == 25


def test_extract_citations_returns_empty_when_citations_absent() -> None:
    mock_choice = MagicMock(message=MagicMock(provider_specific_fields={}))
    assert extract_citations_from_choice(mock_choice) == []


def test_extract_citations_raises_when_provider_specific_fields_invalid() -> None:
    mock_choice = MagicMock(message=MagicMock(provider_specific_fields="bad"))
    with pytest.raises(LMInvalidRequestError, match="provider_specific_fields"):
        extract_citations_from_choice(mock_choice)


def test_extract_citations_raises_when_citations_not_a_list() -> None:
    mock_choice = MagicMock(
        message=MagicMock(provider_specific_fields={"citations": "not-a-list"}),
    )
    with pytest.raises(LMInvalidRequestError, match="citations must be a list"):
        extract_citations_from_choice(mock_choice)
