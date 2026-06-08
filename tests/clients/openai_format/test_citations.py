from unittest.mock import MagicMock

from dspy.clients.openai_format import extract_citations_from_choice


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
