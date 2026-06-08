from dspy.core.types import LMRequest, LMResponse, LMTextPart, User


def test_lm_request_from_call_accepts_prior_lm_response():
    prior = LMResponse.from_text("First turn.")
    request = LMRequest.from_call(
        model="openai/gpt-4o-mini",
        items=(prior, User("Follow-up question")),
    )
    assert len(request.messages) == 2
    assert request.messages[0].role == "assistant"
    assert request.messages[0].parts == [LMTextPart(text="First turn.")]
    assert request.messages[1].role == "user"
    assert request.messages[1].text == "Follow-up question"
