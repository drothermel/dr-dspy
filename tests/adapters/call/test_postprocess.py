from dspy.adapters.call.postprocess import enrich_parsed_value_from_lm_output
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types.reasoning import Reasoning
from dspy.core.types import LMOutput, LMTextPart, LMThinkingPart
from dspy.task_spec import input_field, make_task_spec, output_field


def test_enrich_parsed_value_setdefaults_missing_output_fields():
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "answer": output_field("answer", desc="The answer."),
            "reasoning": output_field("reasoning", type_=Reasoning, desc="The reasoning."),
        },
        instructions="Answer the question.",
    )
    adapter = ChatAdapter()
    output = LMOutput(parts=[LMTextPart(text="Paris")])
    value = enrich_parsed_value_from_lm_output(
        adapter,
        value={"answer": "Paris"},
        output=output,
        original_task_spec=task_spec,
    )
    assert value["answer"] == "Paris"
    assert value["reasoning"] is None


def test_enrich_parsed_value_merges_native_reasoning_from_lm_output():
    task_spec = make_task_spec(
        {
            "question": input_field("question", desc="The question."),
            "reasoning": output_field("reasoning", type_=Reasoning, desc="The reasoning."),
            "answer": output_field("answer", desc="The answer."),
        },
        instructions="Answer the question.",
    )
    adapter = ChatAdapter()
    output = LMOutput(
        parts=[
            LMTextPart(text="Paris"),
            LMThinkingPart(text="Native provider reasoning"),
        ]
    )
    value = enrich_parsed_value_from_lm_output(
        adapter,
        value={"reasoning": Reasoning(content="wrong from text"), "answer": "Paris"},
        output=output,
        original_task_spec=task_spec,
    )
    assert value["reasoning"] == Reasoning(content="Native provider reasoning")


def test_enrich_parsed_value_preserves_logprobs_when_empty_dict():
    task_spec = make_task_spec(
        {"question": input_field("question", desc="The question."), "answer": output_field("answer", desc="The answer.")},
        instructions="Answer the question.",
    )
    adapter = ChatAdapter()
    output = LMOutput(parts=[LMTextPart(text="Paris")], logprobs={})
    value = enrich_parsed_value_from_lm_output(
        adapter,
        value={"answer": "Paris"},
        output=output,
        original_task_spec=task_spec,
    )
    assert value["logprobs"] == {}
