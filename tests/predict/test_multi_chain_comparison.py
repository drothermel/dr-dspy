import asyncio

from dspy.predict.multi_chain_comparison import MultiChainComparison
from dspy.primitives.prediction import Prediction
from dspy.task_spec import FieldSpec, make_task_spec
from dspy.utils.dummies import DummyLM

BasicQA = make_task_spec(
    {"question": FieldSpec.input("question"), "answer": FieldSpec.output("answer", desc="often between 1 and 5 words")},
    instructions="Answer questions with short factoid answers.",
    name="BasicQA",
)
completions = [
    Prediction.from_record(
        {"rationale": "I recall that during clear days, the sky often appears this color.", "answer": "blue"}
    ),
    Prediction.from_record(
        {
            "rationale": "Based on common knowledge, I believe the sky is typically seen as this color.",
            "answer": "green",
        }
    ),
    Prediction.from_record(
        {
            "rationale": "From images and depictions in media, the sky is frequently represented with this hue.",
            "answer": "blue",
        }
    ),
]


def test_basic_example(make_run):
    compare_answers = MultiChainComparison(BasicQA)
    question = "What is the color of the sky?"
    lm = DummyLM([{"rationale": "my rationale", "answer": "blue"}])
    run = make_run(lm=lm)
    final_pred = asyncio.run(compare_answers(completions=completions, question=question, run=run))
    assert final_pred.rationale == "my rationale"
    assert final_pred.answer == "blue"
