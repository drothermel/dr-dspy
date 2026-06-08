from dspy.predict.chain_of_thought import ChainOfThought
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.task_spec import FieldSpec, make_task_spec

SEMANTIC_RECALL_PRECISION_TASK_SPEC = make_task_spec(
    {
        "question": FieldSpec.input("question", str),
        "ground_truth": FieldSpec.input("ground_truth", str),
        "system_response": FieldSpec.input("system_response", str),
        "recall": FieldSpec.output(
            "recall",
            float,
            desc="fraction (out of 1.0) of ground truth covered by the system response",
        ),
        "precision": FieldSpec.output(
            "precision",
            float,
            desc="fraction (out of 1.0) of system response covered by the ground truth",
        ),
    },
    instructions=(
        "Compare a system's response to the ground truth to compute its recall and precision. "
        "If asked to reason, enumerate key ideas in each response, and whether they are present in the other response."
    ),
    name="SemanticRecallPrecision",
)

DECOMPOSITIONAL_SEMANTIC_RECALL_PRECISION_TASK_SPEC = make_task_spec(
    {
        "question": FieldSpec.input("question", str),
        "ground_truth": FieldSpec.input("ground_truth", str),
        "system_response": FieldSpec.input("system_response", str),
        "ground_truth_key_ideas": FieldSpec.output(
            "ground_truth_key_ideas",
            str,
            desc="enumeration of key ideas in the ground truth",
        ),
        "system_response_key_ideas": FieldSpec.output(
            "system_response_key_ideas",
            str,
            desc="enumeration of key ideas in the system response",
        ),
        "discussion": FieldSpec.output(
            "discussion",
            str,
            desc="discussion of the overlap between ground truth and system response",
        ),
        "recall": FieldSpec.output(
            "recall",
            float,
            desc="fraction (out of 1.0) of ground truth covered by the system response",
        ),
        "precision": FieldSpec.output(
            "precision",
            float,
            desc="fraction (out of 1.0) of system response covered by the ground truth",
        ),
    },
    instructions=(
        "Compare a system's response to the ground truth to compute recall and precision of key ideas. "
        "You will first enumerate key ideas in each response, discuss their overlap, and then report recall and precision."
    ),
    name="DecompositionalSemanticRecallPrecision",
)


def f1_score(precision, recall):
    """Compute the F1 score from precision and recall, clamping both to [0, 1]."""
    precision, recall = max(0.0, min(1.0, precision)), max(0.0, min(1.0, recall))
    return 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)


class SemanticF1(Module):
    """Computes semantic F1 between a prediction and ground truth via LLM-based precision/recall.

    Args:
        threshold: Minimum F1 score to accept during optimization. Defaults to 0.66.
        decompositional: If True, uses DecompositionalSemanticRecallPrecision. Defaults to False.
    """

    def __init__(self, threshold=0.66, decompositional=False) -> None:
        self.threshold = threshold

        if decompositional:
            self.module = ChainOfThought(DECOMPOSITIONAL_SEMANTIC_RECALL_PRECISION_TASK_SPEC)
        else:
            self.module = ChainOfThought(SEMANTIC_RECALL_PRECISION_TASK_SPEC)

    async def aforward(self, example, pred, trace=None):
        scores = await self.module(
            question=example.question, ground_truth=example.response, system_response=pred.response
        )
        score = f1_score(precision=scores.precision, recall=scores.recall)

        return Prediction(score=score if trace is None else score >= self.threshold)


###########


ANSWER_COMPLETENESS_TASK_SPEC = make_task_spec(
    {
        "question": FieldSpec.input("question", str),
        "ground_truth": FieldSpec.input("ground_truth", str),
        "system_response": FieldSpec.input("system_response", str),
        "ground_truth_key_ideas": FieldSpec.output(
            "ground_truth_key_ideas",
            str,
            desc="enumeration of key ideas in the ground truth",
        ),
        "system_response_key_ideas": FieldSpec.output(
            "system_response_key_ideas",
            str,
            desc="enumeration of key ideas in the system response",
        ),
        "discussion": FieldSpec.output(
            "discussion",
            str,
            desc="discussion of the overlap between ground truth and system response",
        ),
        "completeness": FieldSpec.output(
            "completeness",
            float,
            desc="fraction (out of 1.0) of ground truth covered by the system response",
        ),
    },
    instructions=(
        "Estimate the completeness of a system's responses, against the ground truth. "
        "You will first enumerate key ideas in each response, discuss their overlap, and then report completeness."
    ),
    name="AnswerCompleteness",
)

ANSWER_GROUNDEDNESS_TASK_SPEC = make_task_spec(
    {
        "question": FieldSpec.input("question", str),
        "retrieved_context": FieldSpec.input("retrieved_context", str),
        "system_response": FieldSpec.input("system_response", str),
        "system_response_claims": FieldSpec.output(
            "system_response_claims",
            str,
            desc="enumeration of non-trivial or check-worthy claims in the system response",
        ),
        "discussion": FieldSpec.output(
            "discussion",
            str,
            desc="discussion of how supported the claims are by the retrieved context",
        ),
        "groundedness": FieldSpec.output(
            "groundedness",
            float,
            desc="fraction (out of 1.0) of system response supported by the retrieved context",
        ),
    },
    instructions=(
        "Estimate the groundedness of a system's responses, against real retrieved documents written by people. "
        "You will first enumerate whatever non-trivial or check-worthy claims are made in the system response, and then "
        "discuss the extent to which some or all of them can be deduced from the retrieved context and basic commonsense."
    ),
    name="AnswerGroundedness",
)


class CompleteAndGrounded(Module):
    """Combines answer completeness and groundedness into a single score.

    Args:
        threshold: Minimum score to accept during optimization. Defaults to 0.66.
    """

    def __init__(self, threshold=0.66) -> None:
        self.threshold = threshold
        self.completeness_module = ChainOfThought(ANSWER_COMPLETENESS_TASK_SPEC)
        self.groundedness_module = ChainOfThought(ANSWER_GROUNDEDNESS_TASK_SPEC)

    async def aforward(self, example, pred, trace=None):
        completeness = await self.completeness_module(
            question=example.question, ground_truth=example.response, system_response=pred.response
        )
        groundedness = await self.groundedness_module(
            question=example.question, retrieved_context=pred.context, system_response=pred.response
        )
        score = f1_score(precision=groundedness.groundedness, recall=completeness.completeness)

        return Prediction(score=score if trace is None else score >= self.threshold)
