from dspy.predict.chain_of_thought import ChainOfThought
from dspy.primitives import Module, Prediction
from dspy.task_spec import FieldSpec, TaskSpec, input_field, output_field

__all__ = ["SemanticF1", "CompleteAndGrounded"]


class SemanticRecallPrecisionTaskSpec(TaskSpec):
    name: str = "framework.evaluate.semantic_recall_precision"
    instructions: str = "Compare a system's response to the ground truth to compute its recall and precision. If asked to reason, enumerate key ideas in each response, and whether they are present in the other response."
    inputs: tuple[FieldSpec, ...] = (
        input_field("question", str, desc="The evaluation question."),
        input_field("ground_truth", str, desc="The reference ground-truth answer."),
        input_field("system_response", str, desc="The system response being evaluated."),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field("recall", float, desc="fraction (out of 1.0) of ground truth covered by the system response"),
        output_field("precision", float, desc="fraction (out of 1.0) of system response covered by the ground truth"),
    )


class DecompositionalSemanticRecallPrecisionTaskSpec(TaskSpec):
    name: str = "framework.evaluate.decompositional_semantic_recall_precision"
    instructions: str = "Compare a system's response to the ground truth to compute recall and precision of key ideas. You will first enumerate key ideas in each response, discuss their overlap, and then report recall and precision."
    inputs: tuple[FieldSpec, ...] = (
        input_field("question", str, desc="The evaluation question."),
        input_field("ground_truth", str, desc="The reference ground-truth answer."),
        input_field("system_response", str, desc="The system response being evaluated."),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field("ground_truth_key_ideas", str, desc="enumeration of key ideas in the ground truth"),
        output_field("system_response_key_ideas", str, desc="enumeration of key ideas in the system response"),
        output_field("discussion", str, desc="discussion of the overlap between ground truth and system response"),
        output_field("recall", float, desc="fraction (out of 1.0) of ground truth covered by the system response"),
        output_field("precision", float, desc="fraction (out of 1.0) of system response covered by the ground truth"),
    )


def f1_score(precision, recall):
    precision, recall = (max(0.0, min(1.0, precision)), max(0.0, min(1.0, recall)))
    return 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)


class SemanticF1(Module):
    def __init__(self, threshold=0.66, decompositional=False) -> None:
        super().__init__()
        self.threshold = threshold
        if decompositional:
            self.module = ChainOfThought(DecompositionalSemanticRecallPrecisionTaskSpec())
        else:
            self.module = ChainOfThought(SemanticRecallPrecisionTaskSpec())

    async def _aforward_impl(self, *, run, options=None, example, pred, trace=None):
        scores = await self.module(
            question=example.question, ground_truth=example.response, system_response=pred.response, run=run
        )
        score = f1_score(precision=scores.precision, recall=scores.recall)
        return Prediction(score=score if trace is None else score >= self.threshold)


class AnswerCompletenessTaskSpec(TaskSpec):
    name: str = "framework.evaluate.answer_completeness"
    instructions: str = "Estimate the completeness of a system's responses, against the ground truth. You will first enumerate key ideas in each response, discuss their overlap, and then report completeness."
    inputs: tuple[FieldSpec, ...] = (
        input_field("question", str, desc="The evaluation question."),
        input_field("ground_truth", str, desc="The reference ground-truth answer."),
        input_field("system_response", str, desc="The system response being evaluated."),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field("ground_truth_key_ideas", str, desc="enumeration of key ideas in the ground truth"),
        output_field("system_response_key_ideas", str, desc="enumeration of key ideas in the system response"),
        output_field("discussion", str, desc="discussion of the overlap between ground truth and system response"),
        output_field(
            "completeness", float, desc="fraction (out of 1.0) of ground truth covered by the system response"
        ),
    )


class AnswerGroundednessTaskSpec(TaskSpec):
    name: str = "framework.evaluate.answer_groundedness"
    instructions: str = "Estimate the groundedness of a system's responses, against real retrieved documents written by people. You will first enumerate whatever non-trivial or check-worthy claims are made in the system response, and then discuss the extent to which some or all of them can be deduced from the retrieved context and basic commonsense."
    inputs: tuple[FieldSpec, ...] = (
        input_field("question", str, desc="The evaluation question."),
        input_field("retrieved_context", str, desc="Retrieved documents used as grounding context."),
        input_field("system_response", str, desc="The system response being evaluated."),
    )
    outputs: tuple[FieldSpec, ...] = (
        output_field(
            "system_response_claims",
            str,
            desc="enumeration of non-trivial or check-worthy claims in the system response",
        ),
        output_field("discussion", str, desc="discussion of how supported the claims are by the retrieved context"),
        output_field(
            "groundedness", float, desc="fraction (out of 1.0) of system response supported by the retrieved context"
        ),
    )


class CompleteAndGrounded(Module):
    def __init__(self, threshold=0.66) -> None:
        super().__init__()
        self.threshold = threshold
        self.completeness_module = ChainOfThought(AnswerCompletenessTaskSpec())
        self.groundedness_module = ChainOfThought(AnswerGroundednessTaskSpec())

    async def _aforward_impl(self, *, run, options=None, example, pred, trace=None):
        completeness = await self.completeness_module(
            question=example.question, ground_truth=example.response, system_response=pred.response, run=run
        )
        groundedness = await self.groundedness_module(
            question=example.question, retrieved_context=pred.context, system_response=pred.response, run=run
        )
        score = f1_score(precision=groundedness.groundedness, recall=completeness.completeness)
        return Prediction(score=score if trace is None else score >= self.threshold)
