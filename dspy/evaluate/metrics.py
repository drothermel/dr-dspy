import re
import string
import unicodedata
from collections import Counter

from dspy.evaluate.dpr import DPR_normalize, has_answer


def EM(prediction, answers_list):
    if not isinstance(answers_list, list):
        raise ValueError(f"`answers_list` must be a list, got {type(answers_list)}")
    return max(em_score(prediction=prediction, ground_truth=ans) for ans in answers_list)


def F1(prediction, answers_list):
    if not isinstance(answers_list, list):
        raise ValueError(f"`answers_list` must be a list, got {type(answers_list)}")
    return max(f1_score(prediction=prediction, ground_truth=ans) for ans in answers_list)


def HotPotF1(prediction, answers_list):
    if not isinstance(answers_list, list):
        raise ValueError(f"`answers_list` must be a list, got {type(answers_list)}")
    return max(hotpot_f1_score(prediction=prediction, ground_truth=ans) for ans in answers_list)


def normalize_text(s):
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub("\\b(a|an|the)\\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_score(prediction, ground_truth):
    return normalize_text(s=prediction) == normalize_text(s=ground_truth)


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_text(s=prediction).split()
    ground_truth_tokens = normalize_text(s=ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def hotpot_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_text(s=prediction)
    normalized_ground_truth = normalize_text(s=ground_truth)
    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return 0
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return 0
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def precision_score(prediction, ground_truth):
    prediction_tokens = normalize_text(s=prediction).split()
    ground_truth_tokens = normalize_text(s=ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    return 1.0 * num_same / len(prediction_tokens)


def _passage_match(passages: list[str], answers: list[str]) -> bool:
    def passage_has_answers(passage: str, answers: list[str]) -> bool:
        return has_answer(
            tokenized_answers=[DPR_normalize(normalize_text(s=ans)) for ans in answers], text=normalize_text(s=passage)
        )

    return any(passage_has_answers(passage=psg, answers=answers) for psg in passages)


def _answer_match(prediction, answers, frac=1.0):
    if frac >= 1.0:
        return EM(prediction=prediction, answers_list=answers)
    return F1(prediction=prediction, answers_list=answers) >= frac


def answer_exact_match(example, pred, trace=None, frac=1.0):
    if isinstance(example.answer, str):
        return _answer_match(prediction=pred.answer, answers=[example.answer], frac=frac)
    if isinstance(example.answer, list):
        return _answer_match(prediction=pred.answer, answers=example.answer, frac=frac)
    raise ValueError(f"Invalid answer type: {type(example.answer)}")


def answer_passage_match(example, pred, trace=None):
    if isinstance(example.answer, str):
        return _passage_match(passages=pred.context, answers=[example.answer])
    if isinstance(example.answer, list):
        return _passage_match(passages=pred.context, answers=example.answer)
    raise ValueError(f"Invalid answer type: {type(example.answer)}")
