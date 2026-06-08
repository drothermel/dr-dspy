from dspy.evaluate.metrics import normalize_text
from dspy.primitives.prediction import Completions, Prediction


def default_normalize(s):
    return normalize_text(s) or None


def majority(prediction_or_completions, normalize=default_normalize, field=None):
    assert any(isinstance(prediction_or_completions, t) for t in [Prediction, Completions, list])
    type(prediction_or_completions)
    if isinstance(prediction_or_completions, Prediction):
        completions = prediction_or_completions.completions
    else:
        completions = prediction_or_completions
    try:
        task_spec = completions.task_spec
    except Exception:
        task_spec = None
    if not field:
        field = list(task_spec.output_fields.keys())[-1] if task_spec else list(completions[0].keys())[-1]
    normalize = normalize if normalize else lambda x: x
    normalized_values = [normalize(completion[field]) for completion in completions]
    normalized_values_ = [x for x in normalized_values if x is not None]
    value_counts = {}
    for value in normalized_values_ or normalized_values:
        value_counts[value] = value_counts.get(value, 0) + 1
    majority_value = max(value_counts, key=lambda value: value_counts[value])
    completion = completions[0]
    for candidate in completions:
        if normalize(candidate[field]) == majority_value:
            completion = candidate
            break
    return Prediction.from_completions([completion], task_spec=task_spec)
