"""Optimizer metric contract for teleprompters.

Standard optimizers accept a sync or async callable ``(example, prediction, trace)``
returning bool, a numeric score in ``[0, 1]``, or a ``Prediction`` with ``score``.
``Module`` metrics are also supported; a non-``None`` ``trace`` enables threshold mode.

``invoke_metric`` normalizes all return shapes to a 0-1 float. GEPA uses the
separate five-argument ``GEPAFeedbackMetric`` protocol in ``dspy.teleprompt.gepa``.
"""

from dspy.evaluate.metric_contract import (
    AsyncOptimizerMetric,
    MetricScore,
    OptimizerMetric,
    SyncOptimizerMetric,
)

__all__ = [
    "AsyncOptimizerMetric",
    "MetricScore",
    "OptimizerMetric",
    "SyncOptimizerMetric",
]
