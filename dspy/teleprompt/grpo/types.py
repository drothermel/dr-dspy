from __future__ import annotations

from dspy.clients.finetune import GRPORolloutGroup

PredictorRolloutBatches = list[list[GRPORolloutGroup]]
