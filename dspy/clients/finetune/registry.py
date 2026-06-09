from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, cast

from dspy.clients.finetune.provider import DefaultFinetuneProvider

if TYPE_CHECKING:
    from dspy.clients.finetune.protocol import FinetuneProvider

_PROVIDER_CLASS_PATHS = (
    "dspy.integrations.finetune.databricks.DatabricksProvider",
    "dspy.integrations.finetune.local.LocalProvider",
    "dspy.integrations.finetune.openai.OpenAIProvider",
)

_PROVIDER_CLASSES: tuple[type[FinetuneProvider], ...] | None = None


def _provider_classes() -> tuple[type[FinetuneProvider], ...]:
    global _PROVIDER_CLASSES
    if _PROVIDER_CLASSES is None:
        classes: list[type[FinetuneProvider]] = []
        for class_path in _PROVIDER_CLASS_PATHS:
            module_name, _, class_name = class_path.rpartition(".")
            module = importlib.import_module(module_name)
            classes.append(cast("type[FinetuneProvider]", getattr(module, class_name)))
        _PROVIDER_CLASSES = tuple(classes)
    return _PROVIDER_CLASSES


def infer_finetune_provider(model: str) -> FinetuneProvider:
    for provider_cls in _provider_classes():
        if provider_cls.is_provider_model(model):
            return provider_cls()
    return DefaultFinetuneProvider()
