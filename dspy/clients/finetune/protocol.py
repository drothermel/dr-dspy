"""Finetune provider contracts.

Providers are namespace classes: static methods plus nested ``TrainingJob`` and
``ReinforceJob`` class attributes. ``LM`` stores a provider instance for
convenience (for example ``OpenAIProvider()``), but the protocol is satisfied
by the class itself — do not assume providers must be instantiated to work.

``is_provider_model`` on each vendor drives ``infer_finetune_provider`` in
``dspy.clients.finetune.registry`` (used by ``FinetuneService``):

- ``databricks/{endpoint}`` → ``DatabricksProvider``
- ``local:{path}`` → ``LocalProvider``
- ``openai/{model}`` or ``ft:{id}`` → ``OpenAIProvider``

See ``docs/migration/finetune.md`` for provider selection and model-id conventions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dspy.clients.finetune.provider import TrainingJob
    from dspy.clients.finetune.utils import GRPOGroup, GRPOStatus, TrainDataFormat
    from dspy.clients.lm import LM


class FinetuneJob(Protocol):
    def status(self) -> Any: ...


@runtime_checkable
class ReinforceJob(Protocol):
    def initialize(self) -> None: ...

    def get_status(self) -> GRPOStatus: ...

    def step(
        self,
        train_data: list[GRPOGroup],
        train_data_format: TrainDataFormat | str | None = None,
    ) -> None: ...

    def terminate(self) -> None: ...


class FinetuneProvider(Protocol):
    finetunable: bool
    reinforceable: bool
    TrainingJob: type[TrainingJob]
    ReinforceJob: type[ReinforceJob]

    @staticmethod
    def is_provider_model(model: str) -> bool: ...

    @staticmethod
    def launch(lm: LM, launch_kwargs: dict[str, Any] | None = None) -> None: ...

    @staticmethod
    def kill(lm: LM, launch_kwargs: dict[str, Any] | None = None) -> None: ...

    @staticmethod
    def finetune(
        job: TrainingJob,
        model: str,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | str | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> str: ...
