from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dspy.clients.lm import LM
    from dspy.clients.provider import TrainingJob
    from dspy.clients.utils_finetune import GRPOStatus, TrainDataFormat


class FinetuneJob(Protocol):
    def status(self) -> Any: ...


@runtime_checkable
class ReinforceJob(Protocol):
    def initialize(self) -> None: ...

    def get_status(self) -> GRPOStatus: ...

    def step(
        self,
        train_data: list[dict[str, Any]],
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
