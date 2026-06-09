from __future__ import annotations

from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

from typing_extensions import override

if TYPE_CHECKING:
    from threading import Thread

    from dspy.clients.protocol import ReinforceJob as ReinforceJobProtocol
    from dspy.clients.utils_finetune import GRPOStatus, TrainDataFormat


class TrainingJob(Future):
    def __init__(
        self,
        thread: Thread | None = None,
        model: str | None = None,
        train_data: list[dict[str, Any]] | None = None,
        train_data_format: TrainDataFormat | None = None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.thread = thread
        self.model = model
        self.train_data = train_data
        self.train_data_format = train_data_format
        self.train_kwargs = train_kwargs or {}
        super().__init__()

    @override
    def cancel(self) -> bool:
        return super().cancel()

    def status(self) -> Any:
        raise NotImplementedError


class _UnsupportedReinforceJob:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Reinforce is not supported by this provider.")

    def initialize(self) -> None:
        raise RuntimeError("Reinforce is not supported by this provider.")

    def get_status(self) -> GRPOStatus:
        raise RuntimeError("Reinforce is not supported by this provider.")

    def step(
        self,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | str | None = None,
    ) -> None:
        raise RuntimeError("Reinforce is not supported by this provider.")

    def terminate(self) -> None:
        raise RuntimeError("Reinforce is not supported by this provider.")


class DefaultFinetuneProvider:
    finetunable = False
    reinforceable = False
    TrainingJob: type[TrainingJob] = TrainingJob
    ReinforceJob: type[ReinforceJobProtocol] = _UnsupportedReinforceJob

    @staticmethod
    def is_provider_model(_model: str) -> bool:
        return False

    @staticmethod
    def launch(_lm: Any, _launch_kwargs: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    @staticmethod
    def kill(_lm: Any, _launch_kwargs: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    @staticmethod
    def finetune(
        _job: TrainingJob,
        _model: str,
        _train_data: list[dict[str, Any]],
        _train_data_format: TrainDataFormat | None,
        _train_kwargs: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError
