from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from dspy.clients.finetune.provider import TrainingJob, _UnsupportedReinforceJob
from dspy.clients.finetune.utils import TrainDataFormat, TrainingStatus, save_data

if TYPE_CHECKING:
    from dspy.clients.finetune.protocol import ReinforceJob as ReinforceJobProtocol

logger = logging.getLogger(__name__)


def _openai() -> Any:
    import openai

    return openai


class TrainingJobOpenAI(TrainingJob):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.provider_file_id = None
        self.provider_job_id = None

    @override
    def cancel(self) -> bool:
        if self.provider_job_id is not None and OpenAIProvider.does_job_exist(self.provider_job_id):
            status = self.status()
            if OpenAIProvider.is_terminal_training_status(status):
                err_msg = "Jobs that are complete cannot be canceled."
                err_msg += f" Job with ID {self.provider_job_id} is done."
                raise Exception(err_msg)
            _openai().fine_tuning.jobs.cancel(self.provider_job_id)
            self.provider_job_id = None
        if self.provider_file_id is not None:
            if OpenAIProvider.does_file_exist(self.provider_file_id):
                _openai().files.delete(self.provider_file_id)
            self.provider_file_id = None
        return super().cancel()

    @override
    def status(self) -> TrainingStatus:
        if self.provider_job_id is None:
            return TrainingStatus.not_started
        return OpenAIProvider.get_training_status(self.provider_job_id)


class OpenAIProvider:
    finetunable = True
    reinforceable = False
    TrainingJob: type[TrainingJob] = TrainingJobOpenAI
    ReinforceJob: type[ReinforceJobProtocol] = _UnsupportedReinforceJob

    @staticmethod
    def is_provider_model(_model: str) -> bool:
        return bool(_model.startswith(("openai/", "ft:")))

    @staticmethod
    def launch(_lm: Any, _launch_kwargs: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    @staticmethod
    def kill(_lm: Any, _launch_kwargs: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    @staticmethod
    def _remove_provider_prefix(model: str) -> str:
        provider_prefix = "openai/"
        return model.replace(provider_prefix, "")

    @staticmethod
    def finetune(
        job: TrainingJob,
        model: str,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | str | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> str:
        model = OpenAIProvider._remove_provider_prefix(model)
        if not isinstance(train_data_format, TrainDataFormat):
            raise TypeError(f"Expected TrainDataFormat, got {type(train_data_format).__name__}.")
        OpenAIProvider.validate_data_format(train_data_format)
        data_path = save_data(train_data)
        provider_file_id = OpenAIProvider.upload_data(data_path)
        job.provider_file_id = provider_file_id
        provider_job_id = OpenAIProvider._start_remote_training(
            train_file_id=job.provider_file_id, model=model, train_kwargs=train_kwargs
        )
        job.provider_job_id = provider_job_id
        OpenAIProvider.wait_for_job(job)
        return OpenAIProvider.get_trained_model(job)

    @staticmethod
    def does_job_exist(job_id: str) -> bool:
        try:
            _openai().fine_tuning.jobs.retrieve(job_id)
            return True
        except Exception:
            return False

    @staticmethod
    def does_file_exist(file_id: str) -> bool:
        try:
            _openai().files.retrieve(file_id)
            return True
        except Exception:
            return False

    @staticmethod
    def is_terminal_training_status(status: TrainingStatus) -> bool:
        return status in [TrainingStatus.succeeded, TrainingStatus.failed, TrainingStatus.cancelled]

    @staticmethod
    def get_training_status(job_id: str) -> TrainingStatus:
        provider_status_to_training_status = {
            "validating_files": TrainingStatus.pending,
            "queued": TrainingStatus.pending,
            "running": TrainingStatus.running,
            "succeeded": TrainingStatus.succeeded,
            "failed": TrainingStatus.failed,
            "cancelled": TrainingStatus.cancelled,
        }
        if job_id is None:
            return TrainingStatus.not_started
        err_msg = f"Job with ID {job_id} does not exist."
        assert OpenAIProvider.does_job_exist(job_id), err_msg
        provider_job = _openai().fine_tuning.jobs.retrieve(job_id)
        provider_status = provider_job.status
        return provider_status_to_training_status.get(provider_status, TrainingStatus.pending)

    @staticmethod
    def validate_data_format(data_format: TrainDataFormat) -> None:
        supported_data_formats = [TrainDataFormat.CHAT, TrainDataFormat.COMPLETION]
        if data_format not in supported_data_formats:
            err_msg = f"OpenAI does not support the data format {data_format}."
            raise ValueError(err_msg)

    @staticmethod
    def upload_data(data_path: str) -> str:
        with open(data_path, "rb") as data_file:
            provider_file = _openai().files.create(file=data_file, purpose="fine-tune")
        return provider_file.id

    @staticmethod
    def _start_remote_training(train_file_id: str, model: str, train_kwargs: dict[str, Any] | None = None) -> str:
        train_kwargs = train_kwargs or {}
        provider_job = _openai().fine_tuning.jobs.create(
            model=model, training_file=train_file_id, hyperparameters=train_kwargs
        )
        return provider_job.id

    @staticmethod
    def wait_for_job(job: TrainingJobOpenAI, poll_frequency: int = 20) -> None:
        done = False
        cur_event_id = None
        reported_estimated_time = False
        while not done:
            if not reported_estimated_time:
                remote_job = _openai().fine_tuning.jobs.retrieve(job.provider_job_id)
                timestamp = remote_job.estimated_finish
                if timestamp:
                    estimated_finish_dt = datetime.fromtimestamp(timestamp)
                    remaining_seconds = (estimated_finish_dt - datetime.now()).total_seconds()
                    logger.debug(
                        "OpenAI fine-tune job %s estimated finish in %.0f seconds",
                        job.provider_job_id,
                        remaining_seconds,
                    )
                    reported_estimated_time = True
            page = _openai().fine_tuning.jobs.list_events(fine_tuning_job_id=job.provider_job_id, limit=1)
            new_event = page.data[0] if page.data else None
            if new_event and new_event.id != cur_event_id:
                logger.debug(
                    "OpenAI fine-tune job %s event %s at %s: %s",
                    job.provider_job_id,
                    new_event.id,
                    datetime.fromtimestamp(new_event.created_at).isoformat(),
                    getattr(new_event, "message", new_event),
                )
                cur_event_id = new_event.id
            time.sleep(poll_frequency)
            done = OpenAIProvider.is_terminal_training_status(job.status())

    @staticmethod
    def get_trained_model(job):
        status = job.status()
        if status != TrainingStatus.succeeded:
            err_msg = f"Job status is {status}."
            err_msg += f" Must be {TrainingStatus.succeeded} to retrieve model."
            raise Exception(err_msg)
        provider_job = _openai().fine_tuning.jobs.retrieve(job.provider_job_id)
        return provider_job.fine_tuned_model
