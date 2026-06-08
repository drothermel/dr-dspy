import time
from datetime import datetime
from typing import Any

from typing_extensions import override

from dspy.clients.provider import Provider, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat, TrainingStatus, save_data


def _openai() -> Any:
    import openai

    return openai


class TrainingJobOpenAI(TrainingJob):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.provider_file_id = None
        self.provider_job_id = None

    @override
    def cancel(self) -> None:  # ty:ignore[invalid-method-override]
        # Cancel the provider job
        if OpenAIProvider.does_job_exist(self.provider_job_id):  # ty:ignore[invalid-argument-type]
            status = self.status()
            if OpenAIProvider.is_terminal_training_status(status):
                err_msg = "Jobs that are complete cannot be canceled."
                err_msg += f" Job with ID {self.provider_job_id} is done."
                raise Exception(err_msg)
            _openai().fine_tuning.jobs.cancel(self.provider_job_id)
            self.provider_job_id = None

        # Delete the provider file
        if self.provider_file_id is not None:
            if OpenAIProvider.does_file_exist(self.provider_file_id):
                _openai().files.delete(self.provider_file_id)
            self.provider_file_id = None

        # Call the super's cancel method after the custom cancellation logic
        super().cancel()

    @override
    def status(self) -> TrainingStatus:
        return OpenAIProvider.get_training_status(self.provider_job_id)  # ty:ignore[invalid-argument-type]


class OpenAIProvider(Provider):
    def __init__(self) -> None:
        super().__init__()
        self.finetunable = True
        self.TrainingJob = TrainingJobOpenAI

    @staticmethod
    @override
    def is_provider_model(model: str) -> bool:
        if model.startswith(("openai/", "ft:")):  # noqa: SIM103 dynamic typing/lint migration for scoped ty adoption
            # In LiteLLM, ft: uniquely identifies OpenAI fine-tuned models:
            # https://github.com/BerriAI/litellm/blob/cd893134b7974d9f21477049a373b469fff747a5/litellm/utils.py#L4495
            return True

        return False

    @staticmethod
    def _remove_provider_prefix(model: str) -> str:
        provider_prefix = "openai/"
        return model.replace(provider_prefix, "")

    @staticmethod
    @override
    def finetune(
        job: TrainingJobOpenAI,
        model: str,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> str:  # ty:ignore[invalid-method-override]
        model = OpenAIProvider._remove_provider_prefix(model)

        OpenAIProvider.validate_data_format(train_data_format)  # ty:ignore[invalid-argument-type]

        data_path = save_data(train_data)

        provider_file_id = OpenAIProvider.upload_data(data_path)
        job.provider_file_id = provider_file_id

        provider_job_id = OpenAIProvider._start_remote_training(
            train_file_id=job.provider_file_id,
            model=model,
            train_kwargs=train_kwargs,
        )
        job.provider_job_id = provider_job_id

        # TODO(feature): Could we stream OAI logs?
        OpenAIProvider.wait_for_job(job)

        return OpenAIProvider.get_trained_model(job)

    @staticmethod
    def does_job_exist(job_id: str) -> bool:
        try:
            # TODO(nit): This call may fail for other reasons. We should check
            # the error message to ensure that the job does not exist.
            _openai().fine_tuning.jobs.retrieve(job_id)
            return True
        except Exception:
            return False

    @staticmethod
    def does_file_exist(file_id: str) -> bool:
        try:
            # TODO(nit): This call may fail for other reasons. We should check
            # the error message to ensure that the file does not exist.
            _openai().files.retrieve(file_id)
            return True
        except Exception:
            return False

    @staticmethod
    def is_terminal_training_status(status: TrainingStatus) -> bool:
        return status in [
            TrainingStatus.succeeded,
            TrainingStatus.failed,
            TrainingStatus.cancelled,
        ]

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

        # Check if there is an active job
        if job_id is None:
            return TrainingStatus.not_started

        err_msg = f"Job with ID {job_id} does not exist."
        assert OpenAIProvider.does_job_exist(job_id), err_msg

        # Retrieve the provider's job and report the status
        provider_job = _openai().fine_tuning.jobs.retrieve(job_id)
        provider_status = provider_job.status
        return provider_status_to_training_status[provider_status]

    @staticmethod
    def validate_data_format(data_format: TrainDataFormat) -> None:
        supported_data_formats = [
            TrainDataFormat.CHAT,
            TrainDataFormat.COMPLETION,
        ]
        if data_format not in supported_data_formats:
            err_msg = f"OpenAI does not support the data format {data_format}."
            raise ValueError(err_msg)

    @staticmethod
    def upload_data(data_path: str) -> str:
        # Upload the data to the provider
        provider_file = _openai().files.create(
            file=open(data_path, "rb"),  # noqa: SIM115 dynamic typing/lint migration for scoped ty adoption
            purpose="fine-tune",
        )
        return provider_file.id

    @staticmethod
    def _start_remote_training(train_file_id: str, model: str, train_kwargs: dict[str, Any] | None = None) -> str:
        train_kwargs = train_kwargs or {}
        provider_job = _openai().fine_tuning.jobs.create(
            model=model,
            training_file=train_file_id,
            hyperparameters=train_kwargs,
        )
        return provider_job.id

    @staticmethod
    def wait_for_job(
        job: TrainingJobOpenAI,
        poll_frequency: int = 20,
    ) -> None:
        # Poll for the job until it is done
        done = False
        cur_event_id = None
        reported_estimated_time = False
        while not done:
            # Report estimated time if not already reported
            if not reported_estimated_time:
                remote_job = _openai().fine_tuning.jobs.retrieve(job.provider_job_id)
                timestamp = remote_job.estimated_finish
                if timestamp:
                    estimated_finish_dt = datetime.fromtimestamp(timestamp)
                    estimated_finish_dt - datetime.now()
                    reported_estimated_time = True

            # Get new events
            page = _openai().fine_tuning.jobs.list_events(fine_tuning_job_id=job.provider_job_id, limit=1)
            new_event = page.data[0] if page.data else None
            if new_event and new_event.id != cur_event_id:
                datetime.fromtimestamp(new_event.created_at)
                cur_event_id = new_event.id

            # Sleep and update the flag
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
