from __future__ import annotations

import time
from typing import Any

from dr_queues import JobEnvelope
from dr_queues.amqp.publish import publish_job
from dr_queues.amqp.session import broker_session
from dr_queues.amqp.topology import declare_durable_queue, declare_durable_queues
from pydantic import BaseModel, ConfigDict, PrivateAttr

DEFAULT_WORKFLOW_QUEUE = "dr-bottleneck.workflow.jobs"
DEFAULT_RESULT_QUEUE_PREFIX = "dr-bottleneck.workflow"
DEFAULT_POLL_INTERVAL_SECONDS = 0.5


class BottleneckWorkflowClient(BaseModel):
    """Queue client for concrete dr-bottleneck workflow jobs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_queue: str = DEFAULT_WORKFLOW_QUEUE
    result_queue: str
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
    _pending_jobs: list[JobEnvelope] = PrivateAttr(default_factory=list)

    def submit(self, payload: Any) -> str:
        workflow_id = str(payload.workflow_id)
        metadata = dict(payload.metadata)
        job = JobEnvelope(
            run_id=str(metadata["optimizer_run_id"]),
            lane=str(metadata.get("candidate_id", "default")),
            repeat=int(metadata.get("seed_index", 0)),
            step_index=0,
            pipeline_id=workflow_id,
            payload=payload.model_dump(mode="json"),
        )
        queues = [
            self.input_queue,
            *[step.input_queue for step in payload.steps],
            *[step.output_queue for step in payload.steps if step.output_queue is not None],
            self.result_queue,
        ]
        with broker_session() as broker:
            declare_durable_queues(broker.channel, sorted(set(queues)))
            publish_job(
                broker.channel,
                self.input_queue,
                job.to_json(),
            )
        return job.job_id

    def submit_many(self, payloads: list[Any]) -> list[str]:
        return [self.submit(payload) for payload in payloads]

    def collect_next(
        self,
        *,
        timeout_seconds: float,
        optimizer_run_id: str | None = None,
    ) -> JobEnvelope:
        pending = self._pop_pending_next(optimizer_run_id=optimizer_run_id)
        if pending is not None:
            return pending

        deadline = time.monotonic() + timeout_seconds
        with broker_session() as broker:
            declare_durable_queue(broker.channel, self.result_queue)
            while time.monotonic() < deadline:
                method, _properties, body = broker.channel.basic_get(
                    queue=self.result_queue,
                    auto_ack=False,
                )
                if method is None:
                    time.sleep(self.poll_interval_seconds)
                    continue
                assert body is not None
                assert method.delivery_tag is not None
                delivery_tag = int(method.delivery_tag)
                job = JobEnvelope.from_json(body)
                if optimizer_run_id is None or job.run_id == optimizer_run_id:
                    broker.channel.basic_ack(
                        delivery_tag=delivery_tag,
                    )
                    return job
                broker.channel.basic_ack(delivery_tag=delivery_tag)
                self._pending_jobs.append(job)
        msg = "Timed out waiting for a workflow result."
        raise TimeoutError(msg)

    def _pop_pending_next(
        self,
        *,
        optimizer_run_id: str | None,
    ) -> JobEnvelope | None:
        if optimizer_run_id is None:
            if self._pending_jobs:
                return self._pending_jobs.pop(0)
            return None
        for index, job in enumerate(self._pending_jobs):
            if job.run_id == optimizer_run_id:
                return self._pending_jobs.pop(index)
        return None


def default_workflow_result_queue(optimizer_run_id: str) -> str:
    return f"{DEFAULT_RESULT_QUEUE_PREFIX}.{optimizer_run_id}.results"
