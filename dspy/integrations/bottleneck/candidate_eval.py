from __future__ import annotations

import time
from importlib import import_module
from typing import Any

from dr_queues import JobEnvelope
from dr_queues.amqp.publish import publish_job
from dr_queues.amqp.session import broker_session
from dr_queues.amqp.topology import declare_durable_queue, declare_durable_queues
from pydantic import BaseModel, ConfigDict, PrivateAttr

CANDIDATE_EVAL_STAGE = "candidate_eval"
DEFAULT_REQUEST_QUEUE = "bottleneck.candidate_eval.requests"
REQUEST_PAYLOAD_KEY = "candidate_eval_request"
DEFAULT_POLL_INTERVAL_SECONDS = 0.5


class BottleneckQueueEvaluator(BaseModel):
    """Queue client for dr-bottleneck candidate evaluation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_queue: str = DEFAULT_REQUEST_QUEUE
    result_queue: str
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
    _pending_results: list[Any] = PrivateAttr(default_factory=list)

    def submit(self, request: Any) -> str:
        CandidateEvalRequest, _CandidateEvalResult = _runtime_types()
        parsed = CandidateEvalRequest.model_validate(request)
        if parsed.result_queue != self.result_queue:
            parsed = parsed.model_copy(update={"result_queue": self.result_queue})
        job = JobEnvelope(
            run_id=parsed.optimizer_run_id,
            lane=CANDIDATE_EVAL_STAGE,
            repeat=0,
            step_index=0,
            pipeline_id=CANDIDATE_EVAL_STAGE,
            payload={REQUEST_PAYLOAD_KEY: parsed.model_dump(mode="json")},
        )
        with broker_session() as broker:
            declare_durable_queues(
                broker.channel,
                [self.request_queue, self.result_queue],
            )
            publish_job(
                broker.channel,
                self.request_queue,
                job.to_json(),
            )
        return job.job_id

    def submit_many(self, requests: list[Any]) -> list[str]:
        return [self.submit(request) for request in requests]

    def collect_result(
        self,
        *,
        optimizer_run_id: str,
        candidate_id: str,
        timeout_seconds: float,
    ) -> Any:
        pending = self._pop_pending_result(
            optimizer_run_id=optimizer_run_id,
            candidate_id=candidate_id,
        )
        if pending is not None:
            return pending

        _CandidateEvalRequest, CandidateEvalResult = _runtime_types()
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
                result = CandidateEvalResult.model_validate(job.step_records[CANDIDATE_EVAL_STAGE])
                if result.optimizer_run_id == optimizer_run_id and result.candidate_id == candidate_id:
                    broker.channel.basic_ack(
                        delivery_tag=delivery_tag,
                    )
                    return result
                broker.channel.basic_ack(
                    delivery_tag=delivery_tag,
                )
                self._pending_results.append(result)
                time.sleep(self.poll_interval_seconds)
        msg = f"Timed out waiting for candidate_id={candidate_id!r} optimizer_run_id={optimizer_run_id!r}."
        raise TimeoutError(msg)

    def collect_next(
        self,
        *,
        timeout_seconds: float,
        optimizer_run_id: str | None = None,
    ) -> Any:
        pending = self._pop_pending_next(optimizer_run_id=optimizer_run_id)
        if pending is not None:
            return pending

        _CandidateEvalRequest, CandidateEvalResult = _runtime_types()
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
                result = CandidateEvalResult.model_validate(job.step_records[CANDIDATE_EVAL_STAGE])
                if optimizer_run_id is None or result.optimizer_run_id == optimizer_run_id:
                    broker.channel.basic_ack(
                        delivery_tag=delivery_tag,
                    )
                    return result
                broker.channel.basic_ack(delivery_tag=delivery_tag)
                self._pending_results.append(result)
        msg = "Timed out waiting for a candidate eval result."
        raise TimeoutError(msg)

    def _pop_pending_result(
        self,
        *,
        optimizer_run_id: str,
        candidate_id: str,
    ) -> Any | None:
        for index, result in enumerate(self._pending_results):
            if result.optimizer_run_id == optimizer_run_id and result.candidate_id == candidate_id:
                return self._pending_results.pop(index)
        return None

    def _pop_pending_next(self, *, optimizer_run_id: str | None) -> Any | None:
        if optimizer_run_id is None:
            if self._pending_results:
                return self._pending_results.pop(0)
            return None
        for index, result in enumerate(self._pending_results):
            if result.optimizer_run_id == optimizer_run_id:
                return self._pending_results.pop(index)
        return None


def default_result_queue(optimizer_run_id: str) -> str:
    return f"bottleneck.candidate_eval.{optimizer_run_id}.results"


def candidate_score(result: Any, *, metric_target: str) -> float:
    metrics = result.aggregate_metrics
    if result.status == "failed":
        return 0.0
    if metric_target == "ast_parse_rate":
        return float(metrics.parse_rate)
    if metric_target == "test_pass_rate":
        return float(metrics.all_tests_passed_rate)
    if metric_target == "correctness_then_compression":
        correctness = float(metrics.all_tests_passed_rate)
        compression_penalty = metrics.mean_compressed_decoder_input_bytes / 1_000_000
        return correctness - compression_penalty
    msg = f"Unknown metric target: {metric_target!r}"
    raise ValueError(msg)


def _runtime_types() -> tuple[Any, Any]:
    candidate_eval = import_module("dr_bottleneck.candidate_eval")
    return (
        candidate_eval.CandidateEvalRequest,
        candidate_eval.CandidateEvalResult,
    )
