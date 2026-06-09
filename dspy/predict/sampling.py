from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict

from dspy.clients.base_lm import BaseLM  # noqa: TC001 — runtime SamplingAttempt fields
from dspy.errors import AdapterParseError, SamplingExhaustedError, is_retryable_lm_error
from dspy.primitives import Module, Prediction
from dspy.runtime import run_with_trace
from dspy.runtime.call_options import ModuleCallOptions  # noqa: TC001 — runtime signature typing
from dspy.runtime.run_context import RunContext  # noqa: TC001 — runtime signature typing

AttemptExecutor = Callable[["SamplingAttempt"], Awaitable[tuple[Prediction, list]]]
AfterAttemptHook = Callable[
    ["SamplingAttempt", "SamplingState", Prediction, list, float],
    Awaitable[None],
]
ShouldStopFn = Callable[["SamplingAttempt", float, "SamplingState"], bool]

_TRANSIENT_EXECUTION_ERRORS = (AdapterParseError,)


def is_transient_sampling_error(error: Exception) -> bool:
    if isinstance(error, _TRANSIENT_EXECUTION_ERRORS):
        return True
    return is_retryable_lm_error(error)


class SamplingAttempt(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    idx: int
    module: Module
    lm: BaseLM
    run: RunContext
    options: ModuleCallOptions | None
    inputs: dict[str, Any]


class SamplingState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    best_pred: Prediction | None = None
    best_trace: list[Any] | None = None
    best_reward: float = -float("inf")


class SamplingMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    best_reward: float
    attempts: int
    failures: int
    threshold: float | None = None
    threshold_met: bool = False
    best_trace: list[Any] | None = None


def set_sampling_metadata(prediction: Prediction, metadata: SamplingMetadata) -> None:
    object.__setattr__(prediction, "_sampling_metadata", metadata)


def get_sampling_metadata(prediction: Prediction) -> SamplingMetadata | None:
    metadata = getattr(prediction, "_sampling_metadata", None)
    if metadata is None:
        return None
    if not isinstance(metadata, SamplingMetadata):
        raise TypeError(f"Expected SamplingMetadata, got {type(metadata).__name__}.")
    return metadata


async def default_execute_attempt(attempt: SamplingAttempt) -> tuple[Prediction, list]:
    lm_copy = attempt.lm.copy(temperature=1.0)
    mod = attempt.module.deepcopy()
    mod.set_lm(lm_copy)
    return await run_with_trace(mod, attempt.inputs, attempt.run, options=attempt.options)


async def sample_with_reward(
    *,
    module: Module,
    num_samples: int,
    fail_count: int,
    reward_fn: Callable[[dict, Prediction], float],
    run: RunContext,
    options: ModuleCallOptions | None,
    inputs: dict[str, Any],
    threshold: float | None = None,
    should_stop: ShouldStopFn | None = None,
    execute_attempt: AttemptExecutor | None = None,
    after_attempt: AfterAttemptHook | None = None,
) -> Prediction:
    lm = module.optional_lm() or run.lm
    state = SamplingState()
    failures_seen = 0
    attempts_seen = 0
    last_exc: Exception | None = None
    execute = execute_attempt or default_execute_attempt

    for idx in range(num_samples):
        attempt = SamplingAttempt(
            idx=idx,
            module=module,
            lm=lm,
            run=run,
            options=options,
            inputs=inputs,
        )
        attempts_seen += 1
        try:
            outputs, trace = await execute(attempt)
        except Exception as err:
            if not is_transient_sampling_error(err):
                raise
            last_exc = err
            failures_seen += 1
            if failures_seen > fail_count:
                raise SamplingExhaustedError(n_attempts=num_samples) from last_exc
            continue

        reward = reward_fn(inputs, outputs)
        if reward > state.best_reward:
            state = state.model_copy(update={"best_reward": reward, "best_pred": outputs, "best_trace": trace})
        stop = (
            should_stop(attempt, reward, state)
            if should_stop is not None
            else threshold is not None and reward >= threshold
        )
        if stop:
            break
        if after_attempt is not None:
            await after_attempt(attempt, state, outputs, trace, reward)

    if state.best_pred is None:
        raise SamplingExhaustedError(n_attempts=num_samples) from last_exc
    set_sampling_metadata(
        state.best_pred,
        SamplingMetadata(
            threshold=threshold,
            threshold_met=threshold is not None and state.best_reward >= threshold,
            best_reward=state.best_reward,
            attempts=attempts_seen,
            failures=failures_seen,
            best_trace=state.best_trace,
        ),
    )
    return state.best_pred
