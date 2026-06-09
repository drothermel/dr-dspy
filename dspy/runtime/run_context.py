"""Explicit runtime configuration for DSPy execution."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TextIO

from pydantic import BaseModel, ConfigDict, Field

from dspy.utils.transparency import TransparencyMode

if TYPE_CHECKING:
    from dspy.utils.usage_tracker import UsageTracker


class ExecutionConfig(BaseModel):
    num_threads: int = 8
    max_errors: int = 10
    provide_traceback: bool = False
    allow_tool_async_sync_conversion: bool = False


class CallLogMode(StrEnum):
    off = "off"
    memory = "memory"
    disk = "disk"
    both = "both"


class TelemetryConfig(BaseModel):
    transparency: TransparencyMode = "strict"
    track_usage: bool = False
    call_log: CallLogMode = CallLogMode.both
    max_call_log_entries: int = 10000
    call_log_dir: str | None = None
    max_optimization_trace_entries: int = 10000
    warn_on_type_mismatch: bool = True


def effective_call_log_mode(telemetry: TelemetryConfig) -> CallLogMode:
    if telemetry.max_call_log_entries == 0:
        return CallLogMode.off
    return telemetry.call_log


def memory_call_log_enabled(telemetry: TelemetryConfig) -> bool:
    return effective_call_log_mode(telemetry) in (CallLogMode.memory, CallLogMode.both)


def disk_call_log_enabled(telemetry: TelemetryConfig) -> bool:
    return effective_call_log_mode(telemetry) in (CallLogMode.disk, CallLogMode.both)


class RunContext(BaseModel):
    """Runtime configuration passed explicitly to DSPy spine APIs via ``run=``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    lm: Any
    adapter: Any
    callbacks: list[Any] = Field(default_factory=list)
    optimization_trace: list[Any] = Field(default_factory=list)
    call_log: list[Any] = Field(default_factory=list)
    usage_tracker: Any | None = None
    retrieval: Any | None = None
    caller_modules: list[Any] = Field(default_factory=list)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    log_session: Any | None = None

    @classmethod
    def create(
        cls,
        *,
        lm: Any,
        adapter: Any,
        callbacks: list[Any] | None = None,
        optimization_trace: list[Any] | None = None,
        call_log: list[Any] | None = None,
        usage_tracker: UsageTracker | None = None,
        retrieval: Any | None = None,
        execution: ExecutionConfig | None = None,
        telemetry: TelemetryConfig | None = None,
        init_run_log: bool = True,
    ) -> RunContext:
        from dspy.clients.base_lm import BaseLM as BaseLMType

        if not isinstance(lm, BaseLMType):
            raise TypeError(f"RunContext requires a BaseLM instance, got {type(lm).__name__}.")
        if adapter is None:
            raise ValueError("RunContext requires an adapter.")

        run = cls(
            lm=lm,
            adapter=adapter,
            callbacks=list(callbacks or []),
            optimization_trace=list(optimization_trace) if optimization_trace is not None else [],
            call_log=list(call_log) if call_log is not None else [],
            usage_tracker=usage_tracker,
            retrieval=retrieval,
            execution=execution or ExecutionConfig(),
            telemetry=telemetry or TelemetryConfig(),
        )
        if init_run_log:
            run._init_run_session()
        return run

    def fork(self, **overrides: Any) -> RunContext:
        execution = overrides.pop("execution", self.execution)
        telemetry = overrides.pop("telemetry", self.telemetry)
        if isinstance(execution, dict):
            execution = self.execution.model_copy(update=execution)
        if isinstance(telemetry, dict):
            telemetry = self.telemetry.model_copy(update=telemetry)

        callbacks = list(overrides.pop("callbacks", self.callbacks))
        optimization_trace = list(overrides.pop("optimization_trace", self.optimization_trace))
        call_log = list(overrides.pop("call_log", self.call_log))
        log_session = overrides.pop("log_session", self.log_session)

        return RunContext(
            lm=overrides.pop("lm", self.lm),
            adapter=overrides.pop("adapter", self.adapter),
            callbacks=callbacks,
            optimization_trace=optimization_trace,
            call_log=call_log,
            usage_tracker=overrides.pop("usage_tracker", self.usage_tracker),
            retrieval=overrides.pop("retrieval", self.retrieval),
            caller_modules=[],
            execution=execution,
            telemetry=telemetry,
            log_session=log_session,
            **overrides,
        )

    def _init_run_session(self) -> None:
        from dspy.utils.run_log import create_run_log_session

        if not disk_call_log_enabled(self.telemetry):
            self.log_session = None
            return
        snapshot = {
            "lm": self.lm.model if hasattr(self.lm, "model") else repr(self.lm),
            "adapter": type(self.adapter).__name__,
            "retrieval": repr(self.retrieval) if self.retrieval is not None else None,
            "execution": self.execution.model_dump(),
            "telemetry": self.telemetry.model_dump(),
        }
        self.log_session = create_run_log_session(
            call_log_dir=self.telemetry.call_log_dir,
            settings_snapshot=snapshot,
        )

    def inspect_call_log(self, n: int = 1, file: TextIO | None = None) -> None:
        from dspy.utils.inspect_call_log import pretty_print_call_log

        pretty_print_call_log(call_log=self.call_log, n=n, file=file)

    def read_call_log(self, n: int = 10) -> list[dict[str, Any]]:
        from dspy.core.types import CallRecord

        records: list[dict[str, Any]] = []
        for entry in self.call_log[-n:]:
            if isinstance(entry, CallRecord) or hasattr(entry, "to_dict"):
                records.append(entry.to_dict())
            else:
                records.append(json.loads(json.dumps(entry, default=str)))
        return records


def resolve_run(
    *,
    run: RunContext | None,
    bound_run: RunContext | None,
) -> RunContext:
    if run is not None:
        return run
    if bound_run is not None:
        return bound_run
    raise RuntimeError(
        "No RunContext available. Pass run=RunContext.create(lm=LM(...), adapter=...) to the call, "
        "or bind run at Module/Predict construction."
    )
