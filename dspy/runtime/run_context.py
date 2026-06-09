"""Explicit runtime configuration for DSPy execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from dspy.utils.transparency import TransparencyMode

if TYPE_CHECKING:
    from dspy.utils.usage_tracker import UsageTracker


class ExecutionConfig(BaseModel):
    num_threads: int = 8
    max_errors: int = 10
    provide_traceback: bool = False
    allow_tool_async_sync_conversion: bool = False


class TelemetryConfig(BaseModel):
    transparency: TransparencyMode = "strict"
    track_usage: bool = False
    disable_history: bool = False
    max_history_size: int = 10000
    max_trace_size: int = 10000
    warn_on_type_mismatch: bool = True
    run_log_enabled: bool = True
    run_log_dir: str | None = None


class RunContext(BaseModel):
    """Runtime configuration passed explicitly to DSPy spine APIs via ``run=``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    lm: Any
    adapter: Any
    callbacks: list[Any] = Field(default_factory=list)
    trace: list[Any] = Field(default_factory=list)
    usage_tracker: Any | None = None
    retrieval: Any | None = None
    caller_modules: list[Any] = Field(default_factory=list)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    @classmethod
    def create(
        cls,
        *,
        lm: Any,
        adapter: Any,
        callbacks: list[Any] | None = None,
        trace: list[Any] | None = None,
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
            trace=list(trace) if trace is not None else [],
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
        trace = list(overrides.pop("trace", self.trace))

        return RunContext(
            lm=overrides.pop("lm", self.lm),
            adapter=overrides.pop("adapter", self.adapter),
            callbacks=callbacks,
            trace=trace,
            usage_tracker=overrides.pop("usage_tracker", self.usage_tracker),
            retrieval=overrides.pop("retrieval", self.retrieval),
            caller_modules=[],
            execution=execution,
            telemetry=telemetry,
            **overrides,
        )

    def _init_run_session(self) -> None:
        from dspy.utils.run_log import init_run_session

        snapshot = {
            "lm": self.lm.model if hasattr(self.lm, "model") else repr(self.lm),
            "adapter": type(self.adapter).__name__,
            "retrieval": repr(self.retrieval) if self.retrieval is not None else None,
            "execution": self.execution.model_dump(),
            "telemetry": self.telemetry.model_dump(),
        }
        init_run_session(
            run_log_enabled=self.telemetry.run_log_enabled,
            run_log_dir=self.telemetry.run_log_dir,
            settings_snapshot=snapshot,
        )


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
