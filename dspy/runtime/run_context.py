"""Explicit runtime configuration for DSPy execution."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol, TextIO

from pydantic import BaseModel, ConfigDict, Field

from dspy.utils.transparency import CallSite, TransparencyMode

if TYPE_CHECKING:
    from dspy.adapters.base import Adapter
    from dspy.clients.base_lm import BaseLM
    from dspy.core.types import CallRecord
    from dspy.primitives.module import Module
    from dspy.utils.callback import BaseCallback
    from dspy.utils.run_log import RunLogSession
    from dspy.utils.usage_tracker import UsageTracker


class RetrievalModule(Protocol):
    def get_objects(self, num_samples: int, fields: list[str]) -> list[dict[str, object]]: ...


class ExecutionConfig(BaseModel):
    max_concurrency: int = 8
    max_errors: int = 10
    provide_traceback: bool = False


class CallLogMode(StrEnum):
    off = "off"
    memory = "memory"
    disk = "disk"
    both = "both"


class TelemetryConfig(BaseModel):
    transparency: TransparencyMode = "strict"
    track_usage: bool = False
    call_log: CallLogMode | Literal["off", "memory", "disk", "both"] = "both"
    max_call_log_entries: int = 10000
    call_log_dir: str | None = None
    max_optimization_trace_entries: int = 10000
    warn_on_type_mismatch: bool = True


def effective_call_log_mode(telemetry: TelemetryConfig) -> CallLogMode:
    if telemetry.max_call_log_entries == 0:
        return CallLogMode.off
    mode = telemetry.call_log
    return mode if isinstance(mode, CallLogMode) else CallLogMode(mode)


def memory_call_log_enabled(telemetry: TelemetryConfig) -> bool:
    return effective_call_log_mode(telemetry) in (CallLogMode.memory, CallLogMode.both)


def disk_call_log_enabled(telemetry: TelemetryConfig) -> bool:
    return effective_call_log_mode(telemetry) in (CallLogMode.disk, CallLogMode.both)


class RunContext(BaseModel):
    """Runtime configuration passed explicitly to DSPy spine APIs via ``run=``."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    lm: BaseLM
    adapter: Adapter
    callbacks: list[BaseCallback] = Field(default_factory=list)
    optimization_trace: list[Any] = Field(default_factory=list)
    call_log: list[CallRecord] = Field(default_factory=list)
    usage_tracker: UsageTracker | None = None
    retrieval: Any | None = None
    caller_modules: list[Module] = Field(default_factory=list)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    log_session: RunLogSession | None = None
    call_site: CallSite | None = None

    @classmethod
    def create(
        cls,
        *,
        lm: BaseLM,
        adapter: Adapter,
        callbacks: list[BaseCallback] | None = None,
        optimization_trace: list[Any] | None = None,
        call_log: list[CallRecord] | None = None,
        usage_tracker: UsageTracker | None = None,
        retrieval: RetrievalModule | None = None,
        execution: ExecutionConfig | None = None,
        telemetry: TelemetryConfig | None = None,
        init_run_log: bool = True,
    ) -> RunContext:
        _ensure_run_context_model()
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
        _ensure_run_context_model()
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
        call_site = overrides.pop("call_site", self.call_site)
        lm = overrides.pop("lm", self.lm)
        adapter = overrides.pop("adapter", self.adapter)
        usage_tracker = overrides.pop("usage_tracker", self.usage_tracker)
        retrieval = overrides.pop("retrieval", self.retrieval)

        if overrides:
            unknown = ", ".join(sorted(overrides))
            raise TypeError(f"RunContext.fork() got unexpected keyword argument(s): {unknown}")

        return RunContext(
            lm=lm,
            adapter=adapter,
            callbacks=callbacks,
            optimization_trace=optimization_trace,
            call_log=call_log,
            usage_tracker=usage_tracker,
            retrieval=retrieval,
            caller_modules=[],
            execution=execution,
            telemetry=telemetry,
            log_session=log_session,
            call_site=call_site,
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
        return [entry.to_dict() for entry in self.call_log[-n:]]


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


_RUN_CONTEXT_MODEL_BUILT = False


def _ensure_run_context_model() -> None:
    global _RUN_CONTEXT_MODEL_BUILT
    if _RUN_CONTEXT_MODEL_BUILT:
        return
    from dspy.adapters.base.adapter import Adapter
    from dspy.clients.base_lm import BaseLM
    from dspy.core.types import CallRecord
    from dspy.primitives.module import Module
    from dspy.utils.callback import BaseCallback
    from dspy.utils.run_log import RunLogSession
    from dspy.utils.usage_tracker import UsageTracker

    RunContext.model_rebuild(
        _types_namespace={
            "BaseLM": BaseLM,
            "Adapter": Adapter,
            "BaseCallback": BaseCallback,
            "CallRecord": CallRecord,
            "UsageTracker": UsageTracker,
            "Module": Module,
            "RunLogSession": RunLogSession,
        }
    )
    _RUN_CONTEXT_MODEL_BUILT = True
