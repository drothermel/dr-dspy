"""Explicit runtime configuration for DSPy execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TextIO

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from dspy.core.types import CallRecord
from dspy.runtime.config import (
    CallSite,
    ExecutionConfig,
    TelemetryConfig,
    disk_call_log_enabled,
)
from dspy.runtime.inspect_call_log import pretty_print_call_log
from dspy.runtime.run_context_model import rebuild_run_context_model
from dspy.runtime.run_log import create_run_log_session, resolve_log_root, resolve_run_bucket

if TYPE_CHECKING:
    from dspy.adapters.base import Adapter
    from dspy.clients.base_lm import BaseLM
    from dspy.primitives import Module
    from dspy.runtime.callback import Callback
    from dspy.runtime.run_log import RunLogSession
    from dspy.runtime.usage_tracker import UsageTracker


class RetrievalModule(Protocol):
    def get_objects(self, num_samples: int, fields: list[str]) -> list[dict[str, object]]: ...


class RunContext(BaseModel):
    """Runtime configuration passed explicitly to DSPy spine APIs via ``run=``."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _model_schema_rebuilt: ClassVar[bool] = False

    lm: BaseLM
    adapter: Adapter
    callbacks: SkipValidation[list[Callback]] = Field(default_factory=list)
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
    def _ensure_model_schema_rebuilt(cls) -> None:
        if cls._model_schema_rebuilt:
            return
        rebuild_run_context_model(cls)
        cls._model_schema_rebuilt = True

    @classmethod
    def create(
        cls,
        *,
        lm: BaseLM,
        adapter: Adapter,
        callbacks: SkipValidation[list[Callback]] | None = None,
        optimization_trace: list[Any] | None = None,
        call_log: list[CallRecord] | None = None,
        usage_tracker: UsageTracker | None = None,
        retrieval: RetrievalModule | None = None,
        execution: ExecutionConfig | None = None,
        telemetry: TelemetryConfig | None = None,
        init_run_log: bool = True,
    ) -> RunContext:
        if not hasattr(lm, "model"):
            raise TypeError(f"RunContext requires a BaseLM instance, got {type(lm).__name__}.")
        if adapter is None:
            raise ValueError("RunContext requires an adapter.")

        cls._ensure_model_schema_rebuilt()

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

        explicit_log_session = "log_session" in overrides
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

        forked = RunContext(
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
        forked._ensure_log_session(explicit_log_session=explicit_log_session)
        return forked

    def _init_run_session(self) -> None:
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

    def _ensure_log_session(self, *, explicit_log_session: bool) -> None:
        if not disk_call_log_enabled(self.telemetry):
            self.log_session = None
            return
        if explicit_log_session:
            return
        if self.log_session is None:
            self._init_run_session()
            return
        expected_root = resolve_log_root(self.telemetry.call_log_dir)
        if self.log_session.run_dir.parent.parent != expected_root / resolve_run_bucket():
            self._init_run_session()

    def inspect_call_log(self, n: int = 1, file: TextIO | None = None) -> None:
        pretty_print_call_log(call_log=self.call_log, n=n, file=file)

    def read_call_log(self, n: int = 10) -> list[dict[str, Any]]:
        records = self.call_log[-n:]
        for entry in records:
            if not isinstance(entry, CallRecord):
                raise TypeError(f"call_log entry must be CallRecord, got {type(entry)!r}")
        return [entry.to_dict() for entry in records]


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
