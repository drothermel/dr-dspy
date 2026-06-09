"""Lightweight runtime configuration types."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class TransparencyMode(StrEnum):
    strict = "strict"
    warn = "warn"
    verbose = "verbose"
    off = "off"


class CallSite(BaseModel):
    model_config = ConfigDict(frozen=True)

    module: str
    phase: str = "predict"
    lm_role: str = "default"


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
    transparency: TransparencyMode = TransparencyMode.strict
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
