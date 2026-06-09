from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic_core import core_schema

from dspy.primitives.repl_types import REPLVariable

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler

    from dspy.task_spec.field_spec import FieldSpec

__all__ = ["SandboxSerializable", "build_repl_variable"]


class SandboxSerializable(ABC):
    @abstractmethod
    def sandbox_setup(self) -> str: ...

    @abstractmethod
    def to_sandbox(self) -> bytes: ...

    @abstractmethod
    def sandbox_assignment(self, var_name: str, data_expr: str) -> str: ...

    @abstractmethod
    def rlm_preview(self, max_chars: int = 500) -> str: ...

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(
            lambda v: v, serialization=core_schema.plain_serializer_function_ser_schema(lambda v: str(v))
        )

    def to_repl_variable(self, name: str, field: FieldSpec | None = None) -> REPLVariable:
        return build_repl_variable(self, name, field=field)


def build_repl_variable(obj: SandboxSerializable, name: str, field: FieldSpec | None = None) -> REPLVariable:
    preview = obj.rlm_preview()
    var = REPLVariable.from_value(name, obj, field=field)
    setup = obj.sandbox_setup().strip()
    desc = var.desc
    if setup:
        setup_note = f"Sandbox imports available:\n{setup}"
        desc = f"{desc}\n{setup_note}" if desc else setup_note
    return var.model_copy(update={"preview": preview, "total_length": len(preview), "desc": desc})
