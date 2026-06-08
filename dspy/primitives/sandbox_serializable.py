from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic_core import core_schema

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler
    from pydantic.fields import FieldInfo

    from dspy.primitives.repl_types import REPLVariable
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

    def to_repl_variable(self, name: str, field_info: FieldInfo | None = None) -> REPLVariable:
        return build_repl_variable(self, name, field_info=field_info)


def build_repl_variable(obj: SandboxSerializable, name: str, field_info: FieldInfo | None = None) -> REPLVariable:
    from dspy.primitives.repl_types import REPLVariable

    preview = obj.rlm_preview()
    var = REPLVariable.from_value(name, obj, field_info=field_info)
    setup = obj.sandbox_setup().strip()
    desc = var.desc
    if setup:
        setup_note = f"Sandbox imports available:\n{setup}"
        desc = f"{desc}\n{setup_note}" if desc else setup_note
    return var.model_copy(update={"preview": preview, "total_length": len(preview), "desc": desc})
