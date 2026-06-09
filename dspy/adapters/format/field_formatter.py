from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from dspy.task_spec import FieldBinding


class FieldFormatter(Protocol):
    def format_field_with_value(
        self,
        fields_with_values: dict[FieldBinding, Any],
        *,
        role_label: str | None = None,
    ) -> str: ...
