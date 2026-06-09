from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dspy.adapters.format.prompt_sections import format_fields_with_headers

if TYPE_CHECKING:
    from dspy.task_spec import FieldBinding


class HeaderFieldFormatter:
    def format_field_with_value(
        self,
        fields_with_values: dict[FieldBinding, Any],
        *,
        role_label: str | None = None,
    ) -> str:
        _ = role_label
        return format_fields_with_headers(fields_with_values)
