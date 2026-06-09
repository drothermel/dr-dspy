from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dspy.adapters.prompt_format import format_field_value

if TYPE_CHECKING:
    from dspy.task_spec import FieldBinding


class XmlFieldFormatter:
    def format_field_with_value(
        self,
        fields_with_values: dict[FieldBinding, Any],
        *,
        role_label: str | None = None,
    ) -> str:
        _ = role_label
        output = []
        for binding, field_value in fields_with_values.items():
            formatted = format_field_value(field=binding.field, value=field_value)
            output.append(f"<{binding.name}>\n{formatted}\n</{binding.name}>")
        return "\n\n".join(output).strip()
