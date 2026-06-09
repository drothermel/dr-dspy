from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from dspy.adapters.format.prompt_sections import format_fields_with_headers
from dspy.serialization.json import to_jsonable

if TYPE_CHECKING:
    from dspy.task_spec import FieldBinding

USER_ROLE_LABEL = "user"
ASSISTANT_ROLE_LABEL = "assistant"


class JsonFieldFormatter:
    def format_field_with_value(
        self,
        fields_with_values: dict[FieldBinding, Any],
        *,
        role_label: str | None = None,
    ) -> str:
        if role_label == USER_ROLE_LABEL or role_label is None:
            return format_fields_with_headers(fields_with_values)
        d = {binding.name: value for binding, value in fields_with_values.items()}
        return json.dumps(to_jsonable(d), indent=2, ensure_ascii=False)
