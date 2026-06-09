from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pydantic
from pydantic import Field
from typing_extensions import override

from dspy.task_spec.json_serialize import serialize_for_json

if TYPE_CHECKING:
    from collections.abc import Iterator

    from dspy.history.turn_event import TurnEvent
    from dspy.task_spec.field_spec import FieldSpec
__all__ = ["REPLVariable", "REPLEntry", "REPLHistory"]


class REPLVariable(pydantic.BaseModel):
    name: str
    type_name: str
    desc: str = ""
    constraints: str = ""
    total_length: int
    preview: str
    model_config = pydantic.ConfigDict(frozen=True)

    @classmethod
    def from_value(
        cls, name: str, value: Any, field: FieldSpec | None = None, preview_chars: int = 1000
    ) -> REPLVariable:
        jsonable = serialize_for_json(value)
        value_str = json.dumps(jsonable, indent=2) if isinstance(jsonable, (dict, list)) else str(jsonable)
        is_truncated = len(value_str) > preview_chars
        if is_truncated:
            half = preview_chars // 2
            preview = value_str[:half] + "..." + value_str[-half:]
        else:
            preview = value_str
        desc = field.desc if field else ""
        constraints = field.constraints or "" if field else ""
        return cls(
            name=name,
            type_name=type(value).__name__,
            desc=desc,
            constraints=constraints,
            total_length=len(value_str),
            preview=preview,
        )

    def format(self) -> str:
        lines = [f"Variable: `{self.name}` (access it in your code)"]
        lines.append(f"Type: {self.type_name}")
        if self.desc:
            lines.append(f"Description: {self.desc}")
        if self.constraints:
            lines.append(f"Constraints: {self.constraints}")
        lines.append(f"Total length: {self.total_length:,} characters")
        lines.append(f"Preview:\n```\n{self.preview}\n```")
        return "\n".join(lines)

    @pydantic.model_serializer()
    def serialize_model(self) -> str:
        return self.format()


class REPLEntry(pydantic.BaseModel):
    reasoning: str = ""
    code: str
    output: str
    model_config = pydantic.ConfigDict(frozen=True)

    @staticmethod
    def format_output(output: str, max_output_chars: int = 10000) -> str:
        raw_len = len(output)
        if raw_len > max_output_chars:
            half = max_output_chars // 2
            omitted = raw_len - max_output_chars
            output = output[:half] + f"\n\n... ({omitted:,} characters omitted) ...\n\n" + output[-half:]
        return f"Output ({raw_len:,} chars):\n{output}"

    def format(self, index: int, max_output_chars: int = 10000) -> str:
        reasoning_line = f"Reasoning: {self.reasoning}\n" if self.reasoning else ""
        code_block = f"```python\n{self.code}\n```"
        return f"=== Step {index + 1} ===\n{reasoning_line}Code:\n{code_block}\n{self.format_output(self.output, max_output_chars)}"


class REPLHistory(pydantic.BaseModel):
    entries: list[REPLEntry] = Field(default_factory=list)
    max_output_chars: int = 10000
    model_config = pydantic.ConfigDict(frozen=True)

    def format(self) -> str:
        if not self.entries:
            return "You have not interacted with the REPL environment yet."
        return "\n".join(
            (entry.format(index=i, max_output_chars=self.max_output_chars) for i, entry in enumerate(self.entries))
        )

    @pydantic.model_serializer()
    def serialize_model(self) -> str:
        return self.format()

    @classmethod
    def empty(cls) -> REPLHistory:
        return cls()

    def append_turn(self, event: TurnEvent) -> REPLHistory:
        return self.append(
            reasoning=str(event.reasoning or ""),
            code=str(event.code or ""),
            output=str(event.output or ""),
        )

    def append(self, *, reasoning: str = "", code: str, output: str) -> REPLHistory:
        new_entry = REPLEntry(reasoning=reasoning, code=code, output=output)
        return REPLHistory(entries=list(self.entries) + [new_entry], max_output_chars=self.max_output_chars)

    def __len__(self) -> int:
        return len(self.entries)

    @override
    def __iter__(self) -> Iterator[REPLEntry]:  # ty:ignore[invalid-method-override]
        return iter(self.entries)

    def __bool__(self) -> bool:
        return len(self.entries) > 0
