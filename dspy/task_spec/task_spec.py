import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dspy.task_spec.field_spec import FieldRole, FieldSpec
from dspy.task_spec.serialize import TASK_SPEC_VERSION, field_spec_from_dict, field_spec_to_dict


class TaskSpec(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    name: str
    instructions: str
    inputs: tuple[FieldSpec, ...] = Field(default_factory=tuple)
    outputs: tuple[FieldSpec, ...] = Field(default_factory=tuple)

    @property
    def input_fields(self) -> dict[str, FieldSpec]:
        return {field.name: field for field in self.inputs}

    @property
    def output_fields(self) -> dict[str, FieldSpec]:
        return {field.name: field for field in self.outputs}

    @property
    def fields(self) -> dict[str, FieldSpec]:
        return {**self.input_fields, **self.output_fields}

    @property
    def spec_string(self) -> str:
        input_names = ", ".join(field.name for field in self.inputs)
        output_names = ", ".join(field.name for field in self.outputs)
        return f"{input_names} -> {output_names}"

    def with_instructions(self, instructions: str) -> "TaskSpec":
        return self.model_copy(update={"instructions": instructions})

    def with_updated_field(
        self,
        name: str,
        *,
        desc: str | None = None,
        prefix: str | None = None,
        type_: Any = None,
        constraints: str | None = None,
    ) -> "TaskSpec":
        if name not in self.fields:
            raise KeyError(f"Unknown field: {name}")
        field = self.fields[name].with_updates(
            desc=desc,
            prefix=prefix,
            type_=type_,
            constraints=constraints,
        )
        return self._replace_field(name, field)

    def append(self, field: FieldSpec) -> "TaskSpec":
        if field.name in self.fields:
            raise ValueError(f"Field already exists: {field.name}")
        if field.role == FieldRole.INPUT:
            return self.model_copy(update={"inputs": (*self.inputs, field)})
        return self.model_copy(update={"outputs": (*self.outputs, field)})

    def prepend(self, field: FieldSpec) -> "TaskSpec":
        if field.name in self.fields:
            raise ValueError(f"Field already exists: {field.name}")
        if field.role == FieldRole.INPUT:
            return self.model_copy(update={"inputs": (field, *self.inputs)})
        return self.model_copy(update={"outputs": (field, *self.outputs)})

    def insert(self, index: int, field: FieldSpec) -> "TaskSpec":
        if field.name in self.fields:
            raise ValueError(f"Field already exists: {field.name}")
        if field.role == FieldRole.INPUT:
            fields = list(self.inputs)
            fields.insert(index, field)
            return self.model_copy(update={"inputs": tuple(fields)})
        fields = list(self.outputs)
        fields.insert(index, field)
        return self.model_copy(update={"outputs": tuple(fields)})

    def delete(self, name: str) -> "TaskSpec":
        if name not in self.fields:
            raise KeyError(f"Unknown field: {name}")
        field = self.fields[name]
        if field.role == FieldRole.INPUT:
            return self.model_copy(update={"inputs": tuple(f for f in self.inputs if f.name != name)})
        return self.model_copy(update={"outputs": tuple(f for f in self.outputs if f.name != name)})

    def fingerprint(self) -> str:
        payload = {
            "name": self.name,
            "instructions": self.instructions,
            "inputs": [field_spec_to_dict(field) for field in self.inputs],
            "outputs": [field_spec_to_dict(field) for field in self.outputs],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "task_spec_version": TASK_SPEC_VERSION,
            "name": self.name,
            "instructions": self.instructions,
            "inputs": [field_spec_to_dict(field) for field in self.inputs],
            "outputs": [field_spec_to_dict(field) for field in self.outputs],
        }

    @classmethod
    def from_dict(cls, data: dict, *, custom_types: dict[str, type] | None = None) -> "TaskSpec":
        version = data.get("task_spec_version")
        if version != TASK_SPEC_VERSION:
            raise ValueError(
                f"Unsupported task_spec_version: {version!r}. Expected {TASK_SPEC_VERSION}. Recompile or recreate the program with the current DSPy version."
            )
        return cls(
            name=data["name"],
            instructions=data["instructions"],
            inputs=tuple(field_spec_from_dict(item, custom_types=custom_types) for item in data["inputs"]),
            outputs=tuple(field_spec_from_dict(item, custom_types=custom_types) for item in data["outputs"]),
        )

    def to_declaration(self) -> str:
        field_lines = [
            f"  {field.role.value} {field.name}: {field.type_!r} desc={field.desc!r} prefix={field.prefix!r}"
            for field in (*self.inputs, *self.outputs)
        ]
        lines = [
            f"TaskSpec(name={self.name!r}, instructions={self.instructions!r})",
            f"  spec: {self.spec_string}",
            *field_lines,
        ]
        return "\n".join(lines)

    def _replace_field(self, name: str, field: FieldSpec) -> "TaskSpec":
        if field.role == FieldRole.INPUT:
            inputs = tuple(field if f.name == name else f for f in self.inputs)
            return self.model_copy(update={"inputs": inputs})
        outputs = tuple(field if f.name == name else f for f in self.outputs)
        return self.model_copy(update={"outputs": outputs})
