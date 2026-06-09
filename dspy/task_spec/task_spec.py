import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dspy.task_spec.field_spec import _UNSET, FieldRole, FieldSpec, validate_field_name
from dspy.task_spec.serialize import TASK_SPEC_VERSION, field_spec_from_dict, field_spec_to_dict


def validate_task_spec(spec: "TaskSpec") -> None:
    """Validate TaskSpec invariants: unique field names within/between roles and at least one field."""
    validate_task_spec_field_names(spec.inputs, spec.outputs)
    if not spec.inputs and not spec.outputs:
        raise ValueError("TaskSpec must have at least one input or output field.")


def validate_task_spec_field_names(
    inputs: tuple[FieldSpec, ...],
    outputs: tuple[FieldSpec, ...],
) -> None:
    def _check_within_role_duplicates(fields: tuple[FieldSpec, ...], role_label: str) -> None:
        names = [field.name for field in fields]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            quoted = ", ".join(f"'{name}'" for name in duplicates)
            raise ValueError(f"Duplicate {role_label} field name(s): {quoted}.")

    _check_within_role_duplicates(inputs, "input")
    _check_within_role_duplicates(outputs, "output")
    for field in (*inputs, *outputs):
        validate_field_name(field.name)
    cross_role = sorted({field.name for field in inputs}.intersection(field.name for field in outputs))
    if cross_role:
        raise ValueError(
            f"Input and output fields must have distinct names, but found duplicates: '{', '.join(cross_role)}'."
        )


class TaskSpec(BaseModel):
    """Task definition for predictors.

    Invariants enforced by ``validate_task_spec``:
    - Input field names are unique within inputs; output field names are unique within outputs.
    - Input and output field names do not overlap.
    - At least one input or output field is present.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    name: str
    instructions: str
    inputs: tuple[FieldSpec, ...] = Field(default_factory=tuple)
    outputs: tuple[FieldSpec, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _validate_invariants(self) -> "TaskSpec":
        validate_task_spec(self)
        return self

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
        constraints: str | None | object = _UNSET,
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
        role_fields = self._role_fields(field.role)
        return self._with_role_fields(field.role, (*role_fields, field))

    def prepend(self, field: FieldSpec) -> "TaskSpec":
        if field.name in self.fields:
            raise ValueError(f"Field already exists: {field.name}")
        role_fields = self._role_fields(field.role)
        return self._with_role_fields(field.role, (field, *role_fields))

    def insert(self, index: int, field: FieldSpec) -> "TaskSpec":
        if field.name in self.fields:
            raise ValueError(f"Field already exists: {field.name}")
        role_fields = list(self._role_fields(field.role))
        role_fields.insert(index, field)
        return self._with_role_fields(field.role, tuple(role_fields))

    def delete(self, name: str) -> "TaskSpec":
        if name not in self.fields:
            raise KeyError(f"Unknown field: {name}")
        field = self.fields[name]
        role_fields = tuple(f for f in self._role_fields(field.role) if f.name != name)
        return self._with_role_fields(field.role, role_fields)

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
        spec = cls(
            name=data["name"],
            instructions=data["instructions"],
            inputs=tuple(field_spec_from_dict(item, custom_types=custom_types) for item in data["inputs"]),
            outputs=tuple(field_spec_from_dict(item, custom_types=custom_types) for item in data["outputs"]),
        )
        validate_task_spec(spec)
        return spec

    def to_debug_string(self) -> str:
        """Human-readable debug dump; not stable for parsing or persistence."""
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

    def _role_fields(self, role: FieldRole) -> tuple[FieldSpec, ...]:
        return self.inputs if role == FieldRole.INPUT else self.outputs

    def _with_role_fields(self, role: FieldRole, fields: tuple[FieldSpec, ...]) -> "TaskSpec":
        if role == FieldRole.INPUT:
            return self.model_copy(update={"inputs": fields})
        return self.model_copy(update={"outputs": fields})

    def _replace_field(self, name: str, field: FieldSpec) -> "TaskSpec":
        role_fields = tuple(field if f.name == name else f for f in self._role_fields(field.role))
        return self._with_role_fields(field.role, role_fields)
