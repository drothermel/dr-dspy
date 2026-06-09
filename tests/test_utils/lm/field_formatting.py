"""Adapter-aware field formatting helpers for DummyLM."""

from __future__ import annotations

from typing import Any, cast

from dspy.task_spec import FieldBinding, output_field


def field_spec_for_value(field_name: str, value: object):
    if isinstance(value, bool):
        return output_field(field_name, bool, desc="dummy")
    if isinstance(value, int):
        return output_field(field_name, int, desc="dummy")
    if isinstance(value, float):
        return output_field(field_name, float, desc="dummy")
    if isinstance(value, list):
        if value and all(isinstance(item, str) for item in value):
            return output_field(field_name, list[str], desc="dummy")
        return output_field(field_name, list[Any], desc="dummy")
    if isinstance(value, dict):
        return output_field(field_name, dict[str, Any], desc="dummy")
    return output_field(field_name, str, desc="dummy")


def format_answer_fields(adapter, field_names_and_values: dict[str, Any]):
    fields_with_values = {
        FieldBinding(name=field_name, field=field_spec_for_value(field_name, value)): value
        for field_name, value in field_names_and_values.items()
    }
    role = adapter.capabilities.field_value_role
    if role == "assistant":
        return cast("Any", adapter).format_field_with_value(fields_with_values=fields_with_values, role="assistant")
    return adapter.format_field_with_value(fields_with_values=fields_with_values)
