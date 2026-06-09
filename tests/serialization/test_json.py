import enum
from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import BaseModel

from dspy.serialization.json import to_jsonable


class _ModelWithBothPaths(BaseModel):
    value: int

    def to_dict(self) -> dict[str, int]:
        return {"legacy": self.value}


class _ToDictOnly:
    def to_dict(self) -> dict[str, str]:
        return {"kind": "legacy"}


class _NestedToDict:
    def to_dict(self) -> dict[str, object]:
        return {"nested": object()}


class _Color(enum.Enum):
    RED = "red"


def test_to_jsonable_prefers_base_model_dump_over_to_dict():
    result = to_jsonable(_ModelWithBothPaths(value=7))
    assert result == {"value": 7}


def test_to_jsonable_uses_to_dict_when_no_base_model():
    result = to_jsonable(_ToDictOnly())
    assert result == {"kind": "legacy"}


def test_to_jsonable_handles_circular_references():
    data: dict[str, object] = {}
    data["self"] = data
    result = to_jsonable(data)
    assert result == {"self": "<circular>"}


def test_to_jsonable_converts_tuples_to_lists():
    assert to_jsonable((1, 2)) == [1, 2]


def test_to_jsonable_serializes_enum_in_dict():
    assert to_jsonable({"color": _Color.RED}) == {"color": "red"}


def test_to_jsonable_recurses_into_to_dict_result():
    result = to_jsonable(_NestedToDict())
    assert result == {"nested": result["nested"]}
    assert isinstance(result["nested"], str)


def test_to_jsonable_strict_rejects_unsupported_types():
    with pytest.raises(TypeError, match="not JSON-serializable"):
        to_jsonable(object(), strict=True)


def test_to_jsonable_converts_set_to_list():
    assert to_jsonable({3, 1, 2}) == [1, 2, 3]


def test_to_jsonable_handles_type_adapter_friendly_values():
    value = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert to_jsonable(value) in {"2024-01-02T03:04:05+00:00", "2024-01-02T03:04:05Z"}
    uuid_value = UUID("12345678-1234-5678-1234-567812345678")
    assert to_jsonable(uuid_value) == "12345678-1234-5678-1234-567812345678"
