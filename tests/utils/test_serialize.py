from pydantic import BaseModel

from dspy.serialization.json import to_jsonable


class _ModelWithBothPaths(BaseModel):
    value: int

    def to_dict(self) -> dict[str, int]:
        return {"legacy": self.value}


class _ToDictOnly:
    def to_dict(self) -> dict[str, str]:
        return {"kind": "legacy"}


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
