import base64
import mimetypes
from pathlib import Path
from typing import Any

import pydantic
from typing_extensions import override

from dspy.adapters.types.field_type import FieldTypeMixin


class File(FieldTypeMixin):
    file_data: str | None = None
    file_id: str | None = None
    filename: str | None = None
    model_config = pydantic.ConfigDict(frozen=True, str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, values: object) -> object:
        if isinstance(values, cls):
            return {"file_data": values.file_data, "file_id": values.file_id, "filename": values.filename}
        if isinstance(values, dict):
            if "file_data" in values or "file_id" in values or "filename" in values:
                return values
            raise ValueError(
                "Value of `dspy.adapters.types.file.File` must contain at least one of: file_data, file_id, or filename"
            )
        return encode_file_to_dict(values)

    @override
    def format(self) -> list[dict[str, Any]]:
        file_dict = {}
        if self.file_data:
            file_dict["file_data"] = self.file_data
        if self.file_id:
            file_dict["file_id"] = self.file_id
        if self.filename:
            file_dict["filename"] = self.filename
        return [{"type": "file", "file": file_dict}]

    @override
    def __str__(self) -> str:
        return str(self.serialize_model())

    @override
    def __repr__(self) -> str:
        parts = []
        if self.file_data is not None:
            if self.file_data.startswith("data:"):
                mime_type = self.file_data.split(";")[0].split(":")[1]
                len_data = (
                    len(self.file_data.split("base64,")[1]) if "base64," in self.file_data else len(self.file_data)
                )
                parts.append(f"file_data=<DATA_URI({mime_type}, {len_data} chars)>")
            else:
                len_data = len(self.file_data)
                parts.append(f"file_data=<DATA({len_data} chars)>")
        if self.file_id is not None:
            parts.append(f"file_id='{self.file_id}'")
        if self.filename is not None:
            parts.append(f"filename='{self.filename}'")
        return f"File({', '.join(parts)})"

    @classmethod
    def from_path(cls, file_path: str, filename: str | None = None, mime_type: str | None = None) -> "File":
        path = Path(file_path)
        if not path.is_file():
            raise ValueError(f"File not found: {file_path}")
        with path.open("rb") as f:
            file_bytes = f.read()
        if filename is None:
            filename = path.name
        if mime_type is None:
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type is None:
                mime_type = "application/octet-stream"
        encoded_data = base64.b64encode(file_bytes).decode("utf-8")
        file_data = f"data:{mime_type};base64,{encoded_data}"
        return cls(file_data=file_data, filename=filename)

    @classmethod
    def from_bytes(
        cls, file_bytes: bytes, filename: str | None = None, mime_type: str = "application/octet-stream"
    ) -> "File":
        encoded_data = base64.b64encode(file_bytes).decode("utf-8")
        file_data = f"data:{mime_type};base64,{encoded_data}"
        return cls(file_data=file_data, filename=filename)

    @classmethod
    def from_file_id(cls, file_id: str, filename: str | None = None) -> "File":
        return cls(file_id=file_id, filename=filename)


def encode_file_to_dict(file_input: object) -> dict[str, str | None]:
    if isinstance(file_input, File):
        result = {}
        if file_input.file_data is not None:
            result["file_data"] = file_input.file_data
        if file_input.file_id is not None:
            result["file_id"] = file_input.file_id
        if file_input.filename is not None:
            result["filename"] = file_input.filename
        return result
    if isinstance(file_input, str):
        if Path(file_input).is_file():
            file_obj = File.from_path(file_input)
        else:
            raise ValueError(f"Unrecognized file string: {file_input}; must be a valid file path")
        return {"file_data": file_obj.file_data, "filename": file_obj.filename}
    if isinstance(file_input, bytes):
        file_obj = File.from_bytes(file_input)
        return {"file_data": file_obj.file_data}
    raise ValueError(f"Unsupported file input type: {type(file_input)}")
