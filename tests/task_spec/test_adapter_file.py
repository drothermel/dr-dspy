import asyncio
import contextlib
import os
import tempfile
from typing import Any, cast

import pydantic
import pytest

from dspy.adapters.types.file import File, encode_file_to_dict
from dspy.predict.predict import Predict
from dspy.primitives import Example
from dspy.task_spec import TaskSpec, input_field, make_task_spec, output_field
from dspy.teleprompt.compile_params import LabeledFewShotCompileParams
from dspy.teleprompt.vanilla import LabeledFewShot
from dspy.testing import DummyLM
from tests.task_spec.helpers import ts


@pytest.fixture
def sample_text_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp_file:
        tmp_file.write("This is a test file.")
        tmp_file_path = tmp_file.name
    yield tmp_file_path
    with contextlib.suppress(Exception):
        os.unlink(tmp_file_path)


def count_messages_with_file_pattern(messages):
    pattern = {"type": "file", "file": lambda x: isinstance(x, dict)}

    def check_pattern(obj, pattern):
        if isinstance(pattern, dict):
            if not isinstance(obj, dict):
                return False
            return all((k in obj and check_pattern(obj[k], v) for k, v in pattern.items()))
        if callable(pattern):
            return pattern(obj)
        return obj == pattern

    def count_patterns(obj, pattern):
        count = 0
        if check_pattern(obj, pattern):
            count += 1
        if isinstance(obj, dict):
            count += sum(count_patterns(v, pattern) for v in obj.values())
        if isinstance(obj, list | tuple):
            count += sum(count_patterns(v, pattern) for v in obj)
        return count

    return count_patterns(messages, pattern)


def setup_predictor(spec, expected_output, make_run):
    lm = DummyLM([expected_output])
    run = make_run(lm=lm)
    if isinstance(spec, str):
        task_spec = ts(spec)
    elif isinstance(spec, TaskSpec):
        task_spec = spec
    else:
        raise TypeError(f"Expected str or TaskSpec, got {type(spec).__name__}")
    return Predict(task_spec), lm, run


def test_file_from_local_path(sample_text_file):
    file_obj = File.from_path(sample_text_file)
    assert file_obj.file_data is not None
    assert file_obj.file_data.startswith("data:text/plain;base64,")
    assert file_obj.filename == os.path.basename(sample_text_file)


def test_file_from_path_method(sample_text_file):
    file_obj = File.from_path(sample_text_file)
    assert file_obj.file_data is not None
    assert file_obj.file_data.startswith("data:text/plain;base64,")
    assert file_obj.filename == os.path.basename(sample_text_file)


def test_file_from_path_with_custom_filename(sample_text_file):
    file_obj = File.from_path(sample_text_file, filename="custom.txt")
    assert file_obj.file_data is not None
    assert file_obj.file_data.startswith("data:text/plain;base64,")
    assert file_obj.filename == "custom.txt"


def test_file_from_bytes():
    file_bytes = b"Test file content"
    file_obj = File.from_bytes(file_bytes)
    assert file_obj.file_data is not None
    assert file_obj.file_data.startswith("data:application/octet-stream;base64,")
    assert file_obj.filename is None


def test_file_from_bytes_with_filename():
    file_bytes = b"Test file content"
    file_obj = File.from_bytes(file_bytes, filename="test.txt")
    assert file_obj.file_data is not None
    assert file_obj.file_data.startswith("data:application/octet-stream;base64,")
    assert file_obj.filename == "test.txt"


def test_file_from_file_id():
    file_obj = File.from_file_id("file-abc123")
    assert file_obj.file_id == "file-abc123"
    assert file_obj.file_data is None


def test_file_from_file_id_with_filename():
    file_obj = File.from_file_id("file-abc123", filename="document.pdf")
    assert file_obj.file_id == "file-abc123"
    assert file_obj.filename == "document.pdf"


def test_file_from_dict_with_file_data(make_run):
    file_obj = File(file_data="data:text/plain;base64,dGVzdA==", filename="test.txt")
    assert file_obj.file_data == "data:text/plain;base64,dGVzdA=="
    assert file_obj.filename == "test.txt"


def test_file_from_dict_with_file_id(make_run):
    file_obj = File(file_id="file-xyz789")
    assert file_obj.file_id == "file-xyz789"


def test_file_format_with_file_data(make_run):
    file_obj = File.from_bytes(b"test", filename="test.txt")
    formatted = file_obj.format()
    assert isinstance(formatted, list)
    assert len(formatted) == 1
    assert formatted[0]["type"] == "file"
    assert "file" in formatted[0]
    assert "file_data" in formatted[0]["file"]
    assert "filename" in formatted[0]["file"]


def test_file_format_with_file_id(make_run):
    file_obj = File.from_file_id("file-123")
    formatted = file_obj.format()
    assert formatted[0]["type"] == "file"
    assert formatted[0]["file"]["file_id"] == "file-123"


def test_file_repr_with_file_data(make_run):
    file_obj = File.from_bytes(b"Test content", filename="test.txt")
    repr_str = repr(file_obj)
    assert "DATA_URI" in repr_str
    assert "application/octet-stream" in repr_str
    assert "filename='test.txt'" in repr_str


def test_file_repr_with_file_id(make_run):
    file_obj = File.from_file_id("file-abc", filename="doc.pdf")
    repr_str = repr(file_obj)
    assert "file_id='file-abc'" in repr_str
    assert "filename='doc.pdf'" in repr_str


def test_file_str(make_run):
    file_obj = File.from_bytes(b"test")
    str_repr = str(file_obj)
    assert str_repr.startswith('[{"type": "file",')
    assert str_repr.endswith("}]")


def test_encode_file_to_dict_from_path(sample_text_file, make_run):
    result = encode_file_to_dict(sample_text_file)
    assert "file_data" in result
    assert result["file_data"] is not None
    assert result["file_data"].startswith("data:text/plain;base64,")
    assert "filename" in result


def test_encode_file_to_dict_from_bytes(make_run):
    result = encode_file_to_dict(b"test content")
    assert "file_data" in result
    assert result["file_data"] is not None
    assert result["file_data"].startswith("data:application/octet-stream;base64,")


def test_invalid_file_string(make_run):
    with pytest.raises(ValueError, match="Unrecognized"):
        encode_file_to_dict("https://this_is_not_a_file_path")


def test_invalid_dict(make_run):
    with pytest.raises(ValueError, match="must contain at least one"):
        File(**cast("dict[str, Any]", {"invalid": "dict"}))


def test_file_in_signature(sample_text_file, make_run):
    signature = "document: File -> summary: str"
    expected = {"summary": "This is a summary"}
    predictor, lm, run = setup_predictor(signature, expected, make_run)
    file_obj = File.from_path(sample_text_file)
    result = asyncio.run(predictor(document=file_obj, run=run))
    assert result.summary == "This is a summary"
    assert count_messages_with_file_pattern(lm.call_log[-1].messages_as_openai) == 1


def test_file_list_in_signature(sample_text_file, make_run):
    FileListSignature = make_task_spec(
        {
            "documents": input_field("documents", type_=list[File], desc="The documents."),
            "summary": output_field("summary", desc="The summary."),
        },
        instructions="Summarize documents.",
        name="FileListSignature",
    )
    expected = {"summary": "Multiple files"}
    predictor, lm, run = setup_predictor(FileListSignature, expected, make_run)
    files = [File.from_path(sample_text_file), File.from_file_id("file-123")]
    result = asyncio.run(predictor(documents=files, run=run))
    assert result.summary == "Multiple files"
    assert count_messages_with_file_pattern(lm.call_log[-1].messages_as_openai) == 2


def test_optional_file_field(make_run):
    OptionalFileSignature = make_task_spec(
        {
            "document": input_field("document", type_=File | None, desc="The document."),
            "output": output_field("output", desc="The output."),
        },
        instructions="Process optional file.",
        name="OptionalFileSignature",
    )
    predictor, lm, run = setup_predictor(OptionalFileSignature, {"output": "Hello"}, make_run)
    result = asyncio.run(predictor(document=None, run=run))
    assert result.output == "Hello"
    assert count_messages_with_file_pattern(lm.call_log[-1].messages_as_openai) == 0


def test_save_load_file_signature(sample_text_file, make_run):
    signature = "document: File -> summary: str"
    file_obj = File.from_path(sample_text_file)
    examples = [Example.from_record({"document": file_obj, "summary": "Test summary"})]
    predictor, lm, run = setup_predictor(signature, {"summary": "A summary"}, make_run)
    optimizer = LabeledFewShot(k=1)
    compile_result = asyncio.run(
        optimizer.compile(
            student=predictor, params=LabeledFewShotCompileParams(trainset=examples, sample=False), run=run
        )
    )
    compiled_predictor = compile_result.program
    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".json") as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = Predict(ts("document: File -> summary: str"))
        loaded_predictor.load(temp_file.name)
    asyncio.run(loaded_predictor(document=File.from_file_id("file-test"), run=make_run(lm=lm)))
    assert count_messages_with_file_pattern(lm.call_log[-1].messages_as_openai) == 2


def test_file_frozen():
    file_obj = File.from_bytes(b"test")
    with pytest.raises((TypeError, ValueError, pydantic.ValidationError)):
        file_obj.file_data = "new_data"


def test_file_with_all_fields():
    file_data_uri = "data:text/plain;base64,dGVzdA=="
    file_obj = File(file_data=file_data_uri, file_id="file-123", filename="test.txt")
    assert file_obj.file_data == file_data_uri
    assert file_obj.file_id == "file-123"
    assert file_obj.filename == "test.txt"
    formatted = file_obj.format()
    assert formatted[0]["file"]["file_data"] == file_data_uri
    assert formatted[0]["file"]["file_id"] == "file-123"
    assert formatted[0]["file"]["filename"] == "test.txt"


def test_file_path_not_found():
    with pytest.raises(ValueError, match="File not found"):
        File.from_path("/nonexistent/path/file.txt")


def test_file_custom_mime_type(sample_text_file):
    file_obj = File.from_path(sample_text_file, mime_type="text/custom")
    assert file_obj.file_data is not None
    assert file_obj.file_data.startswith("data:text/custom;base64,")


def test_file_from_bytes_custom_mime():
    file_obj = File.from_bytes(b"audio data", mime_type="audio/mp3")
    assert file_obj.file_data is not None
    assert file_obj.file_data.startswith("data:audio/mp3;base64,")


def test_file_data_uri_in_format():
    file_obj = File.from_bytes(b"test", filename="test.txt", mime_type="text/plain")
    formatted = file_obj.format()
    assert "data:text/plain;base64," in formatted[0]["file"]["file_data"]
