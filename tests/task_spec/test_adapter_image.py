import asyncio
import contextlib
import os
import tempfile
from io import BytesIO

import pydantic
import pytest
import requests
from PIL import Image as PILImage

from dspy.adapters.types.image import Image, encode_image
from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.task_spec import FieldSpec, TaskSpec, make_task_spec
from dspy.teleprompt.compile_params import LabeledFewShotCompileParams
from dspy.teleprompt.vanilla import LabeledFewShot
from dspy.utils.dummies import DummyLM
from tests.task_spec.helpers import ts


@pytest.fixture
def sample_pil_image():
    url = "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"
    response = requests.get(url)
    response.raise_for_status()
    return PILImage.open(BytesIO(response.content))


@pytest.fixture
def sample_dspy_image_download():
    url = "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"
    return Image(url, download=True)


@pytest.fixture
def sample_url():
    return "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"


@pytest.fixture
def sample_dspy_image_no_download():
    return Image("https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg")


def count_messages_with_image_url_pattern(messages):
    pattern = {"type": "image_url", "image_url": {"url": lambda x: isinstance(x, str)}}
    try:

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
            if isinstance(obj, (list, tuple)):
                count += sum(count_patterns(v, pattern) for v in obj)
            return count

        return count_patterns(messages, pattern)
    except Exception:
        return 0


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


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "probabilistic_classification",
            "signature": "image: Image, class_labels: list[str] -> probabilities: dict[str, float]",
            "inputs": {"image": "https://example.com/dog.jpg", "class_labels": ["dog", "cat", "bird"]},
            "key_output": "probabilities",
            "expected": {"probabilities": {"dog": 0.8, "cat": 0.1, "bird": 0.1}},
        },
        {
            "name": "image_to_code",
            "signature": "ui_image: Image, target_language: str -> generated_code: str",
            "inputs": {"ui_image": "https://example.com/button.png", "target_language": "HTML"},
            "key_output": "generated_code",
            "expected": {"generated_code": "<button>Click me</button>"},
        },
        {
            "name": "bbox_detection",
            "signature": "image: Image -> bboxes: list[Tuple[int, int, int, int]]",
            "inputs": {"image": "https://example.com/image.jpg"},
            "key_output": "bboxes",
            "expected": {"bboxes": [(10, 20, 30, 40), (50, 60, 70, 80)]},
        },
        {
            "name": "multilingual_caption",
            "signature": "image: Image, languages: list[str] -> captions: dict[str, str]",
            "inputs": {"image": "https://example.com/dog.jpg", "languages": ["en", "es", "fr"]},
            "key_output": "captions",
            "expected": {
                "captions": {"en": "A golden retriever", "es": "Un golden retriever", "fr": "Un golden retriever"}
            },
        },
    ],
)
def test_basic_image_operations(test_case, make_run):
    predictor, lm, run = setup_predictor(test_case["signature"], test_case["expected"], make_run)
    inputs = {
        k: Image(v) if isinstance(v, str) and k in ["image", "ui_image"] else v for k, v in test_case["inputs"].items()
    }
    result = asyncio.run(predictor(**inputs, run=run))
    output_field = next(f for f in ["probabilities", "generated_code", "bboxes", "captions"] if hasattr(result, f))
    assert getattr(result, output_field) == test_case["expected"][test_case["key_output"]]
    assert count_messages_with_image_url_pattern(lm.call_log[-1].messages_as_openai) == 1


@pytest.mark.parametrize(
    ("image_input", "description"),
    [
        ("pil_image", "PIL Image"),
        ("encoded_pil_image", "encoded PIL image string"),
        ("dspy_image_download", "Image with download=True"),
        ("dspy_image_no_download", "Image without download"),
    ],
)
def test_image_input_formats(
    request,
    sample_pil_image,
    sample_dspy_image_download,
    sample_dspy_image_no_download,
    image_input,
    description,
    make_run,
):
    signature = "image: Image, class_labels: list[str] -> probabilities: dict[str, float]"
    expected = {"probabilities": {"dog": 0.8, "cat": 0.1, "bird": 0.1}}
    predictor, lm, run = setup_predictor(signature, expected, make_run)
    input_map = {
        "pil_image": sample_pil_image,
        "encoded_pil_image": encode_image(sample_pil_image),
        "dspy_image_download": sample_dspy_image_download,
        "dspy_image_no_download": sample_dspy_image_no_download,
    }
    actual_input = input_map[image_input]
    if image_input in ["pil_image", "encoded_pil_image"]:
        pytest.xfail(f"{description} not fully supported without Image coercion")
    result = asyncio.run(predictor(image=actual_input, class_labels=["dog", "cat", "bird"], run=run))
    assert result.probabilities == expected["probabilities"]
    assert count_messages_with_image_url_pattern(lm.call_log[-1].messages_as_openai) == 1


def test_predictor_save_load(sample_url, sample_pil_image, make_run):
    signature = "image: Image -> caption: str"
    examples = [
        Example.from_record({"image": Image(sample_url), "caption": "Example 1"}),
        Example.from_record({"image": sample_pil_image, "caption": "Example 2"}),
    ]
    predictor, lm, run = setup_predictor(signature, {"caption": "A golden retriever"}, make_run)
    optimizer = LabeledFewShot(k=1)
    compiled_predictor = asyncio.run(
        optimizer.compile(
            student=predictor, params=LabeledFewShotCompileParams(trainset=examples, sample=False), run=run
        )
    )
    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".json") as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = Predict(ts("image: Image -> caption: str"))
        loaded_predictor.load(temp_file.name)
    asyncio.run(loaded_predictor(image=Image("https://example.com/dog.jpg"), run=make_run(lm=lm)))
    assert count_messages_with_image_url_pattern(lm.call_log[-1].messages_as_openai) == 2
    assert "<DSPY_IMAGE_START>" not in str(lm.call_log[-1].messages_as_openai)


def test_save_load_complex_default_types(make_run):
    examples = [
        Example.from_record(
            {
                "image_list": [Image("https://example.com/dog.jpg"), Image("https://example.com/cat.jpg")],
                "caption": "Example 1",
            },
            input_keys=("image_list",),
        )
    ]
    ComplexTypeSignature = make_task_spec(
        {
            "image_list": FieldSpec.input("image_list", type_=list[Image], desc="A list of images"),
            "caption": FieldSpec.output("caption", desc="A caption for the image list"),
        },
        instructions="Caption image lists.",
        name="ComplexTypeSignature",
    )
    predictor, lm, run = setup_predictor(ComplexTypeSignature, {"caption": "A list of images"}, make_run)
    optimizer = LabeledFewShot(k=1)
    compiled_predictor = asyncio.run(
        optimizer.compile(
            student=predictor, params=LabeledFewShotCompileParams(trainset=examples, sample=False), run=run
        )
    )
    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".json") as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = Predict(ComplexTypeSignature)
        loaded_predictor.load(temp_file.name)
    result = asyncio.run(loaded_predictor(**examples[0].as_inputs(), run=make_run(lm=lm)))
    assert result.caption == "A list of images"
    assert str(lm.call_log[-1].messages_as_openai).count("'url'") == 4
    assert "<DSPY_IMAGE_START>" not in str(lm.call_log[-1].messages_as_openai)


BasicImageSignature = make_task_spec(
    {"image": FieldSpec.input("image", type_=Image), "output": FieldSpec.output("output")},
    instructions="Basic signature with a single image input.",
    name="BasicImageSignature",
)
ImageListSignature = make_task_spec(
    {"image_list": FieldSpec.input("image_list", type_=list[Image]), "output": FieldSpec.output("output")},
    instructions="Signature with a list of images input.",
    name="ImageListSignature",
)


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "basic_dspy_signature",
            "task_spec": BasicImageSignature,
            "inputs": {"image": "https://example.com/dog.jpg"},
            "expected": {"output": "A dog photo"},
            "expected_image_urls": 2,
        },
        {
            "name": "list_dspy_signature",
            "task_spec": ImageListSignature,
            "inputs": {"image_list": ["https://example.com/dog.jpg", "https://example.com/cat.jpg"]},
            "expected": {"output": "Multiple photos"},
            "expected_image_urls": 4,
        },
    ],
)
def test_save_load_complex_types(test_case, make_run):
    task_spec = test_case["task_spec"]
    processed_input = {}
    for key, value in test_case["inputs"].items():
        if isinstance(value, str) and "http" in value:
            processed_input[key] = Image(value)
        elif isinstance(value, list) and value and isinstance(value[0], str):
            processed_input[key] = [Image(url) for url in value]
        else:
            processed_input[key] = value
    examples = [
        Example.from_record({**processed_input, **test_case["expected"]}, input_keys=tuple(processed_input.keys()))
    ]
    predictor, lm, run = setup_predictor(task_spec, test_case["expected"], make_run)
    optimizer = LabeledFewShot(k=1)
    compiled_predictor = asyncio.run(
        optimizer.compile(
            student=predictor, params=LabeledFewShotCompileParams(trainset=examples, sample=False), run=run
        )
    )
    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".json") as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = Predict(task_spec)
        loaded_predictor.load(temp_file.name)
    result = asyncio.run(loaded_predictor(**processed_input, run=run))
    for key, value in test_case["expected"].items():
        assert getattr(result, key) == value
    assert count_messages_with_image_url_pattern(lm.call_log[-1].messages_as_openai) == test_case["expected_image_urls"]
    assert "<DSPY_IMAGE_START>" not in str(lm.call_log[-1].messages_as_openai)


def test_save_load_pydantic_model(make_run):

    class ImageModel(pydantic.BaseModel):
        image: Image
        image_list: list[Image] | None = None
        output: str

    PydanticSignature = make_task_spec(
        {"model_input": FieldSpec.input("model_input", type_=ImageModel), "output": FieldSpec.output("output")},
        instructions="Process pydantic image model.",
        name="PydanticSignature",
    )
    model_input = ImageModel(
        image=Image("https://example.com/dog.jpg"),
        image_list=[Image("https://example.com/cat.jpg")],
        output="Multiple photos",
    )
    examples = [
        Example.from_record({"model_input": model_input, "output": "Multiple photos"}, input_keys=("model_input",))
    ]
    predictor, lm, run = setup_predictor(PydanticSignature, {"output": "Multiple photos"}, make_run)
    optimizer = LabeledFewShot(k=1)
    compiled_predictor = asyncio.run(
        optimizer.compile(
            student=predictor, params=LabeledFewShotCompileParams(trainset=examples, sample=False), run=run
        )
    )
    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".json") as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = Predict(PydanticSignature)
        import json

        with open(temp_file.name, encoding="utf-8") as saved_file:
            state = json.load(saved_file)
        model_input_type = state["task_spec"]["inputs"][0]["type"]
        loaded_predictor.load_state(state, custom_types={model_input_type: ImageModel})
    result = asyncio.run(loaded_predictor(model_input=model_input, run=run))
    assert result.output == "Multiple photos"
    assert count_messages_with_image_url_pattern(lm.call_log[-1].messages_as_openai) == 4
    assert "<DSPY_IMAGE_START>" not in str(lm.call_log[-1].messages_as_openai)


def test_optional_image_field(make_run):
    OptionalImageSignature = make_task_spec(
        {"image": FieldSpec.input("image", type_=Image | None), "output": FieldSpec.output("output")},
        instructions="Process optional image.",
        name="OptionalImageSignature",
    )
    predictor, lm, run = setup_predictor(OptionalImageSignature, {"output": "Hello"}, make_run)
    result = asyncio.run(predictor(image=None, run=run))
    assert result.output == "Hello"
    assert count_messages_with_image_url_pattern(lm.call_log[-1].messages_as_openai) == 0


def test_pdf_url_support(make_run):
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    pdf_image = Image(pdf_url, download=True)
    assert "data:application/pdf" in pdf_image.url
    assert ";base64," in pdf_image.url
    PDFSignature = make_task_spec(
        {
            "document": FieldSpec.input("document", type_=Image, desc="A PDF document"),
            "summary": FieldSpec.output("summary", desc="A summary of the PDF"),
        },
        instructions="Summarize PDF documents.",
        name="PDFSignature",
    )
    predictor, lm, run = setup_predictor(PDFSignature, {"summary": "This is a dummy PDF"}, make_run)
    result = asyncio.run(predictor(document=pdf_image, run=run))
    assert result.summary == "This is a dummy PDF"
    assert count_messages_with_image_url_pattern(lm.call_log[-1].messages_as_openai) == 1
    messages_str = str(lm.call_log[-1].messages_as_openai)
    assert "application/pdf" in messages_str


def test_different_mime_types(make_run):
    file_urls = {
        "pdf": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "image": "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg",
    }
    expected_mime_types = {"pdf": "application/pdf", "image": "image/jpeg"}
    for file_type, url in file_urls.items():
        encoded = encode_image(url, download_images=True)
        assert f"data:{expected_mime_types[file_type]}" in encoded
        assert ";base64," in encoded


def test_mime_type_from_response_headers(make_run):
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    response = requests.get(pdf_url)
    expected_mime_type = response.headers.get("Content-Type", "")
    assert "pdf" in expected_mime_type.lower()
    encoded = encode_image(pdf_url, download_images=True)
    assert "application/pdf" in encoded
    assert ";base64," in encoded


def test_pdf_from_file(make_run):
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    response = requests.get(pdf_url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name
    try:
        pdf_image = Image(tmp_file_path)
        assert "data:application/pdf" in pdf_image.url
        assert ";base64," in pdf_image.url
        FilePDFSignature = make_task_spec(
            {
                "document": FieldSpec.input("document", type_=Image, desc="A PDF document from file"),
                "summary": FieldSpec.output("summary", desc="A summary of the PDF"),
            },
            instructions="Summarize PDF from file.",
            name="FilePDFSignature",
        )
        predictor, lm, run = setup_predictor(FilePDFSignature, {"summary": "This is a PDF from file"}, make_run)
        result = asyncio.run(predictor(document=pdf_image, run=run))
        assert result.summary == "This is a PDF from file"
        assert count_messages_with_image_url_pattern(lm.call_log[-1].messages_as_openai) == 1
    finally:
        with contextlib.suppress(Exception):
            os.unlink(tmp_file_path)


def test_image_repr():
    url_image = Image("https://example.com/dog.jpg")
    assert str(url_image) == '[{"type": "image_url", "image_url": {"url": "https://example.com/dog.jpg"}}]'
    assert repr(url_image) == "Image(url='https://example.com/dog.jpg')"
    sample_pil = PILImage.new("RGB", (60, 30), color="red")
    pil_image = Image(sample_pil)
    assert str(pil_image).startswith('[{"type": "image_url",')
    assert str(pil_image).endswith("}]")
    assert "base64" in str(pil_image)


def test_invalid_string_format():
    invalid_string = "this_is_not_a_url_or_file"
    with pytest.raises(ValueError, match="Unrecognized"):
        Image(invalid_string)


def test_pil_image_with_download_parameter():
    sample_pil = PILImage.new("RGB", (60, 30), color="red")
    image_no_download = Image(sample_pil)
    image_with_download = Image(sample_pil, download=True)
    assert image_no_download.url.startswith("data:")
    assert image_with_download.url.startswith("data:")
    assert "base64," in image_no_download.url
    assert "base64," in image_with_download.url
    assert image_no_download.url == image_with_download.url
