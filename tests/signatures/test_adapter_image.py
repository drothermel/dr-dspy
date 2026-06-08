import contextlib
import os
import tempfile
from io import BytesIO

import pydantic
import pytest
import requests
from PIL import Image as PILImage

from dspy.adapters.types.image import Image, encode_image
from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Predict
from dspy.primitives.example import Example
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature
from dspy.teleprompt.vanilla import LabeledFewShot
from dspy.utils.dummies import DummyLM


@pytest.fixture
def sample_pil_image():
    """Fixture to provide a sample image for testing"""
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
                return all(k in obj and check_pattern(obj[k], v) for k, v in pattern.items())
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


def setup_predictor(signature, expected_output):
    """Helper to set up a predictor with DummyLM"""
    lm = DummyLM([expected_output])
    settings.configure(lm=lm)
    return Predict(signature), lm


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
def test_basic_image_operations(test_case):
    """Consolidated test for basic image operations"""
    predictor, lm = setup_predictor(test_case["signature"], test_case["expected"])

    # Convert string URLs to Image objects
    inputs = {
        k: Image(v) if isinstance(v, str) and k in ["image", "ui_image"] else v for k, v in test_case["inputs"].items()
    }

    result = predictor(**inputs)

    # Check result based on output field name
    output_field = next(f for f in ["probabilities", "generated_code", "bboxes", "captions"] if hasattr(result, f))
    assert getattr(result, output_field) == test_case["expected"][test_case["key_output"]]
    assert count_messages_with_image_url_pattern(lm.history[-1].messages_as_openai) == 1


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
    request, sample_pil_image, sample_dspy_image_download, sample_dspy_image_no_download, image_input, description
):
    """Test different input formats for image fields"""
    signature = "image: Image, class_labels: list[str] -> probabilities: dict[str, float]"
    expected = {"probabilities": {"dog": 0.8, "cat": 0.1, "bird": 0.1}}
    predictor, lm = setup_predictor(signature, expected)

    input_map = {
        "pil_image": sample_pil_image,
        "encoded_pil_image": encode_image(sample_pil_image),
        "dspy_image_download": sample_dspy_image_download,
        "dspy_image_no_download": sample_dspy_image_no_download,
    }

    actual_input = input_map[image_input]
    # TODO: Support PIL/base64 inputs without requiring callers to pre-coerce them to Image, then remove this xfail.
    if image_input in ["pil_image", "encoded_pil_image"]:
        pytest.xfail(f"{description} not fully supported without Image coercion")  # ty:ignore[too-many-positional-arguments]

    result = predictor(image=actual_input, class_labels=["dog", "cat", "bird"])
    assert result.probabilities == expected["probabilities"]
    assert count_messages_with_image_url_pattern(lm.history[-1].messages_as_openai) == 1


def test_predictor_save_load(sample_url, sample_pil_image):
    """Test saving and loading predictors with image fields"""
    signature = "image: Image -> caption: str"
    examples = [
        Example(image=Image(sample_url), caption="Example 1"),
        Example(image=sample_pil_image, caption="Example 2"),
    ]

    predictor, lm = setup_predictor(signature, {"caption": "A golden retriever"})
    optimizer = LabeledFewShot(k=1)
    compiled_predictor = optimizer.compile(student=predictor, trainset=examples, sample=False)

    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".json") as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = Predict(signature)
        loaded_predictor.load(temp_file.name)

    loaded_predictor(image=Image("https://example.com/dog.jpg"))
    assert count_messages_with_image_url_pattern(lm.history[-1].messages_as_openai) == 2
    assert "<DSPY_IMAGE_START>" not in str(lm.history[-1].messages_as_openai)


def test_save_load_complex_default_types():
    """Test saving and loading predictors with complex default types (lists of images)"""
    examples = [
        Example(
            image_list=[
                Image("https://example.com/dog.jpg"),
                Image("https://example.com/cat.jpg"),
            ],
            caption="Example 1",
        ).with_inputs("image_list"),
    ]

    class ComplexTypeSignature(Signature):
        image_list: list[Image] = InputField(desc="A list of images")
        caption: str = OutputField(desc="A caption for the image list")

    predictor, lm = setup_predictor(ComplexTypeSignature, {"caption": "A list of images"})
    optimizer = LabeledFewShot(k=1)
    compiled_predictor = optimizer.compile(student=predictor, trainset=examples, sample=False)

    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".json") as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = Predict(ComplexTypeSignature)
        loaded_predictor.load(temp_file.name)

    result = loaded_predictor(**examples[0].inputs())
    assert result.caption == "A list of images"
    assert str(lm.history[-1].messages_as_openai).count("'url'") == 4
    assert "<DSPY_IMAGE_START>" not in str(lm.history[-1].messages_as_openai)


class BasicImageSignature(Signature):
    """Basic signature with a single image input"""

    image: Image = InputField()
    output: str = OutputField()


class ImageListSignature(Signature):
    """Signature with a list of images input"""

    image_list: list[Image] = InputField()
    output: str = OutputField()


@pytest.mark.parametrize(
    "test_case",
    [
        {
            "name": "basic_dspy_signature",
            "signature_class": BasicImageSignature,
            "inputs": {"image": "https://example.com/dog.jpg"},
            "expected": {"output": "A dog photo"},
            "expected_image_urls": 2,
        },
        {
            "name": "list_dspy_signature",
            "signature_class": ImageListSignature,
            "inputs": {"image_list": ["https://example.com/dog.jpg", "https://example.com/cat.jpg"]},
            "expected": {"output": "Multiple photos"},
            "expected_image_urls": 4,
        },
    ],
)
def test_save_load_complex_types(test_case):
    """Test saving and loading predictors with complex types"""
    signature_cls = test_case["signature_class"]

    # Convert string URLs to Image objects in input
    processed_input = {}
    for key, value in test_case["inputs"].items():
        if isinstance(value, str) and "http" in value:
            processed_input[key] = Image(value)
        elif isinstance(value, list) and value and isinstance(value[0], str):
            processed_input[key] = [Image(url) for url in value]
        else:
            processed_input[key] = value

    # Create example and predictor
    examples = [Example(**processed_input, **test_case["expected"]).with_inputs(*processed_input.keys())]

    predictor, lm = setup_predictor(signature_cls, test_case["expected"])
    optimizer = LabeledFewShot(k=1)
    compiled_predictor = optimizer.compile(student=predictor, trainset=examples, sample=False)

    # Test save and load
    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".json") as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = Predict(signature_cls)
        loaded_predictor.load(temp_file.name)

    # Run prediction
    result = loaded_predictor(**processed_input)

    # Verify output matches expected
    for key, value in test_case["expected"].items():
        assert getattr(result, key) == value

    # Verify correct number of image URLs in messages
    assert count_messages_with_image_url_pattern(lm.history[-1].messages_as_openai) == test_case["expected_image_urls"]
    assert "<DSPY_IMAGE_START>" not in str(lm.history[-1].messages_as_openai)


def test_save_load_pydantic_model():
    """Test saving and loading predictors with pydantic models"""

    class ImageModel(pydantic.BaseModel):
        image: Image
        image_list: list[Image] | None = None
        output: str

    class PydanticSignature(Signature):
        model_input: ImageModel = InputField()
        output: str = OutputField()

    # Create model instance
    model_input = ImageModel(
        image=Image("https://example.com/dog.jpg"),
        image_list=[Image("https://example.com/cat.jpg")],
        output="Multiple photos",
    )

    # Create example and predictor
    examples = [Example(model_input=model_input, output="Multiple photos").with_inputs("model_input")]

    predictor, lm = setup_predictor(PydanticSignature, {"output": "Multiple photos"})
    optimizer = LabeledFewShot(k=1)
    compiled_predictor = optimizer.compile(student=predictor, trainset=examples, sample=False)

    # Test save and load
    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".json") as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = Predict(PydanticSignature)
        loaded_predictor.load(temp_file.name)

    # Run prediction
    result = loaded_predictor(model_input=model_input)

    # Verify output matches expected
    assert result.output == "Multiple photos"
    assert count_messages_with_image_url_pattern(lm.history[-1].messages_as_openai) == 4
    assert "<DSPY_IMAGE_START>" not in str(lm.history[-1].messages_as_openai)


def test_optional_image_field():
    """Test that optional image fields are not required"""

    class OptionalImageSignature(Signature):
        image: Image | None = InputField()
        output: str = OutputField()

    predictor, lm = setup_predictor(OptionalImageSignature, {"output": "Hello"})
    result = predictor(image=None)
    assert result.output == "Hello"
    assert count_messages_with_image_url_pattern(lm.history[-1].messages_as_openai) == 0


def test_pdf_url_support():
    """Test support for PDF files from URLs"""
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

    # Create a Image object from the PDF URL with download=True
    pdf_image = Image(pdf_url, download=True)

    # The data URI should contain application/pdf in the MIME type
    assert "data:application/pdf" in pdf_image.url
    assert ";base64," in pdf_image.url

    # Test using it in a predictor
    class PDFSignature(Signature):
        document: Image = InputField(desc="A PDF document")
        summary: str = OutputField(desc="A summary of the PDF")

    predictor, lm = setup_predictor(PDFSignature, {"summary": "This is a dummy PDF"})
    result = predictor(document=pdf_image)

    assert result.summary == "This is a dummy PDF"
    assert count_messages_with_image_url_pattern(lm.history[-1].messages_as_openai) == 1

    # Ensure the URL was properly expanded in messages
    messages_str = str(lm.history[-1].messages_as_openai)
    assert "application/pdf" in messages_str


def test_different_mime_types():
    """Test support for different file types and MIME type detection"""
    # Test with various file types
    file_urls = {
        "pdf": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "image": "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg",
    }

    expected_mime_types = {
        "pdf": "application/pdf",
        "image": "image/jpeg",
    }

    for file_type, url in file_urls.items():
        # Download and encode
        encoded = encode_image(url, download_images=True)

        # Check for correct MIME type in the encoded data - using 'in' instead of startswith
        # to account for possible parameters in the MIME type
        assert f"data:{expected_mime_types[file_type]}" in encoded
        assert ";base64," in encoded


def test_mime_type_from_response_headers():
    """Test that MIME types from response headers are correctly used"""
    # This URL returns proper Content-Type header
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

    # Make an actual request to get the content type from headers
    response = requests.get(pdf_url)
    expected_mime_type = response.headers.get("Content-Type", "")

    # Should be application/pdf or similar
    assert "pdf" in expected_mime_type.lower()

    # Encode with download to test MIME type from headers
    encoded = encode_image(pdf_url, download_images=True)

    # The encoded data should contain the correct MIME type
    assert "application/pdf" in encoded
    assert ";base64," in encoded


def test_pdf_from_file():
    """Test handling a PDF file from disk"""
    # Download a PDF to a temporary file
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    response = requests.get(pdf_url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    try:
        # Create a Image from the file
        pdf_image = Image(tmp_file_path)

        # The constructor encodes the file into a data URI we can inspect directly
        assert "data:application/pdf" in pdf_image.url
        assert ";base64," in pdf_image.url

        # Test the image in a predictor
        class FilePDFSignature(Signature):
            document: Image = InputField(desc="A PDF document from file")
            summary: str = OutputField(desc="A summary of the PDF")

        predictor, lm = setup_predictor(FilePDFSignature, {"summary": "This is a PDF from file"})
        result = predictor(document=pdf_image)

        assert result.summary == "This is a PDF from file"
        assert count_messages_with_image_url_pattern(lm.history[-1].messages_as_openai) == 1
    finally:
        # Clean up the temporary file
        with contextlib.suppress(Exception):
            os.unlink(tmp_file_path)


def test_image_repr():
    """Test string representation of Image objects"""
    url_image = Image("https://example.com/dog.jpg")
    assert str(url_image) == (
        "<<CUSTOM-TYPE-START-IDENTIFIER>>"
        '[{"type": "image_url", "image_url": {"url": "https://example.com/dog.jpg"}}]'
        "<<CUSTOM-TYPE-END-IDENTIFIER>>"
    )
    assert repr(url_image) == "Image(url='https://example.com/dog.jpg')"

    sample_pil = PILImage.new("RGB", (60, 30), color="red")
    pil_image = Image(sample_pil)
    assert str(pil_image).startswith('<<CUSTOM-TYPE-START-IDENTIFIER>>[{"type": "image_url",')
    assert str(pil_image).endswith("<<CUSTOM-TYPE-END-IDENTIFIER>>")
    assert "base64" in str(pil_image)


def test_invalid_string_format():
    """Test that invalid string formats raise a ValueError"""
    invalid_string = "this_is_not_a_url_or_file"

    # Should raise a ValueError and not pass the string through
    with pytest.raises(ValueError, match="Unrecognized"):
        Image(invalid_string)


def test_pil_image_with_download_parameter():
    """Test behavior when PIL image is passed with download=True"""
    sample_pil = PILImage.new("RGB", (60, 30), color="red")

    # PIL image should be encoded regardless of download parameter
    image_no_download = Image(sample_pil)
    image_with_download = Image(sample_pil, download=True)

    # Both should result in base64 encoded data URIs
    assert image_no_download.url.startswith("data:")
    assert image_with_download.url.startswith("data:")
    assert "base64," in image_no_download.url
    assert "base64," in image_with_download.url

    # They should be identical since PIL images are always encoded
    assert image_no_download.url == image_with_download.url
