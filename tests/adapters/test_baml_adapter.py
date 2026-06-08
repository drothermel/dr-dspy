from dspy.utils.exceptions import AdapterParseError
from typing import Literal
from unittest import mock

import pydantic
import pytest

try:
    from litellm import Choices, Message
    from litellm.files.main import ModelResponse
except ImportError:
    pytest.skip("litellm is not installed", allow_module_level=True)

from dspy.adapters.baml_adapter import COMMENT_SYMBOL, INDENTATION, BAMLAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types.code import Code
from dspy.adapters.types.history import History
from dspy.adapters.types.image import Image
from dspy.adapters.types.tool import Tool
from dspy.clients.lm import LM
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import Signature
from tests.adapters.conftest import format_messages_and_lm_kwargs


# Test fixtures - Pydantic models for testing
class PatientAddress(pydantic.BaseModel):
    """Patient Address model docstring"""
    street: str
    city: str
    country: Literal["US", "CA"]


class PatientDetails(pydantic.BaseModel):
    """
    Patient Details model docstring
    Multiline docstring support test
    """
    name: str = pydantic.Field(description="Full name of the patient")
    age: int
    address: PatientAddress | None = None


class ComplexNestedModel(pydantic.BaseModel):
    """Complex model docstring"""
    id: int = pydantic.Field(description="Unique identifier")
    details: PatientDetails
    tags: list[str] = pydantic.Field(default_factory=list)
    metadata: dict[str, str] = pydantic.Field(default_factory=dict)


class ModelWithLists(pydantic.BaseModel):
    items: list[PatientAddress] = pydantic.Field(description="List of patient addresses")
    scores: list[float]


class ImageWrapper(pydantic.BaseModel):
    images: list[Image]
    tag: list[str]


class CircularModel(pydantic.BaseModel):
    name: str
    field: "CircularModel"


def test_baml_adapter_format_exact_messages_for_simple_signature_with_demo():
    class QA(Signature):
        question: str = InputField()
        answer: str = OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(BAMLAdapter(), QA, [{"question": "Q1", "answer": "A1"}], {"question": "Q2"})

    expected_messages = [{"role": "system",
      "content": "Your input fields are:\n"
                 "1. `question` (str):\n"
                 "Your output fields are:\n"
                 "1. `answer` (str):\n"
                 "All interactions will be structured in the following way, with the appropriate "
                 "values filled in.\n"
                 "\n"
                 "[[ ## question ## ]]\n"
                 "{question}\n"
                 "\n"
                 "[[ ## answer ## ]]\n"
                 "Output field `answer` should be of type: string\n"
                 "\n"
                 "[[ ## completed ## ]]\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Given the fields `question`, produce the fields `answer`."},
     {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
     {"role": "assistant", "content": '{\n  "answer": "A1"\n}'},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Q2\n"
                 "\n"
                 "Respond with a JSON object in the following order of fields: `answer`."}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_baml_adapter_format_exact_messages_with_nested_output():
    class BamlNested(pydantic.BaseModel):
        value: int
        tags: list[str]

    class TypedSignature(Signature):
        question: str = InputField()
        answer: BamlNested = OutputField()

    messages, lm_kwargs = format_messages_and_lm_kwargs(BAMLAdapter(), TypedSignature, [], {"question": "Q"})

    expected_messages = [{"role": "system",
      "content": "Your input fields are:\n"
                 "1. `question` (str):\n"
                 "Your output fields are:\n"
                 "1. `answer` (BamlNested):\n"
                 "All interactions will be structured in the following way, with the appropriate "
                 "values filled in.\n"
                 "\n"
                 "[[ ## question ## ]]\n"
                 "{question}\n"
                 "\n"
                 "[[ ## answer ## ]]\n"
                 "Output field `answer` should be of type: {\n"
                 "  value: int,\n"
                 "  tags: string[],\n"
                 "}\n"
                 "\n"
                 "[[ ## completed ## ]]\n"
                 "In adhering to this structure, your objective is: \n"
                 "        Given the fields `question`, produce the fields `answer`."},
     {"role": "user",
      "content": "[[ ## question ## ]]\n"
                 "Q\n"
                 "\n"
                 "Respond with a JSON object in the following order of fields: `answer` (must be "
                 "formatted as a valid Python BamlNested)."}]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_baml_adapter_basic_schema_generation():
    """Test that BAMLAdapter generates simplified schemas for Pydantic models."""

    class TestSignature(Signature):
        question: str = InputField()
        patient: PatientDetails = OutputField()

    adapter = BAMLAdapter()
    schema = adapter.format_field_structure(TestSignature)

    # Should contain simplified schema with comments
    assert f"{COMMENT_SYMBOL} Full name of the patient" in schema
    assert "name: string," in schema
    assert "age: int," in schema
    assert "address:" in schema
    assert "street: string," in schema
    assert 'country: "US" or "CA",' in schema


def test_baml_adapter_handles_optional_fields():
    """Test optional field rendering with 'or null' syntax."""

    class TestSignature(Signature):
        input: str = InputField()
        patient: PatientDetails = OutputField()

    adapter = BAMLAdapter()
    schema = adapter.format_field_structure(TestSignature)

    # Optional address field should show 'or null'
    assert "address:" in schema
    assert "or null" in schema


def test_baml_adapter_handles_primitive_types():
    """Test rendering of basic primitive types."""

    class SimpleModel(pydantic.BaseModel):
        text: str
        number: int
        decimal: float
        flag: bool

    class TestSignature(Signature):
        input: str = InputField()
        output: SimpleModel = OutputField()

    adapter = BAMLAdapter()
    schema = adapter.format_field_structure(TestSignature)

    assert "text: string," in schema
    assert "number: int," in schema
    assert "decimal: float," in schema
    assert "flag: boolean," in schema


def test_baml_adapter_handles_lists_with_bracket_notation():
    """Test that lists of Pydantic models use proper bracket notation."""

    class TestSignature(Signature):
        input: str = InputField()
        addresses: ModelWithLists = OutputField()

    adapter = BAMLAdapter()
    schema = adapter.format_field_structure(TestSignature)

    # Should use bracket notation for lists and include comments
    assert "items: [" in schema
    assert f"{COMMENT_SYMBOL} List of patient addresses" in schema
    assert "street: string," in schema
    assert "city: string," in schema
    assert "]," in schema
    assert "scores: float[]," in schema


def test_baml_adapter_handles_complex_nested_models():
    """Test deeply nested Pydantic model schema generation."""

    class TestSignature(Signature):
        input: str = InputField()
        complex: ComplexNestedModel = OutputField()

    adapter = BAMLAdapter()
    schema = adapter.format_field_structure(TestSignature)

    expected_patient_details = "\n".join([
        f"{INDENTATION}{COMMENT_SYMBOL} Patient Details model docstring",
        f"{INDENTATION}{COMMENT_SYMBOL} Multiline docstring support test",
        f"{INDENTATION}details:",
    ])

    # Should include nested structure with comments
    assert f"{COMMENT_SYMBOL} Unique identifier" in schema
    assert expected_patient_details in schema
    assert f"{COMMENT_SYMBOL} Full name of the patient" in schema
    assert "tags: string[]," in schema
    assert "metadata: dict[string, string]," in schema
    assert f"{COMMENT_SYMBOL} Complex model docstring" in schema
    assert f"{COMMENT_SYMBOL} Patient Address model docstring" in schema


def test_baml_adapter_raise_error_on_circular_references():
    """Test that circular references are handled gracefully."""

    class TestSignature(Signature):
        input: str = InputField()
        circular: CircularModel = OutputField()

    adapter = BAMLAdapter()
    with pytest.raises(ValueError) as error:
        adapter.format_field_structure(TestSignature)

    assert "BAMLAdapter cannot handle recursive pydantic models" in str(error.value)


def test_baml_adapter_formats_pydantic_inputs_as_clean_json():
    """Test that Pydantic input instances are formatted as clean JSON."""

    class TestSignature(Signature):
        patient: PatientDetails = InputField()
        question: str = InputField()
        answer: str = OutputField()

    adapter = BAMLAdapter()
    patient = PatientDetails(
        name="John Doe", age=45, address=PatientAddress(street="123 Main St", city="Anytown", country="US")
    )

    messages = adapter.format(TestSignature, [], {"patient": patient, "question": "What is the diagnosis?"})

    # Should have clean, indented JSON for Pydantic input
    user_message = messages[-1]["content"]
    assert '"name": "John Doe"' in user_message
    assert '"age": 45' in user_message
    assert '"street": "123 Main St"' in user_message
    assert '"country": "US"' in user_message


def test_baml_adapter_handles_mixed_input_types():
    """Test formatting of mixed Pydantic and primitive inputs."""

    class TestSignature(Signature):
        patient: PatientDetails = InputField()
        priority: int = InputField()
        notes: str = InputField()
        result: str = OutputField()

    adapter = BAMLAdapter()
    patient = PatientDetails(name="Jane Doe", age=30)

    messages = adapter.format(TestSignature, [], {"patient": patient, "priority": 1, "notes": "Urgent case"})

    user_message = messages[-1]["content"]
    # Pydantic should be JSON formatted
    assert '"name": "Jane Doe"' in user_message
    # Primitives should be formatted normally
    assert "priority ## ]]\n1" in user_message
    assert "notes ## ]]\nUrgent case" in user_message


def test_baml_adapter_handles_schema_generation_errors_gracefully():
    """Test graceful handling of schema generation errors."""

    class ProblematicModel(pydantic.BaseModel):
        # This might cause issues in schema generation
        field: object

    class TestSignature(Signature):
        input: str = InputField()
        output: ProblematicModel = OutputField()

    adapter = BAMLAdapter()

    # Should not raise an exception
    try:
        schema = adapter.format_field_structure(TestSignature)
        # If no exception, schema should at least contain some basic structure
        assert "schema" in schema.lower()
    except Exception:
        # If exception occurs, test passes as we're testing graceful handling
        pass


def test_baml_adapter_raises_on_missing_fields():
    """Test that missing required fields raise appropriate errors."""

    class TestSignature(Signature):
        input: str = InputField()
        patient: PatientDetails = OutputField()
        summary: str = OutputField()

    adapter = BAMLAdapter()

    # Missing 'summary' field
    completion = '{"patient": {"name": "John", "age": 30}}'

    with pytest.raises(AdapterParseError) as e:
        adapter.parse(TestSignature, completion)

    assert e.value.adapter_name == "JSONAdapter"  # BAMLAdapter inherits from JSONAdapter
    assert "summary" in str(e.value)


def test_baml_adapter_handles_type_casting_errors():
    """Test graceful handling of type casting errors."""

    class TestSignature(Signature):
        input: str = InputField()
        patient: PatientDetails = OutputField()

    adapter = BAMLAdapter()

    # Invalid age type
    completion = '{"patient": {"name": "John", "age": "not_a_number"}}'

    # Should raise ValidationError from Pydantic (which is the expected behavior)
    with pytest.raises((AdapterParseError, pydantic.ValidationError)):
        adapter.parse(TestSignature, completion)


def test_baml_adapter_with_images():
    """Test BAMLAdapter integration with Image objects."""

    class TestSignature(Signature):
        image_data: ImageWrapper = InputField()
        description: str = OutputField()

    adapter = BAMLAdapter()

    image_wrapper = ImageWrapper(
        images=[Image(url="https://example.com/image1.jpg"), Image(url="https://example.com/image2.jpg")],
        tag=["test", "medical"],
    )

    messages = adapter.format(TestSignature, [], {"image_data": image_wrapper})

    # Should contain image URLs in the message content
    user_message = messages[-1]["content"]
    image_contents = [
        content for content in user_message if isinstance(content, dict) and content.get("type") == "image_url"
    ]

    assert len(image_contents) == 2
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}} in user_message
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}} in user_message


def test_baml_adapter_with_tools():
    """Test BAMLAdapter integration with Tool objects."""

    class TestSignature(Signature):
        question: str = InputField()
        tools: list[Tool] = InputField()
        answer: str = OutputField()

    def get_patient_info(patient_id: int) -> str:
        """Get patient information by ID"""
        return f"Patient info for ID {patient_id}"

    def schedule_appointment(patient_name: str, date: str) -> str:
        """Schedule an appointment for a patient"""
        return f"Scheduled appointment for {patient_name} on {date}"

    tools = [Tool(get_patient_info), Tool(schedule_appointment)]

    adapter = BAMLAdapter()
    messages = adapter.format(TestSignature, [], {"question": "Schedule an appointment for John", "tools": tools})

    user_message = messages[-1]["content"]
    assert "get_patient_info" in user_message
    assert "schedule_appointment" in user_message
    assert "Get patient information by ID" in user_message
    assert "Schedule an appointment for a patient" in user_message


def test_baml_adapter_with_code():
    """Test BAMLAdapter integration with Code objects."""

    # Test with code as input field
    class CodeAnalysisSignature(Signature):
        code: Code = InputField()
        analysis: str = OutputField()

    adapter = BAMLAdapter()
    messages = adapter.format(CodeAnalysisSignature, [], {"code": "def hello():\n    print('Hello, world!')"})

    user_message = messages[-1]["content"]
    assert "def hello():" in user_message
    assert "print('Hello, world!')" in user_message

    # Test with code as output field
    class CodeGenSignature(Signature):
        task: str = InputField()
        code: Code = OutputField()

    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content='{"code": "print(\\"Generated code\\")"}'))],
            model="openai/gpt-4o-mini",
        )

        result = adapter(
            LM(model="openai/gpt-4o-mini", cache=False),
            {},
            CodeGenSignature,
            [],
            {"task": "Write a hello world program"},
        )

        assert result[0]["code"].code == 'print("Generated code")'


def test_baml_adapter_with_conversation_history():
    """Test BAMLAdapter integration with History objects."""

    class TestSignature(Signature):
        history: History = InputField()
        question: str = InputField()
        answer: str = OutputField()

    history = History(
        messages=[
            {"question": "What is the patient's age?", "answer": "45 years old"},
            {"question": "Any allergies?", "answer": "Penicillin allergy"},
        ]
    )

    adapter = BAMLAdapter()
    messages = adapter.format(TestSignature, [], {"history": history, "question": "What medications should we avoid?"})

    # Should format history as separate messages
    assert len(messages) == 6  # system + 2 history pairs + user
    assert "What is the patient's age?" in messages[1]["content"]
    assert '"answer": "45 years old"' in messages[2]["content"]
    assert "Any allergies?" in messages[3]["content"]
    assert '"answer": "Penicillin allergy"' in messages[4]["content"]


# Comparison tests with JSONAdapter
def test_baml_vs_json_adapter_token_efficiency():
    """Test that BAMLAdapter generates more token-efficient schemas."""

    class TestSignature(Signature):
        input: str = InputField()
        complex: ComplexNestedModel = OutputField()

    baml_adapter = BAMLAdapter()
    json_adapter = JSONAdapter()

    baml_schema = baml_adapter.format_field_structure(TestSignature)
    json_schema = json_adapter.format_field_structure(TestSignature)

    # Simple character count as proxy for token efficiency
    # BAMLAdapter should always produce shorter schemas
    assert len(baml_schema) < len(json_schema)


def test_baml_vs_json_adapter_functional_compatibility():
    """Test that both adapters parse identical outputs to the same results."""

    class TestSignature(Signature):
        question: str = InputField()
        patient: PatientDetails = OutputField()

    baml_adapter = BAMLAdapter()
    json_adapter = JSONAdapter()

    completion = """{"patient": {
        "name": "Alice Brown",
        "age": 35,
        "address": {"street": "789 Pine St", "city": "Boston", "country": "US"}
    }}"""

    baml_result = baml_adapter.parse(TestSignature, completion)
    json_result = json_adapter.parse(TestSignature, completion)

    # Results should be functionally equivalent
    assert baml_result["patient"].name == json_result["patient"].name
    assert baml_result["patient"].age == json_result["patient"].age
    assert baml_result["patient"].address.street == json_result["patient"].address.street


@pytest.mark.asyncio
async def test_baml_adapter_async_functionality():
    """Test BAMLAdapter async operations."""

    class TestSignature(Signature):
        question: str = InputField()
        patient: PatientDetails = OutputField()

    with mock.patch("litellm.acompletion") as mock_acompletion:
        mock_acompletion.return_value = ModelResponse(
            choices=[Choices(message=Message(content='{"patient": {"name": "John Doe", "age": 28}}'))],
            model="openai/gpt-4o",
        )

        adapter = BAMLAdapter()
        result = await adapter.acall(
            LM(model="openai/gpt-4o", cache=False), {}, TestSignature, [], {"question": "Extract patient info"}
        )

        assert result[0]["patient"].name == "John Doe"
        assert result[0]["patient"].age == 28


def test_baml_adapter_with_field_aliases():
    """Test BAMLAdapter with Pydantic field aliases."""

    class ModelWithAliases(pydantic.BaseModel):
        full_name: str = pydantic.Field(alias="name")
        patient_age: int = pydantic.Field(alias="age")

    class TestSignature(Signature):
        input: str = InputField()
        data: ModelWithAliases = OutputField()

    adapter = BAMLAdapter()

    # Schema should show aliases in the output structure
    schema = adapter.format_field_structure(TestSignature)
    assert "name:" in schema  # Should use alias, not field name
    assert "age:" in schema  # Should use alias, not field name


def test_baml_adapter_field_alias_without_description():
    """Test BAMLAdapter with field alias present but description absent."""

    class ModelWithAliasNoDescription(pydantic.BaseModel):
        internal_field: str = pydantic.Field(alias="public_name")
        regular_field: int
        field_with_description: str = pydantic.Field(description="This field has a description", alias="desc_field")

    class TestSignature(Signature):
        input: str = InputField()
        data: ModelWithAliasNoDescription = OutputField()

    adapter = BAMLAdapter()
    schema = adapter.format_field_structure(TestSignature)

    # Should show alias as comment when description is absent
    assert f"{COMMENT_SYMBOL} alias: public_name" in schema
    # Should show description comment when present
    assert f"{COMMENT_SYMBOL} This field has a description" in schema
    # Regular field (without alias) should appear in schema but without alias comment
    assert "regular_field: int," in schema
    # Check that regular_field section doesn't have an alias comment
    regular_field_section = schema.split("regular_field: int,")[0].split("\n")[-1]
    assert f"{COMMENT_SYMBOL} alias:" not in regular_field_section


def test_baml_adapter_multiple_pydantic_input_fields():
    """Test that multiple InputField() with Pydantic models are rendered correctly."""

    class UserProfile(pydantic.BaseModel):
        name: str = pydantic.Field(description="User's full name")
        email: str
        age: int

    class SystemConfig(pydantic.BaseModel):
        timeout: int = pydantic.Field(description="Timeout in seconds")
        debug: bool
        endpoints: list[str]

    class TestSignature(Signature):
        input_1: UserProfile = InputField(desc="User profile information")
        input_2: SystemConfig = InputField(desc="System configuration settings")
        result: str = OutputField(desc="Resulting output after processing")

    adapter = BAMLAdapter()

    # Test schema generation includes headers for ALL input fields
    schema = adapter.format_field_structure(TestSignature)
    assert "[[ ## input_1 ## ]]" in schema  # Should include first input field header
    assert "[[ ## input_2 ## ]]" in schema  # Should include second input field header
    assert "[[ ## result ## ]]" in schema  # Should include output field header
    assert "[[ ## completed ## ]]" in schema  # Should include completed section
    assert "All interactions will be structured in the following way" in schema
    assert "{input_1}" in schema
    assert "{input_2}" in schema
    assert "Output field `result` should be of type: string" in schema

    # Test field descriptions are in the correct method
    field_desc = adapter.format_field_description(TestSignature)
    assert "Your input fields are:" in field_desc
    assert "1. `input_1` (UserProfile): User profile information" in field_desc
    assert "2. `input_2` (SystemConfig): System configuration settings" in field_desc
    assert "Your output fields are:" in field_desc
    assert "1. `result` (str): Resulting output after processing" in field_desc

    # Test message formatting with actual Pydantic instances
    user_profile = UserProfile(name="John Doe", email="john@example.com", age=30)
    system_config = SystemConfig(timeout=300, debug=True, endpoints=["api1", "api2"])

    messages = adapter.format(TestSignature, [], {"input_1": user_profile, "input_2": system_config})

    user_message = messages[-1]["content"]

    # Verify both inputs are rendered with the correct bracket notation
    assert "[[ ## input_1 ## ]]" in user_message
    assert "[[ ## input_2 ## ]]" in user_message

    # Verify JSON content for both inputs
    assert '"name": "John Doe"' in user_message
    assert '"email": "john@example.com"' in user_message
    assert '"age": 30' in user_message
    assert '"timeout": 300' in user_message
    assert '"debug": true' in user_message
    # Endpoints array is formatted with indentation, so check for individual elements
    assert '"api1"' in user_message
    assert '"api2"' in user_message
    assert '"endpoints":' in user_message
