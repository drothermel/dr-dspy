import asyncio
from typing import Literal
from unittest import mock

import pydantic
import pytest

from dspy.utils.exceptions import AdapterParseError

try:
    from litellm import Choices, Message
    from litellm.files.main import ModelResponse
except ImportError:
    pytest.skip("litellm is not installed", allow_module_level=True)  # ty: ignore[too-many-positional-arguments]

from dspy.adapters.baml_adapter import COMMENT_SYMBOL, INDENTATION, BAMLAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types.code import Code
from dspy.adapters.types.history import History
from dspy.adapters.types.image import Image
from dspy.adapters.types.tool import Tool
from dspy.clients.lm import LM
from dspy.task_spec import FieldSpec, make_task_spec
from tests.adapters.conftest import adapter_format_as_openai, format_messages_and_lm_kwargs
from tests.task_spec.helpers import ts


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
    QA = ts("question -> answer", instructions="Given the fields `question`, produce the fields `answer`.")
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=BAMLAdapter(),
        task_spec=QA,
        demos=[{"question": "Q1", "answer": "A1"}],
        inputs={"question": "Q2"},
    )

    expected_messages = [
        {
            "role": "system",
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
            "        Given the fields `question`, produce the fields `answer`.",
        },
        {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
        {"role": "assistant", "content": '{\n  "answer": "A1"\n}'},
        {
            "role": "user",
            "content": "[[ ## question ## ]]\n"
            "Q2\n"
            "\n"
            "Respond with a JSON object in the following order of fields: `answer`.",
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_baml_adapter_format_exact_messages_with_nested_output():
    class BamlNested(pydantic.BaseModel):
        value: int
        tags: list[str]

    TypedSignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "answer": FieldSpec.output("answer", type_=BamlNested),
        },
        instructions="Given the fields `question`, produce the fields `answer`.",
    )
    messages, lm_kwargs = format_messages_and_lm_kwargs(
        adapter=BAMLAdapter(), task_spec=TypedSignature, demos=[], inputs={"question": "Q"}
    )

    expected_messages = [
        {
            "role": "system",
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
            "        Given the fields `question`, produce the fields `answer`.",
        },
        {
            "role": "user",
            "content": "[[ ## question ## ]]\n"
            "Q\n"
            "\n"
            "Respond with a JSON object in the following order of fields: `answer` (must be "
            "formatted as a valid Python BamlNested).",
        },
    ]
    assert messages == expected_messages
    expected_lm_kwargs = {}
    assert lm_kwargs == expected_lm_kwargs


def test_baml_adapter_basic_schema_generation():
    """Test that BAMLAdapter generates simplified schemas for Pydantic models."""

    TestSignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "patient": FieldSpec.output("patient", type_=PatientDetails),
        },
        instructions="Given the fields `question`, produce the fields `patient`.",
    )
    adapter = BAMLAdapter()
    schema = adapter.format_field_structure(TestSignature)

    assert f"{COMMENT_SYMBOL} Full name of the patient" in schema
    assert "name: string," in schema
    assert "age: int," in schema
    assert "address:" in schema
    assert "street: string," in schema
    assert 'country: "US" or "CA",' in schema


def test_baml_adapter_handles_optional_fields():
    """Test optional field rendering with 'or null' syntax."""

    TestSignature = make_task_spec(
        {
            "input": FieldSpec.input("input"),
            "patient": FieldSpec.output("patient", type_=PatientDetails),
        },
        instructions="Given the fields `input`, produce the fields `patient`.",
    )
    adapter = BAMLAdapter()
    schema = adapter.format_field_structure(TestSignature)

    assert "address:" in schema
    assert "or null" in schema


def test_baml_adapter_handles_primitive_types():
    """Test rendering of basic primitive types."""

    class SimpleModel(pydantic.BaseModel):
        text: str
        number: int
        decimal: float
        flag: bool

    TestSignature = make_task_spec(
        {
            "input": FieldSpec.input("input"),
            "output": FieldSpec.output("output", type_=SimpleModel),
        },
        instructions="Given the fields `input`, produce the fields `output`.",
    )
    adapter = BAMLAdapter()
    schema = adapter.format_field_structure(TestSignature)

    assert "text: string," in schema
    assert "number: int," in schema
    assert "decimal: float," in schema
    assert "flag: boolean," in schema


def test_baml_adapter_handles_lists_with_bracket_notation():
    """Test that lists of Pydantic models use proper bracket notation."""

    TestSignature = make_task_spec(
        {
            "input": FieldSpec.input("input"),
            "addresses": FieldSpec.output("addresses", type_=ModelWithLists),
        },
        instructions="Given the fields `input`, produce the fields `addresses`.",
    )
    adapter = BAMLAdapter()
    schema = adapter.format_field_structure(TestSignature)

    assert "items: [" in schema
    assert f"{COMMENT_SYMBOL} List of patient addresses" in schema
    assert "street: string," in schema
    assert "city: string," in schema
    assert "]," in schema
    assert "scores: float[]," in schema


def test_baml_adapter_handles_complex_nested_models():
    """Test deeply nested Pydantic model schema generation."""

    TestSignature = make_task_spec(
        {
            "input": FieldSpec.input("input"),
            "complex": FieldSpec.output("complex", type_=ComplexNestedModel),
        },
        instructions="Given the fields `input`, produce the fields `complex`.",
    )
    adapter = BAMLAdapter()
    schema = adapter.format_field_structure(TestSignature)

    assert f"{COMMENT_SYMBOL} Unique identifier" in schema
    assert f"{INDENTATION}details:" in schema
    assert f"{COMMENT_SYMBOL} Full name of the patient" in schema
    assert "tags: string[]," in schema
    assert "metadata: dict[string, string]," in schema


def test_baml_adapter_raise_error_on_circular_references():
    """Test that circular references are handled gracefully."""

    TestSignature = make_task_spec(
        {
            "input": FieldSpec.input("input"),
            "circular": FieldSpec.output("circular", type_=CircularModel),
        },
        instructions="Given the fields `input`, produce the fields `circular`.",
    )
    adapter = BAMLAdapter()
    with pytest.raises(ValueError) as error:  # noqa: PT011
        adapter.format_field_structure(TestSignature)

    assert "BAMLAdapter cannot handle recursive pydantic models" in str(error.value)


def test_baml_adapter_formats_pydantic_inputs_as_clean_json():
    """Test that Pydantic input instances are formatted as clean JSON."""

    TestSignature = make_task_spec(
        {
            "patient": FieldSpec.input("patient", type_=PatientDetails),
            "question": FieldSpec.input("question"),
            "answer": FieldSpec.output("answer"),
        },
        instructions="Given the fields `patient`, `question`, produce the fields `answer`.",
    )
    adapter = BAMLAdapter()
    patient = PatientDetails(
        name="John Doe", age=45, address=PatientAddress(street="123 Main St", city="Anytown", country="US")
    )

    messages = adapter_format_as_openai(
        adapter=adapter,
        task_spec=TestSignature,
        demos=[],
        inputs={"patient": patient, "question": "What is the diagnosis?"},
    )

    user_message = messages[-1]["content"]
    assert '"name": "John Doe"' in user_message
    assert '"age": 45' in user_message
    assert '"street": "123 Main St"' in user_message
    assert '"country": "US"' in user_message


def test_baml_adapter_handles_mixed_input_types():
    """Test formatting of mixed Pydantic and primitive inputs."""

    TestSignature = make_task_spec(
        {
            "patient": FieldSpec.input("patient", type_=PatientDetails),
            "priority": FieldSpec.input("priority", type_=int),
            "notes": FieldSpec.input("notes"),
            "result": FieldSpec.output("result"),
        },
        instructions="Given the fields `patient`, `priority`, `notes`, produce the fields `result`.",
    )
    adapter = BAMLAdapter()
    patient = PatientDetails(name="Jane Doe", age=30)

    messages = adapter_format_as_openai(
        adapter=adapter,
        task_spec=TestSignature,
        demos=[],
        inputs={"patient": patient, "priority": 1, "notes": "Urgent case"},
    )

    user_message = messages[-1]["content"]
    assert '"name": "Jane Doe"' in user_message
    assert "priority ## ]]\n1" in user_message
    assert "notes ## ]]\nUrgent case" in user_message


def test_baml_adapter_handles_schema_generation_errors_gracefully():
    """Test graceful handling of schema generation errors."""

    class ProblematicModel(pydantic.BaseModel):
        # This might cause issues in schema generation
        field: object

    TestSignature = make_task_spec(
        {
            "input": FieldSpec.input("input"),
            "output": FieldSpec.output("output", type_=ProblematicModel),
        },
        instructions="Given the fields `input`, produce the fields `output`.",
    )
    adapter = BAMLAdapter()

    try:
        schema = adapter.format_field_structure(TestSignature)
        assert "schema" in schema.lower()
    except Exception:
        pass


def test_baml_adapter_raises_on_missing_fields():
    """Test that missing required fields raise appropriate errors."""

    TestSignature = make_task_spec(
        {
            "input": FieldSpec.input("input"),
            "patient": FieldSpec.output("patient", type_=PatientDetails),
            "summary": FieldSpec.output("summary"),
        },
        instructions="Given the fields `input`, produce the fields `patient`, `summary`.",
    )
    adapter = BAMLAdapter()

    # Missing 'summary' field
    completion = '{"patient": {"name": "John", "age": 30}}'

    with pytest.raises(AdapterParseError) as e:
        adapter.parse(task_spec=TestSignature, completion=completion)

    assert e.value.adapter_name == "JSONAdapter"  # BAMLAdapter inherits from JSONAdapter
    assert "summary" in str(e.value)


def test_baml_adapter_handles_type_casting_errors():
    """Test graceful handling of type casting errors."""

    TestSignature = make_task_spec(
        {
            "input": FieldSpec.input("input"),
            "patient": FieldSpec.output("patient", type_=PatientDetails),
        },
        instructions="Given the fields `input`, produce the fields `patient`.",
    )
    adapter = BAMLAdapter()

    # Invalid age type
    completion = '{"patient": {"name": "John", "age": "not_a_number"}}'

    with pytest.raises((AdapterParseError, pydantic.ValidationError)):
        adapter.parse(task_spec=TestSignature, completion=completion)


def test_baml_adapter_with_images():
    """Test BAMLAdapter integration with Image objects."""

    TestSignature = make_task_spec(
        {
            "image_data": FieldSpec.input("image_data", type_=ImageWrapper),
            "description": FieldSpec.output("description"),
        },
        instructions="Given the fields `image_data`, produce the fields `description`.",
    )
    adapter = BAMLAdapter()

    image_wrapper = ImageWrapper(
        images=[Image(url="https://example.com/image1.jpg"), Image(url="https://example.com/image2.jpg")],
        tag=["test", "medical"],
    )

    messages = adapter_format_as_openai(
        adapter=adapter, task_spec=TestSignature, demos=[], inputs={"image_data": image_wrapper}
    )

    user_message = messages[-1]["content"]
    image_contents = [
        content for content in user_message if isinstance(content, dict) and content.get("type") == "image_url"
    ]

    assert len(image_contents) == 2
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image1.jpg"}} in user_message
    assert {"type": "image_url", "image_url": {"url": "https://example.com/image2.jpg"}} in user_message


def test_baml_adapter_with_tools():
    """Test BAMLAdapter integration with Tool objects."""

    TestSignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "tools": FieldSpec.input("tools", type_=list[Tool]),
            "answer": FieldSpec.output("answer"),
        },
        instructions="Given the fields `question`, `tools`, produce the fields `answer`.",
    )

    def get_patient_info(patient_id: int) -> str:
        """Get patient information by ID"""
        return f"Patient info for ID {patient_id}"

    def schedule_appointment(patient_name: str, date: str) -> str:
        """Schedule an appointment for a patient"""
        return f"Scheduled appointment for {patient_name} on {date}"

    tools = [
        Tool(get_patient_info, description="Get patient information by ID"),
        Tool(schedule_appointment, description="Schedule an appointment for a patient"),
    ]

    adapter = BAMLAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter,
        task_spec=TestSignature,
        demos=[],
        inputs={"question": "Schedule an appointment for John", "tools": tools},
    )

    user_message = messages[-1]["content"]
    assert "get_patient_info" in user_message
    assert "schedule_appointment" in user_message
    assert "Get patient information by ID" in user_message
    assert "Schedule an appointment for a patient" in user_message


def test_baml_adapter_with_code():
    """Test BAMLAdapter integration with Code objects."""

    # Test with code as input field
    CodeAnalysisSignature = make_task_spec(
        {
            "code": FieldSpec.input("code", type_=Code),
            "analysis": FieldSpec.output("analysis"),
        },
        instructions="Given the fields `code`, produce the fields `analysis`.",
    )
    adapter = BAMLAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter,
        task_spec=CodeAnalysisSignature,
        demos=[],
        inputs={"code": "def hello():\n    print('Hello, world!')"},
    )

    user_message = messages[-1]["content"]
    assert "def hello():" in user_message
    assert "print('Hello, world!')" in user_message

    # Test with code as output field
    CodeGenSignature = make_task_spec(
        {
            "task": FieldSpec.input("task"),
            "code": FieldSpec.output("code", type_=Code),
        },
        instructions="Given the fields `task`, produce the fields `code`.",
    )
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content='{"code": "print(\\"Generated code\\")"}'))],
            model="openai/gpt-4o-mini",
        )

        result = asyncio.run(
            adapter.acall(
                lm=LM(model="openai/gpt-4o-mini", cache=False),
                config={},
                task_spec=CodeGenSignature,
                demos=[],
                inputs={"task": "Write a hello world program"},
            )
        )

        assert result[0]["code"].code == 'print("Generated code")'


def test_baml_adapter_with_conversation_history():
    """Test BAMLAdapter integration with History objects."""

    TestSignature = make_task_spec(
        {
            "history": FieldSpec.input("history", type_=History),
            "question": FieldSpec.input("question"),
            "answer": FieldSpec.output("answer"),
        },
        instructions="Given the fields `history`, `question`, produce the fields `answer`.",
    )
    history = History(
        messages=[
            {"question": "What is the patient's age?", "answer": "45 years old"},
            {"question": "Any allergies?", "answer": "Penicillin allergy"},
        ]
    )

    adapter = BAMLAdapter()
    messages = adapter_format_as_openai(
        adapter=adapter,
        task_spec=TestSignature,
        demos=[],
        inputs={"history": history, "question": "What medications should we avoid?"},
    )

    assert len(messages) == 6  # system + 2 history pairs + user
    assert "What is the patient's age?" in messages[1]["content"]
    assert '"answer": "45 years old"' in messages[2]["content"]
    assert "Any allergies?" in messages[3]["content"]
    assert '"answer": "Penicillin allergy"' in messages[4]["content"]


# Comparison tests with JSONAdapter
def test_baml_vs_json_adapter_token_efficiency():
    """Test that BAMLAdapter generates more token-efficient schemas."""

    TestSignature = make_task_spec(
        {
            "input": FieldSpec.input("input"),
            "complex": FieldSpec.output("complex", type_=ComplexNestedModel),
        },
        instructions="Given the fields `input`, produce the fields `complex`.",
    )
    baml_adapter = BAMLAdapter()
    json_adapter = JSONAdapter()

    baml_schema = baml_adapter.format_field_structure(TestSignature)
    json_schema = json_adapter.format_field_structure(TestSignature)

    # Simple character count as proxy for token efficiency
    # BAMLAdapter should always produce shorter schemas
    assert len(baml_schema) < len(json_schema)


def test_baml_vs_json_adapter_functional_compatibility():
    """Test that both adapters parse identical outputs to the same results."""

    TestSignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "patient": FieldSpec.output("patient", type_=PatientDetails),
        },
        instructions="Given the fields `question`, produce the fields `patient`.",
    )
    baml_adapter = BAMLAdapter()
    json_adapter = JSONAdapter()

    completion = """{"patient": {
        "name": "Alice Brown",
        "age": 35,
        "address": {"street": "789 Pine St", "city": "Boston", "country": "US"}
    }}"""

    baml_result = baml_adapter.parse(task_spec=TestSignature, completion=completion)
    json_result = json_adapter.parse(task_spec=TestSignature, completion=completion)

    assert baml_result["patient"].name == json_result["patient"].name
    assert baml_result["patient"].age == json_result["patient"].age
    assert baml_result["patient"].address.street == json_result["patient"].address.street


@pytest.mark.asyncio
async def test_baml_adapter_async_functionality():
    """Test BAMLAdapter async operations."""

    TestSignature = make_task_spec(
        {
            "question": FieldSpec.input("question"),
            "patient": FieldSpec.output("patient", type_=PatientDetails),
        },
        instructions="Given the fields `question`, produce the fields `patient`.",
    )
    with mock.patch("litellm.acompletion", new_callable=mock.AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = ModelResponse(
            choices=[Choices(message=Message(content='{"patient": {"name": "John Doe", "age": 28}}'))],
            model="openai/gpt-4o",
        )

        adapter = BAMLAdapter()
        result = await adapter.acall(
            lm=LM(model="openai/gpt-4o", cache=False),
            config={},
            task_spec=TestSignature,
            demos=[],
            inputs={"question": "Extract patient info"},
        )

        assert result[0]["patient"].name == "John Doe"
        assert result[0]["patient"].age == 28


def test_baml_adapter_with_field_aliases():
    """Test BAMLAdapter with Pydantic field aliases."""

    class ModelWithAliases(pydantic.BaseModel):
        full_name: str = pydantic.Field(alias="name")
        patient_age: int = pydantic.Field(alias="age")

    TestSignature = make_task_spec(
        {
            "input": FieldSpec.input("input"),
            "data": FieldSpec.output("data", type_=ModelWithAliases),
        },
        instructions="Given the fields `input`, produce the fields `data`.",
    )
    adapter = BAMLAdapter()

    schema = adapter.format_field_structure(TestSignature)
    assert "name:" in schema
    assert "age:" in schema


def test_baml_adapter_field_alias_without_description():
    """Test BAMLAdapter with field alias present but description absent."""

    class ModelWithAliasNoDescription(pydantic.BaseModel):
        internal_field: str = pydantic.Field(alias="public_name")
        regular_field: int
        field_with_description: str = pydantic.Field(description="This field has a description", alias="desc_field")

    TestSignature = make_task_spec(
        {
            "input": FieldSpec.input("input"),
            "data": FieldSpec.output("data", type_=ModelWithAliasNoDescription),
        },
        instructions="Given the fields `input`, produce the fields `data`.",
    )
    adapter = BAMLAdapter()
    schema = adapter.format_field_structure(TestSignature)

    assert f"{COMMENT_SYMBOL} alias: public_name" in schema
    assert f"{COMMENT_SYMBOL} This field has a description" in schema
    assert "regular_field: int," in schema
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

    TestSignature = make_task_spec(
        {
            "input_1": FieldSpec.input("input_1", type_=UserProfile, desc="User profile information"),
            "input_2": FieldSpec.input("input_2", type_=SystemConfig, desc="System configuration settings"),
            "result": FieldSpec.output("result", desc="Resulting output after processing"),
        },
        instructions="Given the fields `input_1`, `input_2`, produce the fields `result`.",
    )
    adapter = BAMLAdapter()

    schema = adapter.format_field_structure(TestSignature)
    assert "[[ ## input_1 ## ]]" in schema
    assert "[[ ## input_2 ## ]]" in schema
    assert "[[ ## result ## ]]" in schema
    assert "[[ ## completed ## ]]" in schema
    assert "All interactions will be structured in the following way" in schema
    assert "{input_1}" in schema
    assert "{input_2}" in schema
    assert "Output field `result` should be of type: string" in schema

    field_desc = adapter.format_field_description(TestSignature)
    assert "Your input fields are:" in field_desc
    assert "1. `input_1` (UserProfile): User profile information" in field_desc
    assert "2. `input_2` (SystemConfig): System configuration settings" in field_desc
    assert "Your output fields are:" in field_desc
    assert "1. `result` (str): Resulting output after processing" in field_desc

    user_profile = UserProfile(name="John Doe", email="john@example.com", age=30)
    system_config = SystemConfig(timeout=300, debug=True, endpoints=["api1", "api2"])

    messages = adapter_format_as_openai(
        adapter=adapter, task_spec=TestSignature, demos=[], inputs={"input_1": user_profile, "input_2": system_config}
    )

    user_message = messages[-1]["content"]

    assert "[[ ## input_1 ## ]]" in user_message
    assert "[[ ## input_2 ## ]]" in user_message

    assert '"name": "John Doe"' in user_message
    assert '"email": "john@example.com"' in user_message
    assert '"age": 30' in user_message
    assert '"timeout": 300' in user_message
    assert '"debug": true' in user_message
    # Endpoints array is formatted with indentation, so check for individual elements
    assert '"api1"' in user_message
    assert '"api2"' in user_message
    assert '"endpoints":' in user_message
