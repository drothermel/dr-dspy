from dspy.history.repl_history import REPLVariable
from dspy.primitives import SandboxSerializable, build_repl_variable
from dspy.primitives.sandbox_protocol import SandboxSerializablePydanticMixin
from dspy.task_spec import input_field, output_field


class ExampleSerializable(SandboxSerializablePydanticMixin):
    def __init__(self, data: str = "example_data"):
        self.data = data

    def sandbox_setup(self) -> str:
        return "import json"

    def to_sandbox(self) -> bytes:
        return self.data.encode("utf-8")

    def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
        return f"{var_name} = json.loads({data_expr})"

    def rlm_preview(self, max_chars: int = 500) -> str:
        preview = f"ExampleData: {self.data}"
        return preview[:max_chars] + "..." if len(preview) > max_chars else preview


class IncompleteSerializable:
    def sandbox_setup(self) -> str:
        return ""


class NotASubclass:
    def sandbox_setup(self) -> str:
        return "import json"

    def to_sandbox(self) -> bytes:
        return b""

    def sandbox_assignment(self, var_name: str, data_expr: str) -> str:
        return f"{var_name} = {data_expr}"

    def rlm_preview(self, max_chars: int = 500) -> str:
        return "NotASubclass"


class TestProtocolConformance:
    def test_subclass_conformance(self):
        assert isinstance(ExampleSerializable(), SandboxSerializable)

    def test_incomplete_class_can_instantiate(self):
        obj = IncompleteSerializable()
        assert not isinstance(obj, SandboxSerializable)

    def test_structural_conformance_accepted(self):
        assert isinstance(NotASubclass(), SandboxSerializable)
        assert not isinstance("hello", SandboxSerializable)


class TestCoreMethods:
    def test_sandbox_setup(self):
        assert ExampleSerializable().sandbox_setup() == "import json"

    def test_to_sandbox_returns_bytes(self):
        payload = ExampleSerializable("hello").to_sandbox()
        assert isinstance(payload, bytes)
        assert payload == b"hello"

    def test_sandbox_assignment(self):
        code = ExampleSerializable().sandbox_assignment("my_var", "raw_data")
        assert "my_var" in code
        assert "raw_data" in code
        assert "json.loads" in code

    def test_rlm_preview(self):
        preview = ExampleSerializable("test_value").rlm_preview()
        assert "ExampleData" in preview
        assert "test_value" in preview

    def test_rlm_preview_truncation(self):
        preview = ExampleSerializable("x" * 1000).rlm_preview(max_chars=50)
        assert len(preview) <= 53
        assert preview.endswith("...")


class TestBuildReplVariable:
    def test_builds_variable_with_preview_from_rlm_preview(self):
        obj = ExampleSerializable("my_data")
        var = build_repl_variable(obj, "my_var")
        assert isinstance(var, REPLVariable)
        assert var.name == "my_var"
        assert var.preview == obj.rlm_preview()
        assert var.total_length == len(obj.rlm_preview())

    def test_surfaces_sandbox_setup_in_description(self):
        var = build_repl_variable(ExampleSerializable(), "x")
        assert "import json" in var.desc

    def test_passes_field_info_through(self):
        from dspy.task_spec import make_task_spec

        spec = make_task_spec(
            {
                "data": input_field("data", type_=ExampleSerializable, desc="A data column"),
                "answer": output_field("answer", desc="The answer."),
            },
            instructions="Process data.",
        )
        field = spec.input_fields["data"]
        var = build_repl_variable(ExampleSerializable(), "data", field=field)
        assert "A data column" in var.desc
        assert "import json" in var.desc
