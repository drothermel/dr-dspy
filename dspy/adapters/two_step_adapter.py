from typing import Any

from typing_extensions import override

from dspy.adapters.base import Adapter
from dspy.adapters.call.capabilities import AdapterCapabilities
from dspy.adapters.call.mode import AdapterCallMode
from dspy.adapters.call.pipeline import AdapterCallPipeline
from dspy.adapters.call.postprocess import strip_native_response_output_fields
from dspy.clients.base_lm import BaseLM
from dspy.core.types.config import LMConfig
from dspy.errors import AdapterOperationError
from dspy.runtime.config import CallSite
from dspy.runtime.run_context import RunContext
from dspy.runtime.transparency import resolve_adapter, resolve_lm_config
from dspy.task_spec import TaskSpec, input_field, make_task_spec
from dspy.task_spec.formatting import get_field_spec_description_string


class TwoStepAdapter(Adapter):
    """Two-step adapter: main LM produces free text; extraction adapter parses fields.

    ``parse`` is not a supported entrypoint — use ``AdapterCallPipeline.execute`` via
    ``await adapter(...)`` / ``TwoStepAdapter.__call__``.
    """

    call_mode = AdapterCallMode.TWO_STEP
    capabilities = AdapterCapabilities(
        supports_finetune=False,
        field_value_role="none",
        default_native_fc=False,
        supports_structured_output=False,
    )

    def __init__(self, extraction_model: BaseLM, extraction_adapter: Adapter | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(extraction_model, BaseLM):
            raise ValueError("extraction_model must be an instance of dspy.clients.base_lm.BaseLM")
        self.extraction_model = extraction_model
        self.extraction_adapter = extraction_adapter

    @override
    def parse(self, task_spec: TaskSpec, completion: str) -> dict[str, Any]:
        _ = (task_spec, completion)
        raise AdapterOperationError(
            "TwoStepAdapter.parse is not supported. Use AdapterCallPipeline.execute via TwoStepAdapter.__call__."
        )

    async def _run_extraction(self, *, original_task_spec: TaskSpec, text: str, run: RunContext) -> dict[str, Any]:
        extraction_adapter, _adapter_notes = resolve_adapter(self.extraction_adapter or run.adapter)
        extractor_task_spec = self._create_extractor_task_spec(original_task_spec)
        config, _provenance = resolve_lm_config(self.extraction_model, LMConfig())
        extraction_site = CallSite(
            module="TwoStepAdapter",
            phase="two_step.extraction",
            lm_role="extraction_model",
        )
        results = await AdapterCallPipeline.execute(
            extraction_adapter,
            lm=self.extraction_model,
            config=config,
            task_spec=extractor_task_spec,
            demos=[],
            inputs={"text": text},
            run=run,
            call_site=extraction_site,
        )
        return results[0]

    @override
    def format_system_message(self, task_spec: TaskSpec) -> str:
        parts = []
        parts.append("You are a helpful assistant that can solve tasks based on user input.")
        parts.append(
            "As input, you will be provided with:\n" + get_field_spec_description_string(task_spec.input_fields)
        )
        parts.append("Your outputs must contain:\n" + get_field_spec_description_string(task_spec.output_fields))
        parts.append("You should lay out your outputs in detail so that your answer can be understood by another agent")
        if task_spec.instructions:
            parts.append(f"Specific instructions: {task_spec.instructions}")
        return "\n".join(parts)

    @override
    def format_field_description(self, task_spec: TaskSpec) -> str:
        return ""

    @override
    def format_field_structure(self, task_spec: TaskSpec) -> str:
        return ""

    @override
    def format_task_description(self, task_spec: TaskSpec) -> str:
        return ""

    @override
    def format_user_message_content(
        self,
        task_spec: TaskSpec,
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        _ = main_request
        parts = [prefix]
        parts.extend(f"{name}: {inputs.get(name, '')}" for name in task_spec.input_fields if name in inputs)
        parts.append(suffix)
        return "\n\n".join(parts).strip()

    @override
    def format_assistant_message_content(
        self, task_spec: TaskSpec, outputs: dict[str, Any], missing_field_message: str | None = None
    ) -> str:
        parts = [
            f"{name}: {outputs.get(name, missing_field_message)}" for name in task_spec.output_fields if name in outputs
        ]
        return "\n\n".join(parts).strip()

    def _create_extractor_task_spec(self, original_task_spec: TaskSpec) -> TaskSpec:
        extractable_spec = strip_native_response_output_fields(original_task_spec, self.native_response_types)
        new_fields = {
            "text": input_field(
                "text", str, desc="Raw completion text from the main language model to extract structured fields from."
            ),
            **dict(extractable_spec.output_fields),
        }
        outputs_str = ", ".join(f"`{field}`" for field in extractable_spec.output_fields)
        instructions = f"The input is a text that should contain all the necessary information to produce the fields {outputs_str}. Your job is to extract the fields from the text verbatim. Extract precisely the appropriate value (content) for each field."
        return make_task_spec(new_fields, instructions=instructions, name="framework.two_step.extractor")
