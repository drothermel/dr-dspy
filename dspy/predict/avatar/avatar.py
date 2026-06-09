from typing import Any, cast

from dspy.adapters.types.tool import Tool
from dspy.core.types.call_options import ModuleCallOptions
from dspy.history import TurnEvent, TurnLog, call_with_turn_log_truncation
from dspy.predict.avatar.models import Action, ActionOutput
from dspy.predict.predict import Predict
from dspy.primitives import Module, Prediction
from dspy.runtime.run_context import RunContext
from dspy.task_spec import FieldSpec, TaskSpec, input_field, make_task_spec, output_field


class Avatar(Module):
    def __init__(
        self,
        task_spec: TaskSpec,
        tools: list[Tool],
        max_iters: int = 3,
        verbose: bool = False,
    ) -> None:
        if not isinstance(task_spec, TaskSpec):
            raise TypeError(f"Avatar requires a TaskSpec instance, got {type(task_spec).__name__}.")
        super().__init__()
        self.task_spec = task_spec
        self.input_fields = task_spec.input_fields
        self.output_fields = task_spec.output_fields
        tools_by_name: dict[str, Tool] = {}
        for tool in tools:
            if not isinstance(tool, Tool):
                raise TypeError(
                    "tools must be Tool instances with an explicit description. Use Tool(func, description='...')."
                )
            if tool.name is None:
                raise ValueError("Tool name could not be determined.")
            tools_by_name[tool.name] = tool
        outputs = ", ".join([f"`{k}`" for k in task_spec.output_fields])
        tools_by_name["Finish"] = Tool(
            func=lambda: "Completed.",
            description=f"Marks the task as complete when all information for producing {outputs} is available.",
            name="Finish",
            args={},
        )
        self.tools = list(tools_by_name.values())
        self.tools_by_name = tools_by_name
        self.verbose = verbose
        self.max_iters = max_iters
        actor_fields: dict[str, FieldSpec] = {
            "goal": input_field("goal", str, desc="Task to be accomplished."),
            "tools": input_field("tools", list[str], desc="list of tools to use"),
            "turn_log": input_field("turn_log", TurnLog, desc="Previous actions and tool results."),
        }
        for field_name, field in self.input_fields.items():
            actor_fields[field_name] = input_field(field_name, field.type_, desc=field.desc, prefix=field.prefix)
        actor_fields["action"] = output_field(
            "action",
            Action,
            desc="Next action to take, including tool_name and tool_args for the selected tool.",
        )
        actor_instructions = (
            "You will be given `Tools` which will be a list of tools to use to accomplish the `Goal`. "
            "Given the user query, your task is to decide which tool to use and what input values to provide.\n\n"
            "You will output the action needed to accomplish the `Goal`. `Action` should have a tool to use "
            "and JSON tool_args to pass to the tool.\n\n"
            "Note: You can opt to use no tools and provide the final answer directly. You can also use one tool "
            "multiple times with different tool_args if applicable."
        )
        self.actor = Predict(make_task_spec(actor_fields, instructions=actor_instructions, name="Actor"))
        finish_fields = dict(actor_fields)
        finish_fields.pop("action")
        for field_name, field in self.output_fields.items():
            finish_fields[field_name] = output_field(field_name, field.type_, desc=field.desc, prefix=field.prefix)
        self.finish = Predict(
            make_task_spec(
                finish_fields,
                instructions=f"{task_spec.instructions}\n\nProduce the final outputs using the turn log.",
                name="AvatarFinish",
            )
        )

    async def _acall_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str | None:
        tool = self.tools_by_name.get(tool_name)
        if tool is None:
            return None
        result = await tool.acall(**tool_args)
        if result is None:
            return None
        return str(result)

    async def _aforward_impl(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **inputs,
    ):
        args = {
            "goal": self.task_spec.instructions,
            "tools": [tool.name for tool in self.tools],
            "turn_log": TurnLog.empty(),
        }
        for key in self.input_fields:
            if key in inputs:
                args[key] = inputs[key]
        turn_log = TurnLog.empty()
        actor_inputs = {key: value for key, value in args.items() if key != "turn_log"}
        action_results: list[ActionOutput] = []
        max_iters = cast("int", inputs.get("max_iters", self.max_iters))
        remaining = max_iters
        while remaining > 0:
            extracted = await call_with_turn_log_truncation(
                self.actor,
                turn_log=turn_log,
                run=run,
                options=options,
                max_attempts=3,
                **actor_inputs,
            )
            turn_log = extracted.turn_log
            actor_output = extracted.result
            action = actor_output.action
            tool_name = action.tool_name
            tool_args = action.tool_args
            if tool_name == "Finish":
                turn_log = turn_log.append_turn(
                    TurnEvent(
                        action=action,
                        result="Gathered all information needed to finish the task.",
                    )
                )
                break
            tool_output = await self._acall_tool(tool_name, tool_args)
            action_results.append(ActionOutput(tool_name=tool_name, tool_args=tool_args, tool_output=tool_output))
            turn_log = turn_log.append_turn(
                TurnEvent(action=action, result=tool_output if tool_output is not None else "")
            )
            remaining -= 1
        extracted = await call_with_turn_log_truncation(
            self.finish,
            turn_log=turn_log,
            run=run,
            options=options,
            max_attempts=3,
            **actor_inputs,
        )
        final_answer = extracted.result
        turn_log = extracted.turn_log
        return Prediction(
            **{key: getattr(final_answer, key) for key in self.output_fields},
            turn_log=turn_log,
            actions=action_results,
        )
