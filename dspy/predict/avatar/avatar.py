import logging
from typing import Any, cast

from dspy.adapters.types.tool import Tool
from dspy.history import AvatarTurnEvent, TruncationExhaustedError, TurnLog, call_with_history_truncation
from dspy.predict.agent_constants import AVATAR_TERMINAL_TOOL
from dspy.predict.agent_loop import AgentLoopControl, AgentLoopRunner, AgentStepResult
from dspy.predict.agent_termination import AgentTerminationReason
from dspy.predict.avatar.models import Action, ActionOutput
from dspy.predict.predict import Predict
from dspy.predict.tools import normalize_tools
from dspy.primitives import Module, Prediction
from dspy.runtime.call_options import ModuleCallOptions
from dspy.runtime.run_context import RunContext
from dspy.task_spec import FieldSpec, TaskSpec, input_field, make_task_spec, output_field

logger = logging.getLogger(__name__)


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
        tools_by_name = normalize_tools(tools)
        outputs = ", ".join([f"`{k}`" for k in task_spec.output_fields])
        tools_by_name[AVATAR_TERMINAL_TOOL] = Tool(
            func=lambda: "Completed.",
            description=f"Marks the task as complete when all information for producing {outputs} is available.",
            name=AVATAR_TERMINAL_TOOL,
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
            "turn_log": TurnLog.model_validate(inputs.pop("turn_log", TurnLog.empty())),
        }
        for key in self.input_fields:
            if key in inputs:
                args[key] = inputs[key]
        turn_log = args["turn_log"]
        actor_inputs = {key: value for key, value in args.items() if key != "turn_log"}
        action_results: list[ActionOutput] = []
        max_iters = cast("int", inputs.get("max_iters", self.max_iters))

        async def step(_turn_index: int, turn_log: TurnLog) -> AgentStepResult[TurnLog]:
            try:
                extracted = await call_with_history_truncation(
                    self.actor,
                    turn_log=turn_log,
                    run=run,
                    options=options,
                    max_attempts=3,
                    **actor_inputs,
                )
            except TruncationExhaustedError as err:
                logger.warning("Ending Avatar loop after context window exceeded: %s", err)
                return AgentStepResult(
                    history=turn_log,
                    control=AgentLoopControl.BREAK,
                    termination_reason=AgentTerminationReason.CONTEXT_WINDOW_EXCEEDED,
                )
            turn_log = extracted.turn_log
            actor_output = extracted.result
            action = actor_output.action
            tool_name = action.tool_name
            tool_args = action.tool_args
            if tool_name == AVATAR_TERMINAL_TOOL:
                return AgentStepResult(
                    history=turn_log.append_turn(
                        AvatarTurnEvent(
                            action=action,
                            result="Gathered all information needed to finish the task.",
                        )
                    ),
                    control=AgentLoopControl.BREAK,
                    termination_reason=AgentTerminationReason.SUBMIT,
                )
            tool_output = await self._acall_tool(tool_name, tool_args)
            action_results.append(ActionOutput(tool_name=tool_name, tool_args=tool_args, tool_output=tool_output))
            return AgentStepResult(
                history=turn_log.append_turn(
                    AvatarTurnEvent(action=action, result=tool_output if tool_output is not None else "")
                )
            )

        loop_result = await AgentLoopRunner[TurnLog]().run(
            max_iters=max_iters,
            initial_history=turn_log,
            step=step,
        )
        try:
            extracted = await call_with_history_truncation(
                self.finish,
                turn_log=loop_result.history,
                run=run,
                options=options,
                max_attempts=3,
                **actor_inputs,
            )
        except TruncationExhaustedError as err:
            logger.warning("Avatar finish failed after context window exceeded: %s", err)
            return Prediction(
                turn_log=loop_result.history,
                actions=action_results,
                termination_reason=loop_result.termination_reason or AgentTerminationReason.CONTEXT_WINDOW_EXCEEDED,
            )
        final_answer = extracted.result
        turn_log = extracted.turn_log
        return Prediction(
            **{key: getattr(final_answer, key) for key in self.output_fields},
            turn_log=turn_log,
            actions=action_results,
            termination_reason=loop_result.termination_reason,
        )
