import logging
from typing import Any, cast

from dspy.adapters.types.tool import Tool
from dspy.core.types.call_options import ModuleCallOptions
from dspy.history import TurnEvent, TurnLog, call_with_turn_log_truncation
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.runtime.run_context import RunContext, resolve_run
from dspy.task_spec import TaskSpec, input_field, make_task_spec, output_field

logger = logging.getLogger(__name__)


class ReAct(Module):
    def __init__(self, task_spec: TaskSpec, tools: list[Tool], max_iters: int = 20) -> None:
        super().__init__()
        if not isinstance(task_spec, TaskSpec):
            raise TypeError(f"ReAct requires a TaskSpec instance, got {type(task_spec).__name__}.")
        self.task_spec = task_spec
        self.max_iters = max_iters
        tools_by_name: dict[str, Tool] = {}
        for tool in tools:
            if not isinstance(tool, Tool):
                raise TypeError(
                    "tools must be Tool instances with an explicit description. Use Tool(func, description='...')."
                )
            if tool.name is None:
                raise ValueError("Tool name could not be determined.")
            tools_by_name[tool.name] = tool
        inputs = ", ".join([f"`{k}`" for k in task_spec.input_fields])
        outputs = ", ".join([f"`{k}`" for k in task_spec.output_fields])
        instr = [f"{task_spec.instructions}\n"] if task_spec.instructions else []
        instr.extend(
            [
                f"You are an Agent. In each episode, you will be given the fields {inputs} as input. And you can see your past turn log so far.",
                f"Your goal is to use one or more of the supplied tools to collect any necessary information for producing {outputs}.\n",
                "To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.",
                "After each tool call, you receive a resulting observation, which gets appended to your turn log.\n",
                "When writing next_thought, you may reason about the current situation and plan for future steps.",
                "When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n",
            ]
        )
        tools_by_name["finish"] = Tool(
            func=lambda: "Completed.",
            description=f"Marks the task as complete. That is, signals that all information for producing the outputs, i.e. {outputs}, are now available to be extracted.",
            name="finish",
            args={},
        )
        for idx, tool in enumerate(tools_by_name.values()):
            instr.append(f"({idx + 1}) {tool}")
        instr.append("When providing `next_tool_args`, the value inside the field must be in JSON format")
        react_task_spec = (
            make_task_spec(dict(task_spec.input_fields), instructions="\n".join(instr))
            .append(input_field("turn_log", TurnLog, desc="Previous thoughts, tool calls, and tool results."))
            .append(output_field("next_thought", str, desc="Your next reasoning step toward solving the task."))
            .append(output_field("next_tool_name", str, desc="Name of the tool to call next."))
            .append(output_field("next_tool_args", dict[str, Any], desc="JSON arguments for the next tool call."))
        )
        fallback_task_spec = make_task_spec(
            {**task_spec.input_fields, **task_spec.output_fields}, instructions=task_spec.instructions
        ).append(input_field("turn_log", TurnLog, desc="Previous thoughts, tool calls, and tool results."))
        self.tools = tools_by_name
        self.react = Predict(react_task_spec)
        self.extract = ChainOfThought(fallback_task_spec)

    async def aforward(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **input_args,
    ):
        run = resolve_run(run=run, bound_run=self.run)
        turn_log = TurnLog.empty()
        max_iters = input_args.pop("max_iters", self.max_iters)
        for _idx in range(max_iters):
            try:
                pred = await self._call_with_potential_turn_log_truncation(
                    self.react, turn_log, run, options=options, **input_args
                )
            except ValueError as err:
                logger.warning(f"Ending the agent loop: Agent failed to select a valid tool: {_fmt_exc(err)}")
                break
            try:
                tool = self.tools[pred.next_tool_name]
                observation = await cast("Any", tool).acall(**pred.next_tool_args)
            except Exception as err:
                observation = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"
            turn_log = turn_log.append_turn(
                TurnEvent(
                    thought=pred.next_thought,
                    tool_name=pred.next_tool_name,
                    tool_args=pred.next_tool_args,
                    observation=observation,
                )
            )
            if pred.next_tool_name == "finish":
                break
        extract = await call_with_turn_log_truncation(
            self.extract, turn_log=turn_log, run=run, options=options, **input_args
        )
        return Prediction(turn_log=turn_log, **extract)


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    import traceback

    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()
