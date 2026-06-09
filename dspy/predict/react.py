import logging
from typing import Any, cast

from dspy.adapters.types.tool import Tool
from dspy.compile.resolve import resolve_adapter
from dspy.core.types.call_options import ModuleCallOptions
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.runtime.run_context import RunContext, resolve_run
from dspy.task_spec import TaskSpec, default_task_instructions, input_field, make_task_spec, output_field
from dspy.utils.exceptions import ContextWindowExceededError

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
                f"You are an Agent. In each episode, you will be given the fields {inputs} as input. And you can see your past trajectory so far.",
                f"Your goal is to use one or more of the supplied tools to collect any necessary information for producing {outputs}.\n",
                "To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.",
                "After each tool call, you receive a resulting observation, which gets appended to your trajectory.\n",
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
            .append(input_field("trajectory", Any))
            .append(output_field("next_thought", str))
            .append(output_field("next_tool_name", str))
            .append(output_field("next_tool_args", dict[str, Any]))
        )
        fallback_task_spec = make_task_spec(
            {**task_spec.input_fields, **task_spec.output_fields}, instructions=task_spec.instructions
        ).append(input_field("trajectory", Any))
        self.tools = tools_by_name
        self.react = Predict(react_task_spec)
        self.extract = ChainOfThought(fallback_task_spec)

    def _format_trajectory(self, trajectory: dict[str, Any], run: RunContext):
        adapter, _ = resolve_adapter(run.adapter, transparency=run.telemetry.transparency)
        trajectory_keys = ", ".join(trajectory.keys())
        trajectory_task_spec = make_task_spec(
            f"{trajectory_keys} -> x",
            instructions=default_task_instructions(inputs=tuple(trajectory.keys()), outputs=("x",)),
        )
        return adapter.format_user_message_content(task_spec=trajectory_task_spec, inputs=trajectory)

    async def aforward(
        self,
        *,
        run: RunContext,
        options: ModuleCallOptions | None = None,
        **input_args,
    ):
        run = resolve_run(run=run, bound_run=self.run)
        trajectory = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            try:
                pred = await self._call_with_potential_trajectory_truncation(
                    self.react, trajectory, run, options=options, **input_args
                )
            except ValueError as err:
                logger.warning(f"Ending the trajectory: Agent failed to select a valid tool: {_fmt_exc(err)}")
                break
            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args
            try:
                tool = self.tools[pred.next_tool_name]
                trajectory[f"observation_{idx}"] = await cast("Any", tool).acall(**pred.next_tool_args)
            except Exception as err:
                trajectory[f"observation_{idx}"] = f"Execution error in {pred.next_tool_name}: {_fmt_exc(err)}"
            if pred.next_tool_name == "finish":
                break
        extract = await self._call_with_potential_trajectory_truncation(
            self.extract, trajectory, run, options=options, **input_args
        )
        return Prediction(trajectory=trajectory, **extract)

    async def _call_with_potential_trajectory_truncation(self, module, trajectory, run, *, options=None, **input_args):
        for _ in range(3):
            try:
                return await module(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory, run),
                    run=run,
                    options=options,
                )
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                trajectory = self.truncate_trajectory(trajectory)
        raise ValueError("The context window was exceeded even after 3 attempts to truncate the trajectory.")

    def truncate_trajectory(self, trajectory):
        keys = list(trajectory.keys())
        if len(keys) < 4:
            raise ValueError(
                "The trajectory is too long so your prompt exceeded the context window, but the trajectory cannot be truncated because it only has one tool call."
            )
        for key in keys[:4]:
            trajectory.pop(key)
        return trajectory


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    import traceback

    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()
