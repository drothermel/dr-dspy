import logging
from collections.abc import Callable
from typing import Any, cast

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types.tool import Tool
from dspy.dsp.utils.settings import settings
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.task_spec import FieldSpec, TaskSpec, default_task_instructions, make_task_spec
from dspy.utils.exceptions import ContextWindowExceededError

logger = logging.getLogger(__name__)


class ReAct(Module):
    def __init__(self, task_spec: TaskSpec, tools: list[Callable], max_iters: int = 20) -> None:
        """
        ReAct stands for "Reasoning and Acting," a popular paradigm for building tool-using agents.
        In this approach, the language model is iteratively provided with a list of tools and has
        to reason about the current situation. The model decides whether to call a tool to gather more
        information or to finish the task based on its reasoning process. The DSPy version of ReAct is
        generalized to work over any task spec, thanks to task-spec polymorphism.

        Args:
            task_spec: The task spec of the module, which defines the input and output of the react module.
            tools (list[Callable]): A list of functions, callable objects, or `dspy.adapters.types.tool.Tool`
                instances.
            max_iters (int | None): The maximum number of iterations to run. Defaults to 10.

        Examples:

        ```python
        def get_weather(city: str) -> str:
            return f"The weather in {city} is sunny."

        from dspy.predict.react import ReAct
        from dspy.task_spec import make_task_spec

        react = ReAct(
            make_task_spec("question->answer", instructions="Answer the question."),
            tools=[get_weather],
        )
        pred = react(question="What is the weather in Tokyo?")
        ```
        """
        super().__init__()
        if not isinstance(task_spec, TaskSpec):
            raise TypeError(f"ReAct requires a TaskSpec instance, got {type(task_spec).__name__}.")
        self.task_spec = task_spec
        self.max_iters = max_iters

        normalized_tools: list[Tool] = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        tools_by_name: dict[str, Tool] = {}
        for tool in normalized_tools:
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
            name="finish",
            desc=f"Marks the task as complete. That is, signals that all information for producing the outputs, i.e. {outputs}, are now available to be extracted.",
            args={},
        )

        for idx, tool in enumerate(tools_by_name.values()):
            instr.append(f"({idx + 1}) {tool}")
        instr.append("When providing `next_tool_args`, the value inside the field must be in JSON format")

        react_task_spec = (
            make_task_spec(
                dict(task_spec.input_fields),
                instructions="\n".join(instr),
            )
            .append(FieldSpec.input("trajectory", str))
            .append(FieldSpec.output("next_thought", str))
            .append(FieldSpec.output("next_tool_name", str))
            .append(FieldSpec.output("next_tool_args", dict[str, Any]))
        )

        fallback_task_spec = make_task_spec(
            {**task_spec.input_fields, **task_spec.output_fields},
            instructions=task_spec.instructions,
        ).append(FieldSpec.input("trajectory", str))

        self.tools = tools_by_name
        self.react = Predict(react_task_spec)
        self.extract = ChainOfThought(fallback_task_spec)

    def _format_trajectory(self, trajectory: dict[str, Any]):
        adapter = settings.adapter or ChatAdapter()
        trajectory_keys = ", ".join(trajectory.keys())
        trajectory_task_spec = make_task_spec(
            f"{trajectory_keys} -> x",
            instructions=default_task_instructions(
                inputs=tuple(trajectory.keys()),
                outputs=("x",),
            ),
        )
        return adapter.format_user_message_content(task_spec=trajectory_task_spec, inputs=trajectory)

    async def aforward(self, **input_args):
        trajectory = {}
        max_iters = input_args.pop("max_iters", self.max_iters)
        for idx in range(max_iters):
            try:
                pred = await self._call_with_potential_trajectory_truncation(self.react, trajectory, **input_args)
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

        extract = await self._call_with_potential_trajectory_truncation(self.extract, trajectory, **input_args)
        return Prediction(trajectory=trajectory, **extract)

    async def _call_with_potential_trajectory_truncation(self, module, trajectory, **input_args):
        for _ in range(3):
            try:
                return await module(
                    **input_args,
                    trajectory=self._format_trajectory(trajectory),
                )
            except ContextWindowExceededError:
                logger.warning("Trajectory exceeded the context window, truncating the oldest tool call information.")
                trajectory = self.truncate_trajectory(trajectory)
        raise ValueError("The context window was exceeded even after 3 attempts to truncate the trajectory.")

    def truncate_trajectory(self, trajectory):
        """Truncates the trajectory so that it fits in the context window.

        Users can override this method to implement their own truncation logic.
        """
        keys = list(trajectory.keys())
        if len(keys) < 4:
            # Every tool call has 4 keys: thought, tool_name, tool_args, and observation.
            raise ValueError(
                "The trajectory is too long so your prompt exceeded the context window, but the trajectory cannot be "
                "truncated because it only has one tool call."
            )

        for key in keys[:4]:
            trajectory.pop(key)

        return trajectory


def _fmt_exc(err: BaseException, *, limit: int = 5) -> str:
    """
    Return a one-string traceback summary.
    * `limit` - how many stack frames to keep (from the innermost outwards).
    """

    import traceback

    return "\n" + "".join(traceback.format_exception(type(err), err, err.__traceback__, limit=limit)).strip()
