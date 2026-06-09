from copy import deepcopy

from dspy.predict.avatar.models import Action, ActionOutput, Tool
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.primitives.prediction import Prediction
from dspy.task_spec import FieldSpec, TaskSpec, input_field, output_field


class ActorTaskSpec(TaskSpec):
    name: str = "Actor"
    instructions: str = "You will be given `Tools` which will be a list of tools to use to accomplish the `Goal`. Given the user query, your task is to decide which tool to use and what input values to provide.\n\nYou will output action needed to accomplish the `Goal`. `Action` should have a tool to use and the input query to pass to the tool.\n\nNote: You can opt to use no tools and provide the final answer directly. You can also one tool multiple times with different input queries if applicable."
    inputs: tuple[FieldSpec, ...] = (
        input_field("goal", str, desc="Task to be accomplished."),
        input_field("tools", list[str], desc="list of tools to use"),
    )
    outputs: tuple[FieldSpec, ...] = (output_field("action_1", Action, desc="1st action to take."),)


def get_number_with_suffix(number: int) -> str:
    if number == 1:
        return "1st"
    if number == 2:
        return "2nd"
    if number == 3:
        return "3rd"
    return f"{number}th"


class Avatar(Module):
    def __init__(self, task_spec: TaskSpec, tools, max_iters=3, verbose=False) -> None:
        if not isinstance(task_spec, TaskSpec):
            raise TypeError(f"Avatar requires a TaskSpec instance, got {type(task_spec).__name__}.")
        super().__init__()
        self.task_spec = task_spec
        self.input_fields = task_spec.input_fields
        self.output_fields = task_spec.output_fields
        self.finish_tool = Tool(tool=None, name="Finish", desc="returns the final output and finishes the task")
        self.tools = tools + [self.finish_tool]
        actor_task_spec = ActorTaskSpec()
        for field_name in list(self.input_fields.keys())[::-1]:
            field = self.input_fields[field_name]
            actor_task_spec = actor_task_spec.prepend(
                input_field(field_name, field.type_, desc=field.desc, prefix=field.prefix)
            )
        self.verbose = verbose
        self.max_iters = max_iters
        self.actor = Predict(actor_task_spec)
        self.actor_clone = deepcopy(self.actor)

    def _promote_output_to_input(self, task_spec: TaskSpec, name: str, *, type_) -> TaskSpec:
        field = task_spec.output_fields[name]
        return task_spec.delete(name).append(input_field(name, type_, desc=field.desc, prefix=field.prefix))

    def _update_task_spec(self, idx: int, omit_action: bool = False) -> None:
        task_spec = self.actor.task_spec
        task_spec = self._promote_output_to_input(task_spec, f"action_{idx}", type_=Action)
        task_spec = task_spec.append(
            input_field(f"result_{idx}", str, desc=f"{get_number_with_suffix(idx)} result", prefix=f"Result {idx}:")
        )
        if omit_action:
            for field_name, field in self.output_fields.items():
                task_spec = task_spec.append(
                    output_field(field_name, field.type_, desc=field.desc, prefix=field.prefix)
                )
        else:
            task_spec = task_spec.append(
                output_field(
                    f"action_{idx + 1}",
                    Action,
                    desc=f"{get_number_with_suffix(idx + 1)} action to taken",
                    prefix=f"Action {idx + 1}:",
                )
            )
        self.actor.task_spec = task_spec

    def _call_tool(self, tool_name: str, tool_input_query: str) -> str | None:
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.tool.run(tool_input_query)
        return None

    async def aforward(self, **kwargs):
        if self.verbose:
            pass
        args = {"goal": self.task_spec.instructions, "tools": [tool.name for tool in self.tools]}
        for key in self.input_fields:
            if key in kwargs:
                args[key] = kwargs[key]
        idx = 1
        tool_name = None
        action_results: list[ActionOutput] = []
        max_iters = kwargs.get("max_iters")
        while tool_name != "Finish" and (max_iters > 0 if max_iters else True):
            actor_output = await self.actor(**args)
            action = getattr(actor_output, f"action_{idx}")
            tool_name = action.tool_name
            tool_input_query = action.tool_input_query
            if self.verbose:
                pass
            if tool_name != "Finish":
                tool_output = self._call_tool(tool_name, tool_input_query)
                action_results.append(
                    ActionOutput(tool_name=tool_name, tool_input_query=tool_input_query, tool_output=tool_output)
                )
                self._update_task_spec(idx)
                args[f"action_{idx}"] = action
                args[f"result_{idx}"] = tool_output if tool_output is not None else ""
            else:
                self._update_task_spec(idx, omit_action=True)
                args[f"action_{idx}"] = action
                args[f"result_{idx}"] = "Gathered all information needed to finish the task."
                break
            idx += 1
            if max_iters:
                max_iters -= 1
        final_answer = await self.actor(**args)
        self.actor = deepcopy(self.actor_clone)
        return Prediction(**{key: getattr(final_answer, key) for key in self.output_fields}, actions=action_results)
