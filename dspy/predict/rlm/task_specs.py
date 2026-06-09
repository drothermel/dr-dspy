from __future__ import annotations

from typing import TYPE_CHECKING

from dspy.adapters.prompt_format import translate_field_type
from dspy.history.repl_history import REPLHistory
from dspy.predict.rlm.tools import format_tool_docs
from dspy.task_spec import TaskSpec, input_field, make_task_spec, output_field

if TYPE_CHECKING:
    from dspy.predict.rlm.module import RLM
ACTION_INSTRUCTIONS_TEMPLATE = "You are tasked with producing the following outputs given the inputs {inputs}:\n{output_fields}\n\nYou have access to a Python REPL environment. Write Python code and it will be executed. You will see the output, then write more code based on what you learned. This is an iterative process.\n\nAvailable:\n- Variables: {inputs} (your input data)\n- `llm_query(prompt)` - query a sub-LLM (~500K char capacity) for semantic analysis\n- `llm_query_batched(prompts)` - query multiple prompts concurrently (much faster for multiple queries)\n- `print()` - ALWAYS print to see results\n- `SUBMIT({final_output_names})` - submit final output when done\n- Standard libraries: re, json, collections, math, etc.\n\nIMPORTANT: This is ITERATIVE. Each code block you write will execute, you'll see the output, then you decide what to do next. Do NOT try to solve everything in one step.\n\n1. EXPLORE FIRST - Look at your data before processing it. Print samples, check types/lengths, understand the structure.\n2. ITERATE - Write small code snippets, observe outputs, then decide next steps. State persists between iterations.\n3. VERIFY BEFORE SUBMITTING - If results seem wrong (zeros, empty, unexpected), reconsider your approach.\n4. USE llm_query FOR SEMANTICS - String matching finds WHERE things are; llm_query understands WHAT things mean.\n5. MINIMIZE RETYPING (INPUTS & OUTPUTS) - When values are long, precise, or error-prone (IDs, numbers, code, quotes), re-access them via variables and parse/compute in code instead of retyping. Use small, targeted prints to sanity-check, but avoid manual copying when variables can carry the exact value.\n6. SUBMIT ONLY AFTER SEEING OUTPUTS - SUBMIT ends the current run immediately. If you need to inspect printed output, run it in one step, review the result, then call SUBMIT in a later step.\n\nYou have max {max_llm_calls} sub-LLM calls. When done, call SUBMIT() with your output."


def build_task_specs(rlm: RLM) -> tuple[TaskSpec, TaskSpec]:
    inputs_str = ", ".join(f"`{n}`" for n in rlm.task_spec.input_fields)
    final_output_names = ", ".join(rlm.task_spec.output_fields.keys())
    output_fields = "\n".join(f"- {translate_field_type(field)}" for field in rlm.task_spec.output_fields.values())
    task_instructions = f"{rlm.task_spec.instructions}\n\n" if rlm.task_spec.instructions else ""
    tool_docs = format_tool_docs(rlm._user_tools)
    action_sig = make_task_spec(
        {
            "variables_info": input_field(
                "variables_info", str, desc="Metadata about the variables available in the REPL"
            ),
            "turn_log": input_field("turn_log", REPLHistory, desc="Previous REPL code executions and their outputs"),
            "iteration": input_field(
                "iteration", str, desc="Current iteration number (1-indexed) out of max_iterations"
            ),
            "reasoning": output_field(
                "reasoning", str, desc="Think step-by-step: what do you know? What remains? Plan your next action."
            ),
            "code": output_field(
                "code", str, desc="Python code to execute. Use markdown code block format: ```python\\n<code>\\n```"
            ),
        },
        instructions=task_instructions
        + ACTION_INSTRUCTIONS_TEMPLATE.format(
            inputs=inputs_str,
            final_output_names=final_output_names,
            output_fields=output_fields,
            max_llm_calls=rlm.max_llm_calls,
        )
        + tool_docs,
    )
    extract_instructions = "Based on the REPL trajectory, extract the final outputs now.\n\n            Review your trajectory to see what information you gathered and what values you computed, then provide the final outputs."
    extended_task_instructions = ""
    if task_instructions:
        extended_task_instructions = (
            "The trajectory was generated with the following objective: \n" + task_instructions + "\n"
        )
    full_extract_instructions = extended_task_instructions + extract_instructions
    extract_sig = make_task_spec(
        {
            "variables_info": input_field(
                "variables_info", str, desc="Metadata about the variables available in the REPL"
            ),
            "turn_log": input_field("turn_log", REPLHistory, desc="Your REPL interactions so far"),
            **rlm.task_spec.output_fields,
        },
        instructions=full_extract_instructions,
    )
    return (action_sig, extract_sig)
