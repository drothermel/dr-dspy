# Changelog

## 3.3.0b1

### Breaking changes

Compatibility shims removed in the LM boundary and core API cutover. Users upgrading should expect:

1. `lm.history[i]["messages"]` → `lm.history[i].request.messages` (typed `LMHistoryEntry`)
2. `LM(..., reasoning_effort="low")` → `LM(..., reasoning={"effort": "low"})` or typed `LMConfig` fields
3. OpenAI-shaped `tool_choice` dicts are rejected at the LM boundary; use `LMToolChoice`
4. `InputField(prefix=…)` is rejected
5. `Example.toDict()` removed → use `Example.to_dict()`
6. `ChainOfThought(rationale_field=…)` and `rationale_field_type` removed; CoT always prepends `Reasoning`
7. `streamify()` without listeners no longer yields raw provider chunks
8. Metrics must accept `(example, prediction, trace)`; `Evaluate` passes `list(settings.trace)`
9. Custom `Type` values render as LM content blocks at format time, not marker strings
10. `Adapter.format()` returns `list[LMMessage]` instead of OpenAI chat dicts
11. `Type.parse_lm_response()` removed; implement `parse_lm_output(LMOutput)` for native response types
12. Cross-version JSON program state from DSPy 3.0.x is not supported; use pickle program saves or re-optimize
13. `named_parameters()` / `named_predictors()` paths are aligned with `named_sub_modules()` (e.g. `self.predict` instead of `predict`)
