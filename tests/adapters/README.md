# Adapter tests

## Tiers

1. **Contract** (`test_format_contract.py`, `call/test_*.py`) — roles, field markers, `lm_kwargs` structure, pipeline stage mutations.
2. **Golden** (`format_exact_messages_*` in adapter-specific files) — full prompt snapshots; keep curated.
3. **Integration** — end-to-end `acall` with `DummyLM` or mocked litellm.

## Layout

- `call/` — pipeline, postprocess, stages (`PreparedAdapterCall.mutations`), parse fallback, response format routing, wrappers, hierarchy.
- `scenarios/` — shared `TaskSpec` fixtures for parametrized contract tests.
- `types/tool/` — `Tool`, `ToolCalls`, schema helpers.
- `utils/` — parse-value and formatting utilities.
- `test_format_shared.py` — shared format helper contracts (`output_field_type_hint`, `build_role_field_sections`).
- `test_base_tool_calls.py` — provider tool-call normalization parity with `normalize_tool_call_dict`.
- `test_repl_history_format.py` — REPLHistory stays inline across Chat/JSON/XML adapters.

Golden prompt snapshots live in adapter-specific files (`test_chat_adapter.py`, `test_json_adapter.py`, etc.) as `format_exact_messages_*` tests.
