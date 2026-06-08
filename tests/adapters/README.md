# Adapter tests

## Tiers

1. **Contract** (`test_format_contract.py`, `call/test_*.py`) — roles, field markers, `lm_kwargs` structure.
2. **Golden** (`format_exact_messages_*` in adapter-specific files) — full prompt snapshots; keep curated.
3. **Integration** — end-to-end `acall` with `DummyLM` or mocked litellm.

## Layout

- `call/` — pipeline, parse fallback, response format routing, hierarchy.
- `scenarios/` — shared `TaskSpec` fixtures for parametrized contract tests.
- `types/tool/` — `Tool`, `ToolCalls`, schema helpers.
- `utils/` — parse-value and formatting utilities.
