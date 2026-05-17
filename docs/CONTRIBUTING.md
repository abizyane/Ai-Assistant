# Contributing

## Branch Strategy

- `master` — production-ready; protected; requires CI green + PR review.
- `feat/<task>` — feature branches; merge via PR.
- `fix/<issue>` — bug fix branches.

## Commit Conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add OpenAI adapter
fix: handle empty retrieval results gracefully
docs: update ARCHITECTURE.md with RRF formula
test: add unit tests for BGEReranker
refactor: extract citation grounding to domain service
chore: bump langchain to 0.3.x
```

## PR Checklist

Before opening a PR, ensure all of the following pass locally:

```bash
make lint        # ruff format + ruff check
make typecheck   # mypy --strict
make test        # pytest -m "not slow and not e2e and not integration"
```

CI will also run `docker build` and eval schema validation.

## Running the Local Stack

```bash
# Full stack (Postgres + API + Streamlit + Langfuse + Grafana)
make demo

# API only (faster for backend development)
make up

# Tear down
make down

# Wipe volumes (fresh start)
make clean
```

## Adding a New LLM Adapter

1. Create `src/infrastructure/llm/<provider>_llm.py`.
2. Implement `LLMPort` from `src/domain/ports/llm_port.py`.
3. Register in `src/infrastructure/container.py` under the `provider` switch.
4. Add unit tests in `tests/infrastructure/llm/test_<provider>_unit.py`.
5. Update the `RAG_LLM__PROVIDER` documentation in `README.md`.

## Adding a New Retrieval Strategy

1. Add a new method to `VectorStorePort` in `src/domain/ports/vector_store_port.py`.
2. Implement in `src/infrastructure/vector_store/pg_vector_store.py`.
3. Wire into the `retrieve` node in `src/infrastructure/agent/graph.py`.
4. Add integration tests in `tests/infrastructure/vector_store/`.

## Code Style

- **Formatter**: `ruff format` (Black-compatible, line length 100).
- **Linter**: `ruff check` with `select = ["E", "F", "I", "UP", "B", "SIM"]`.
- **Type checker**: `mypy --strict` — no `Any`, no untyped functions.
- **Docstrings**: Google-style on all public methods.
- **Imports**: stdlib → third-party → local; no relative imports.
