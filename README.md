# 1337 RAG Assistant

> Production-grade multilingual agentic RAG over the 1337 Coding School corpus

[![CI](https://github.com/abizyane/Ai-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/abizyane/Ai-Assistant/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-82%25-brightgreen)](htmlcov/index.html)
[![mypy](https://img.shields.io/badge/mypy-strict-blue)](https://mypy.readthedocs.io/)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Langfuse](https://img.shields.io/badge/observability-Langfuse-purple)](https://langfuse.com/)
[![Prometheus](https://img.shields.io/badge/metrics-Prometheus-orange)](https://prometheus.io/)

## Why this project

- **Hexagonal architecture + SOLID** — domain logic is fully isolated from infrastructure; swap Gemini for OpenAI or pgvector for Qdrant by implementing one port interface, zero domain changes.
- **Full observability stack** — every LLM call, retrieval step, and rerank decision is traced in Langfuse, metered in Prometheus, visualised in Grafana, and logged to Loki. Nothing is a black box.
- **Eval-gated CI** — Ragas faithfulness/relevancy/precision thresholds are enforced in GitHub Actions; a regression in retrieval quality breaks the build before it reaches production.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Interfaces                                                      │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Typer CLI│  │ FastAPI REST │  │    Streamlit Web UI      │  │
│  └────┬─────┘  └──────┬───────┘  └────────────┬─────────────┘  │
│       └───────────────┴──────────────────────┬─┘               │
│                                              │                  │
│  ┌───────────────────────────────────────────▼──────────────┐  │
│  │  DI Container (composition root)                         │  │
│  └───────────────────────────────────────────┬──────────────┘  │
│                                              │                  │
│  ┌───────────────────────────────────────────▼──────────────┐  │
│  │  LangGraph Agent (rewrite → retrieve → grade →           │  │
│  │                   generate → verify)                     │  │
│  └───────────────────────────────────────────┬──────────────┘  │
│                                              │                  │
│  ┌───────────────────────────────────────────▼──────────────┐  │
│  │  Use Cases: Ingest | Answer | Evaluate                   │  │
│  └───────────────────────────────────────────┬──────────────┘  │
│                                              │                  │
│  ┌───────────────────────────────────────────▼──────────────┐  │
│  │  Ports (interfaces)                                      │  │
│  │  LLMPort | EmbeddingPort | VectorStorePort |             │  │
│  │  SessionRepoPort | TracerPort | MetricsPort              │  │
│  └───────────────────────────────────────────┬──────────────┘  │
│                                              │                  │
│  ┌───────────────────────────────────────────▼──────────────┐  │
│  │  Adapters                                                │  │
│  │  Gemini/OpenAI │ BGE-M3 │ pgvector │ Langfuse │ Prom.   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full deep-dive.

## Quickstart

```bash
git clone https://github.com/abizyane/Ai-Assistant
cd Ai-Assistant
cp .env.example .env         # set RAG_LLM__API_KEY and provider
make demo                    # spins up Postgres + API + Streamlit + Langfuse + Grafana
```

Services after `make demo`:

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI REST | http://localhost:8000/docs |
| Langfuse traces | http://localhost:3000 |
| Grafana dashboards | http://localhost:3001 |
| Prometheus | http://localhost:9090 |

## Feature Matrix

| Feature | Status | Notes |
|---------|--------|-------|
| Multilingual (EN / FR / AR) | ✅ | BGE-M3 multilingual embeddings |
| Hybrid retrieval (dense + sparse + RRF) | ✅ | pgvector HNSW + BM25, k=60 |
| Cross-encoder reranking | ✅ | BGE-reranker-v2-m3 |
| Agentic loop (rewrite / grade / verify) | ✅ | LangGraph state machine |
| Langfuse distributed tracing | ✅ | Every span with token + cost |
| Prometheus metrics | ✅ | Latency, token counts, eval scores |
| Grafana dashboards | ✅ | RAG ops + LLM costs + eval trends |
| Eval-gated CI | ✅ | Ragas faithfulness ≥ 0.85 |
| Pluggable LLMs (Gemini / OpenAI) | ✅ | One port, multiple adapters |
| CPU-only deployment | ✅ | No GPU required |
| Persistent sessions | ✅ | PostgreSQL-backed chat history |

## Multilingual Demo

The system handles English, French, and Arabic queries over the same corpus — no per-language configuration required. BGE-M3 encodes all three in the same embedding space.

**English**
```
$ python -m src.interface.cli.main chat --query "What are the entry requirements for 1337?"

Answer: 1337 has no formal entry requirements — admission is based solely on the Piscine
(a 26-day intensive coding bootcamp). No prior programming experience is necessary.

Sources: [chunk_042] admissions.pdf p.3 · [chunk_071] faq.pdf p.1
Langfuse trace: https://localhost:3000/trace/abc123
```

**Français**
```
$ python -m src.interface.cli.main chat --query "Quels sont les projets du cursus C ?"

Réponse: Le cursus C comprend une série de projets allant de libft (recoder la stdlib C)
jusqu'à minishell (un shell POSIX simplifié), en passant par push_swap, pipex et ft_printf.

Sources: [chunk_018] curriculum.pdf p.7 · [chunk_019] curriculum.pdf p.8
```

**العربية**
```
$ python -m src.interface.cli.main chat --query "ما هي مشاريع مستوى 0 ؟"

الجواب: مشاريع المستوى 0 تشمل libft التي تعيد تنفيذ مكتبة C القياسية، وft_printf لتنفيذ
دالة printf، وget_next_line لقراءة السطر التالي من ملف.

Sources: [chunk_022] projects.pdf p.2 · [chunk_031] projects.pdf p.5
```

## Tech Stack

| Component | Library | Why chosen |
|-----------|---------|------------|
| LLM | `langchain-google-genai` (Gemini 2.0 Flash) | Best cost/quality ratio; pluggable via port |
| Embeddings | `BAAI/bge-m3` | State-of-the-art multilingual dense + sparse in one model |
| Reranker | `BAAI/bge-reranker-v2-m3` | Cross-encoder; same multilingual coverage as embedder |
| Vector store | `pgvector` + `langchain-postgres` | HNSW index; no extra infra beyond Postgres |
| Agent framework | `langgraph` | Explicit state machine; easy to inspect and extend |
| Observability | `langfuse` | Native LangChain integration; self-hostable |
| Metrics | `prometheus-client` | Industry standard; Grafana-native |
| API | `fastapi` + `uvicorn` | Async-first; SSE streaming built-in |
| CLI | `typer` | Type-safe; auto-generated help |
| UI | `streamlit` | Rapid prototyping; no JS required |
| Config | `pydantic-settings` | Env-var validation with type safety |
| Testing | `pytest` + `pytest-asyncio` + `testcontainers` | Full pyramid: unit → integration → e2e |
| Eval | `ragas` | Standard RAG evaluation metrics |
| Linting | `ruff` + `mypy --strict` | Zero-tolerance code quality |

## Evaluation

Golden dataset: [`evals/golden_set.jsonl`](evals/golden_set.jsonl) — 41 QA pairs across EN (13), FR (13), AR (10), mixed (5).

Run evaluation:

```bash
make eval
```

Current thresholds enforced in CI:

| Metric | Threshold |
|--------|-----------|
| Faithfulness | ≥ 0.85 |
| Answer Relevancy | ≥ 0.80 |
| Context Precision | ≥ 0.75 |
| Context Recall | ≥ 0.70 |

> Live scores require a real LLM API key. CI validates dataset schema only (no API calls).

See [`docs/EVALUATION.md`](docs/EVALUATION.md) for methodology and threshold rationale.

## Project Structure

```
.
├── src/
│   ├── config/
│   │   └── settings.py          # Pydantic-settings; all env vars
│   ├── domain/
│   │   ├── entities/            # Chunk, Session, Message, EvalResult
│   │   └── ports/               # Abstract interfaces (LLMPort, VectorStorePort, …)
│   ├── application/
│   │   └── use_cases/           # ingest.py, answer.py, evaluate.py
│   ├── infrastructure/
│   │   ├── llm/                 # GeminiLLM, OpenAILLM adapters
│   │   ├── embedding/           # BGEM3Embedding adapter
│   │   ├── reranker/            # BGEReranker adapter
│   │   ├── vector_store/        # PGVectorStore adapter
│   │   ├── persistence/         # SessionRepository (Postgres)
│   │   ├── observability/       # LangfuseTracer, PrometheusMetrics
│   │   └── agent/               # LangGraph agentic RAG graph
│   └── interface/
│       ├── api/                 # FastAPI app (chat SSE, ingest, health)
│       ├── cli/                 # Typer CLI (ingest, chat, evaluate, health)
│       └── web/                 # Streamlit UI
├── tests/
│   ├── domain/                  # Pure unit tests
│   ├── application/             # Use-case unit tests (mocked ports)
│   ├── infrastructure/          # Integration tests (testcontainers)
│   └── e2e/                     # End-to-end tests
├── evals/
│   ├── golden_set.jsonl         # 41 QA pairs EN+FR+AR+mixed
│   └── schema.json
├── docs/
│   ├── ARCHITECTURE.md
│   ├── OBSERVABILITY.md
│   ├── EVALUATION.md
│   └── CONTRIBUTING.md
├── .github/workflows/ci.yml     # Lint + type + test + eval + docker build
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

## Configuration

All settings use `pydantic-settings` with env-var prefix `RAG_` and nested delimiter `__`.

| Env Var | Default | Description |
|---------|---------|-------------|
| `RAG_LLM__API_KEY` | *(required)* | API key for the LLM provider |
| `RAG_LLM__PROVIDER` | `gemini` | LLM provider: `gemini` \| `openai` |
| `RAG_LLM__MODEL` | `gemini-2.0-flash-exp` | Model name/ID |
| `RAG_LLM__TEMPERATURE` | `0.1` | Sampling temperature (0–2) |
| `RAG_LLM__MAX_TOKENS` | `4096` | Maximum output tokens |
| `RAG_LLM__MAX_RETRIES` | `3` | Max retry attempts on transient errors |
| `RAG_EMBEDDING__MODEL` | `BAAI/bge-m3` | HuggingFace embedding model ID |
| `RAG_EMBEDDING__BATCH_SIZE` | `32` | Encoding batch size |
| `RAG_EMBEDDING__CACHE_DIR` | `/tmp/hf_cache` | HuggingFace model cache |
| `RAG_VECTORSTORE__DATABASE_URL` | `postgresql+asyncpg://raguser:ragpassword@localhost:5432/ragdb` | Async PostgreSQL URL |
| `RAG_VECTORSTORE__TABLE_NAME` | `chunks` | pgvector table name |
| `RAG_VECTORSTORE__TOP_K_DENSE` | `20` | Dense retrieval top-k |
| `RAG_VECTORSTORE__TOP_K_SPARSE` | `20` | Sparse (BM25) retrieval top-k |
| `RAG_VECTORSTORE__TOP_K_RERANK` | `5` | After-rerank top-k |
| `RAG_RERANKER__MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model ID |
| `RAG_LANGFUSE__HOST` | `http://localhost:3000` | Langfuse server URL |
| `RAG_LANGFUSE__PUBLIC_KEY` | `` | Langfuse public key |
| `RAG_LANGFUSE__SECRET_KEY` | `` | Langfuse secret key |
| `RAG_LANGFUSE__ENABLED` | `false` | Enable Langfuse tracing |
| `RAG_PROMETHEUS__PORT` | `9090` | Prometheus scrape port |
| `RAG_EVAL__FAITHFULNESS` | `0.85` | Minimum faithfulness threshold |
| `RAG_EVAL__ANSWER_RELEVANCY` | `0.80` | Minimum answer relevancy threshold |
| `RAG_EVAL__CONTEXT_PRECISION` | `0.75` | Minimum context precision threshold |
| `RAG_EVAL__CONTEXT_RECALL` | `0.70` | Minimum context recall threshold |

## Development Guide

```bash
# Install dev dependencies
uv sync --group dev

# Lint + format check
make lint

# Type-check (mypy --strict)
make typecheck

# Run tests (unit + application, no slow/e2e)
make test

# Run full test suite including integration
make test-all

# Run evaluation (requires LLM API key)
make eval

# Spin up full stack for manual QA
make demo

# Smoke test against running stack
make smoke
```

### Adding a new LLM adapter

1. Implement `src/domain/ports/llm_port.py::LLMPort` in a new file under `src/infrastructure/llm/`.
2. Register it in `src/infrastructure/container.py` behind a `provider` config switch.
3. Add unit tests in `tests/infrastructure/llm/`.

### Adding a new language to the golden set

1. Write QA pairs grounded in `data/knowledge_base/` source documents.
2. Append JSON lines to `evals/golden_set.jsonl` following the schema in `evals/schema.json`.
3. Run `python evals/validate_dataset.py evals/golden_set.jsonl` to verify.

## Roadmap (out of scope, documented for reference)

- **CRAG / Self-RAG** — corrective and self-reflective retrieval loops
- **Multi-agent orchestration** — specialist sub-agents per domain
- **Auth / multi-tenancy** — JWT + per-tenant vector namespaces
- **Kubernetes deployment** — Helm chart + HPA for the API tier

## Screenshots

| Streamlit EN | Streamlit FR | Langfuse trace |
|---|---|---|
| ![EN chat](docs/assets/screenshots/streamlit-chat-en.png) | ![FR chat](docs/assets/screenshots/streamlit-chat-fr.png) | ![trace](docs/assets/screenshots/langfuse-trace.png) |

| Grafana RAG ops | Grafana LLM costs | Grafana eval trends |
|---|---|---|
| ![rag ops](docs/assets/screenshots/grafana-rag-ops.png) | ![llm costs](docs/assets/screenshots/grafana-llm-costs.png) | ![eval trends](docs/assets/screenshots/grafana-eval-trends.png) |

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built on [LangChain](https://github.com/langchain-ai/langchain), [LangGraph](https://github.com/langchain-ai/langgraph), [Langfuse](https://github.com/langfuse/langfuse), [Ragas](https://github.com/explodinggradients/ragas), [pgvector](https://github.com/pgvector/pgvector), and [BGE models](https://huggingface.co/BAAI) from BAAI.
