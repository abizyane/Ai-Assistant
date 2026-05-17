# Architecture

This system is built as a hexagonal (Ports & Adapters) application. Business rules live
in the `domain` layer with zero dependencies on frameworks or I/O. Concrete technology
choices live in `infrastructure` and are wired into the application at startup.

## Layered diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ Interface                                                        │
│   FastAPI REST   │   Typer CLI   │   Streamlit Web              │
└──────────────┬───────────────────┬────────────────┬─────────────┘
               │                   │                │
               ▼                   ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│ Application                                                      │
│   Use cases: IngestDocuments · AnswerQuery · Evaluate            │
│   LangGraph agent: retrieve → rerank → generate → validate       │
└──────────────────────────────┬──────────────────────────────────┘
                               │ depends only on ports
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ Domain                                                           │
│   Entities: Chunk, Document, Session, Message, EvalReport        │
│   Ports:   LLMPort, EmbeddingPort, RerankerPort,                 │
│            VectorStorePort, SessionRepoPort,                     │
│            TracerPort, MetricsPort                               │
└──────────────────────────────▲──────────────────────────────────┘
                               │ implements ports
┌──────────────────────────────┴──────────────────────────────────┐
│ Infrastructure (adapters)                                        │
│   LLM: Gemini, OpenAI │ Embeddings: BGE-M3 │ Reranker: BGE       │
│   VectorStore: pgvector │ Persistence: SQLAlchemy + Postgres     │
│   Observability: Langfuse, Prometheus, structlog                 │
└─────────────────────────────────────────────────────────────────┘
```

## Ports and adapters

| Port | File | Concrete adapter(s) |
|------|------|---------------------|
| `LLMPort` | `src/domain/ports/llm_port.py` | `GeminiLLM`, `OpenAILLM` (`src/infrastructure/llm/`) |
| `EmbeddingPort` | `src/domain/ports/embedding_port.py` | `BGEM3Embedding` (`src/infrastructure/embeddings/`) |
| `RerankerPort` | `src/domain/ports/reranker_port.py` | `BGEReranker` (`src/infrastructure/reranking/`) |
| `VectorStorePort` | `src/domain/ports/vector_store_port.py` | `PGVectorStore` (`src/infrastructure/vector_store/`) |
| `SessionRepoPort` | `src/domain/ports/session_repo_port.py` | `SQLSessionRepo` (`src/infrastructure/persistence/`) |
| `TracerPort` | `src/domain/ports/tracer_port.py` | `LangfuseTracer` (`src/infrastructure/observability/`) |
| `MetricsPort` | `src/domain/ports/metrics_port.py` | `PrometheusMetrics` (`src/infrastructure/observability/`) |

## Request lifecycle — query path

1. Interface receives a query (CLI argument, HTTP POST, or Streamlit input).
2. `AnswerQuery` use case opens a Langfuse trace span and loads the session.
3. LangGraph agent runs the state machine:
   - **retrieve** — dense search via `VectorStorePort.search_dense` + sparse BM25; results fused with Reciprocal Rank Fusion.
   - **rerank** — top candidates passed through BGE cross-encoder; top-k kept.
   - **generate** — prompt assembled with retrieved chunks; `LLMPort.complete` called.
   - **validate** — answer checked for grounding; on failure, optional re-query.
4. Answer + citations + token counts returned. Trace span closed.
5. Prometheus metrics emitted: `rag_query_seconds`, `rag_tokens_total`, `rag_retrieval_hits`.

## Request lifecycle — ingest path

1. Interface invokes `IngestDocuments` with a directory or file list.
2. Loader reads files (PDF/MD/TXT) via `docling`; documents normalised.
3. Chunker splits into ~500-token windows with overlap.
4. `EmbeddingPort.embed_batch` produces dense vectors (BGE-M3, 1024 dims).
5. `VectorStorePort.upsert` writes rows to the `chunks` table (HNSW index).
6. Sparse BM25 index updated alongside dense vectors (same store).

## Key design decisions

- **Hexagonal layering**: testability and adapter swap without touching domain.
- **LangGraph over ad-hoc chains**: explicit state, easier debugging, retries per node.
- **Hybrid retrieval (dense + sparse)**: BM25 catches exact tokens; dense catches paraphrase.
- **Reciprocal Rank Fusion**: parameter-free; no tuning of dense/sparse weights.
- **pgvector over a dedicated vector DB**: one storage system, transactional consistency, lower ops cost.
- **BGE-M3**: single multilingual model covers EN/FR/AR with one embedding space.
- **Pluggable LLM**: provider switch via env var; no code change.
