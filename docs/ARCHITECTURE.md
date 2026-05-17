# Architecture

## Hexagonal Architecture (Ports & Adapters)

The codebase follows strict hexagonal architecture. The dependency rule is enforced at every layer boundary: **outer layers depend on inner layers, never the reverse**.

```
src/
├── domain/          ← innermost; no external imports
│   ├── entities/    ← pure data classes (Chunk, Session, Message, EvalResult)
│   └── ports/       ← abstract interfaces (LLMPort, VectorStorePort, …)
├── application/     ← orchestrates domain; depends only on domain
│   └── use_cases/   ← Ingest, Answer, Evaluate
├── infrastructure/  ← implements ports; depends on domain + third-party libs
│   ├── llm/         ← GeminiLLM, OpenAILLM
│   ├── embedding/   ← BGEM3Embedding
│   ├── reranker/    ← BGEReranker
│   ├── vector_store/← PGVectorStore
│   ├── persistence/ ← SessionRepository
│   ├── observability/← LangfuseTracer, PrometheusMetrics
│   └── agent/       ← LangGraph agentic RAG graph
└── interface/       ← outermost; depends on application + infrastructure
    ├── api/         ← FastAPI REST
    ├── cli/         ← Typer CLI
    └── web/         ← Streamlit UI
```

## Port Contracts

Each port is an abstract base class in `src/domain/ports/`. Adapters in `src/infrastructure/` implement them.

| Port | Adapter(s) | Purpose |
|------|-----------|---------|
| `LLMPort` | `GeminiLLM`, `OpenAILLM` | Generate text completions |
| `EmbeddingPort` | `BGEM3Embedding` | Encode text to dense + sparse vectors |
| `RerankerPort` | `BGEReranker` | Cross-encoder reranking |
| `VectorStorePort` | `PGVectorStore` | Upsert and hybrid search |
| `SessionRepoPort` | `SessionRepository` | Persist chat history |
| `TracerPort` | `LangfuseTracer`, `NoopTracer` | Distributed tracing |
| `MetricsPort` | `PrometheusMetrics`, `NoopMetrics` | Prometheus counters/histograms |

## Agent Graph (LangGraph)

The agentic RAG loop is a LangGraph `StateGraph` with five nodes:

```
query
  │
  ▼
rewrite ──► retrieve ──► grade ──► generate ──► verify
                           │                      │
                           └──── (low grade) ─────┘
                                 retry up to N times
```

1. **rewrite** — HyDE-style query rewriting for better retrieval recall.
2. **retrieve** — hybrid dense+sparse search → RRF fusion → cross-encoder rerank.
3. **grade** — LLM-as-judge: are retrieved chunks relevant to the (rewritten) query?
4. **generate** — answer synthesis with inline citations (`[chunk_id]`).
5. **verify** — faithfulness check: does the answer stay grounded in retrieved context?

## Retrieval Scoring (RRF)

Reciprocal Rank Fusion merges dense and sparse ranked lists:

```
RRF(d) = Σ_r  1 / (k + rank_r(d))
```

where `k = 60` (default) and `r` iterates over dense and sparse rankings. Higher score = more relevant.

## DI Container

`src/infrastructure/container.py` is the composition root. It reads `Settings`, instantiates all adapters, and wires them into use cases. No adapter is instantiated outside the container.

## Citation Grounding

Each generated answer includes inline citations in the form `[chunk_id]`. The `generate` node receives the top-k reranked chunks with their IDs; the LLM prompt instructs it to cite every factual claim. The Streamlit UI renders citations as expandable source cards.
