# Observability

## Langfuse Tracing

Every request through the agentic RAG pipeline is traced end-to-end in Langfuse.

### Setup

```bash
# In .env
RAG_LANGFUSE__ENABLED=true
RAG_LANGFUSE__HOST=http://localhost:3000
RAG_LANGFUSE__PUBLIC_KEY=<your-public-key>
RAG_LANGFUSE__SECRET_KEY=<your-secret-key>
```

### Span Tree

Each request produces a trace with the following span hierarchy:

```
answer (root)
├── rewrite          — query rewriting; input: raw query; output: rewritten query
├── retrieve         — hybrid search + rerank; output: top-k chunks with scores
│   ├── dense_search
│   ├── sparse_search
│   ├── rrf_fusion
│   └── rerank
├── grade            — relevance grading; output: pass/fail per chunk
├── generate         — answer synthesis; output: answer + citations
└── verify           — faithfulness check; output: pass/fail
```

Every span records: input tokens, output tokens, model name, latency, and cost estimate.

## Prometheus Metrics

The API exposes `/metrics` (default port 9090). All metrics use the `rag_` prefix.

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `rag_requests_total` | Counter | `endpoint`, `status` | Total HTTP requests |
| `rag_request_duration_seconds` | Histogram | `endpoint` | Request latency |
| `rag_llm_tokens_total` | Counter | `provider`, `model`, `type` | Token usage (input/output) |
| `rag_llm_cost_usd_total` | Counter | `provider`, `model` | Estimated LLM cost |
| `rag_retrieval_chunks_returned` | Histogram | `stage` | Chunks at each retrieval stage |
| `rag_agent_iterations_total` | Histogram | — | Agent loop iterations per request |
| `rag_eval_score` | Gauge | `metric`, `language` | Latest Ragas eval scores |
| `rag_ingest_documents_total` | Counter | `status` | Documents ingested |

## Grafana Dashboards

Three dashboards are provisioned automatically at `http://localhost:3001`:

### RAG Operations (`grafana/dashboards/rag-ops.json`)

- Request rate and error rate (5m window)
- P50/P95/P99 latency by endpoint
- Agent iteration distribution
- Retrieval chunk counts at each stage

### LLM Costs (`grafana/dashboards/llm-costs.json`)

- Token usage rate by model
- Cumulative cost by provider
- Cost per request (rolling average)
- Token efficiency (output/input ratio)

### Eval Trends (`grafana/dashboards/eval-trends.json`)

- Faithfulness score over time by language
- Answer relevancy trend
- Context precision trend
- Threshold breach alerts

## Loki Log Queries

Structured JSON logs are shipped to Loki. Useful queries:

```logql
# All errors in the last hour
{app="rag-assistant"} | json | level="error"

# Slow requests (>5s)
{app="rag-assistant"} | json | duration_ms > 5000

# Agent retries
{app="rag-assistant"} | json | event="agent_retry"

# Retrieval grade failures
{app="rag-assistant"} | json | event="grade_fail"
```

## Alert Runbook

### `RagHighErrorRate`

**Condition**: `rate(rag_requests_total{status="5xx"}[5m]) > 0.05`

**Action**:
1. Check `docker compose logs api` for stack traces.
2. Verify Postgres connectivity: `make smoke`.
3. Check LLM API key validity and quota.

### `RagHighLatency`

**Condition**: `histogram_quantile(0.95, rag_request_duration_seconds) > 10`

**Action**:
1. Check reranker model load time (first request after cold start is slow).
2. Verify pgvector HNSW index exists: `SELECT * FROM pg_indexes WHERE tablename='chunks'`.
3. Check LLM provider latency via Langfuse trace waterfall.

### `RagEvalScoreBelowThreshold`

**Condition**: `rag_eval_score{metric="faithfulness"} < 0.85`

**Action**:
1. Run `make eval` locally to reproduce.
2. Inspect failing rows in Langfuse (filter by low faithfulness score).
3. Check if knowledge base was updated without re-ingestion.
