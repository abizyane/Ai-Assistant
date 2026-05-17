# Observability

Every request is traceable, every component is measured, every log line is queryable.

## Tracing — Langfuse

Langfuse is the system of record for LLM interactions. Each query opens a trace that
spans the full agentic loop:

- root span: `answer_query` with input + output
- child spans: `retrieve`, `rerank`, `generate`, `validate`
- LLM generations recorded with model, prompt, completion, token counts, cost

Configuration:

```env
RAG_LANGFUSE__HOST=http://localhost:3000
RAG_LANGFUSE__PUBLIC_KEY=...
RAG_LANGFUSE__SECRET_KEY=...
RAG_LANGFUSE__ENABLED=true
```

Open the UI: http://localhost:3000 (after `make demo`).

The Langfuse adapter implements `TracerPort` (`src/infrastructure/observability/`). To
disable in tests, swap to a no-op adapter in the DI container.

## Metrics — Prometheus

Prometheus scrapes `app:8000/metrics`. Custom metrics:

| Metric | Type | Labels |
|--------|------|--------|
| `rag_query_seconds` | Histogram | `route`, `status` |
| `rag_tokens_total` | Counter | `provider`, `model`, `direction` |
| `rag_retrieval_hits` | Counter | `mode` (`dense`/`sparse`/`fused`) |
| `rag_rerank_seconds` | Histogram | — |
| `rag_eval_score` | Gauge | `metric` |

Scrape config: `infra/prometheus/prometheus.yml`.
Alert rules: `infra/prometheus/rules/`.

Open the UI: http://localhost:9090.

## Dashboards — Grafana

Three pre-provisioned dashboards live under `infra/grafana/dashboards/`:

- **RAG ops** — query latency, error rate, retrieval hits per mode
- **LLM costs** — token rate by provider/model, cumulative cost
- **Eval trends** — last 30 days of Ragas scores per metric

Provisioning: `infra/grafana/provisioning/` (data sources + dashboard loader).

Open the UI: http://localhost:3001 (default `admin/admin`).

## Logs — Loki

`structlog` emits JSON log lines from the app. Loki ingests them with labels
(`service`, `level`, `trace_id`). Grafana's Explore tab can correlate logs with
Prometheus metrics by `trace_id` for full request-level forensics.

Loki config: `infra/loki/loki-config.yml`.

## Alerting

Prometheus rules in `infra/prometheus/rules/` define alerts for:

- p99 query latency > 5s for 10 minutes
- error rate > 5% for 5 minutes
- faithfulness score < 0.80 (set from `rag_eval_score{metric="faithfulness"}`)

Alertmanager is not bundled — wire your preferred receiver (Slack, PagerDuty, email)
to Prometheus' `--alertmanager.url` flag in production.
