# Evaluation

We use [Ragas](https://github.com/explodinggradients/ragas) to evaluate retrieval and
generation quality on every change. Evaluation is wired into CI as a quality gate.

## Metrics

| Metric | What it measures | Default threshold |
|--------|------------------|-------------------|
| `faithfulness` | Fraction of answer claims supported by retrieved context | ≥ 0.85 |
| `answer_relevancy` | Semantic similarity between question and answer | ≥ 0.80 |
| `context_precision` | Of retrieved chunks, how many are relevant | ≥ 0.75 |
| `context_recall` | Of relevant chunks, how many were retrieved | ≥ 0.70 |
| `answer_correctness` | Overall correctness vs ground truth | ≥ 0.70 |

Thresholds are configured in `src/config/settings.py` (`EvalSettings`) and overridable
via env vars (`RAG_EVAL__FAITHFULNESS=0.90`, …).

## Golden set

`evals/golden_set.jsonl` — 41 question/answer pairs grounded in the corpus:

- 13 English
- 13 French
- 10 Arabic
- 5 mixed-language (e.g., French question, English source)

Each row contains:

```json
{
  "query": "...",
  "ground_truth_answer": "...",
  "ground_truth_contexts": ["chunk text 1", "chunk text 2"]
}
```

Schema is enforced by `evals/schema.json` and validated by `evals/validate_dataset.py`.

## Running locally

```bash
make eval           # writes report to evals/runs/<timestamp>.json, fails on threshold breach
```

## CI gating

The CI `eval` job runs without an LLM API key (no real Ragas calls). It validates two
things:

1. Dataset schema — every row has required fields and correct types.
2. Threshold wiring — `Settings → EvalSettings → check_thresholds()` returns the
   expected pass/fail verdict for hand-crafted score fixtures (one passing, one failing).

Live Ragas runs are reserved for:

- Manual local runs via `make eval` with `RAG_LLM__API_KEY` set.
- A separate scheduled workflow (nightly) where the key is stored as a repository secret.

This keeps the public CI key-free and avoids accidental API spend on every push.
