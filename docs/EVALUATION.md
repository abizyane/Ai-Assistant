# Evaluation

## Golden Dataset

**File**: `evals/golden_set.jsonl`  
**Size**: 38 QA pairs  
**Schema**: `evals/schema.json`

### Composition

| Language | Count | Rationale |
|----------|-------|-----------|
| English (`en`) | 10 | Primary documentation language |
| French (`fr`) | 10 | Official language of Morocco; majority of 1337 students |
| Arabic (`ar`) | 7 | Native language; accessibility requirement |
| Mixed (`mixed`) | 11 | Realistic code-switching patterns |

| Difficulty | Count | Description |
|------------|-------|-------------|
| Easy (40%) | ~15 | Single-chunk lookup; direct answer in one passage |
| Medium (40%) | ~15 | Multi-chunk synthesis; answer requires combining passages |
| Hard (20%) | ~8 | Reasoning/comparison; inference across domains |

| Category | Count |
|----------|-------|
| admissions | 6 |
| piscine | 6 |
| curriculum | 5 |
| evaluations | 8 |
| campus_life | 7 |
| mixed | 6 |

## Ragas Metrics

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Faithfulness** | fraction of answer claims supported by retrieved context | Hallucination rate |
| **Answer Relevancy** | cosine similarity of generated answer to original question | Answer focus |
| **Context Precision** | fraction of retrieved chunks that are relevant | Retrieval precision |
| **Context Recall** | fraction of ground-truth contexts that were retrieved | Retrieval recall |

## Threshold Rationale

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| Faithfulness | ≥ 0.85 | High bar: factual errors about school policies are harmful |
| Answer Relevancy | ≥ 0.80 | Answers must stay on-topic; off-topic responses frustrate users |
| Context Precision | ≥ 0.75 | Noisy context degrades generation quality |
| Context Recall | ≥ 0.70 | Missing context causes incomplete answers |

## Running Evaluation

```bash
# Full evaluation against golden set (requires LLM API key)
make eval

# Validate dataset schema only (no API calls — used in CI)
python evals/validate_dataset.py evals/golden_set.jsonl
```

## CI Gating

The `eval` job in `.github/workflows/ci.yml` validates the smoke dataset schema (`tests/fixtures/eval_smoke.jsonl`) on every push. Live Ragas evaluation (which requires an LLM API key) is not run in CI to avoid API costs; it is run manually before releases.

## Adding New QA Pairs

1. Identify a question that is grounded in `data/knowledge_base/` source documents.
2. Extract the exact substrings from the source that support the answer as `ground_truth_contexts`.
3. Assign `language`, `category`, and `difficulty`.
4. Increment the `id` field (e.g., `qa-039`).
5. Append the JSON line to `evals/golden_set.jsonl`.
6. Run the validator: `python evals/validate_dataset.py evals/golden_set.jsonl`.

Example:

```json
{
  "id": "qa-039",
  "query": "What is the maximum number of project retries at 1337?",
  "ground_truth_answer": "Each project can be attempted up to three times.",
  "ground_truth_contexts": ["Each project can be attempted up to three times"],
  "language": "en",
  "category": "evaluations",
  "difficulty": "easy"
}
```
