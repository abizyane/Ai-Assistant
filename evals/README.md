# Golden Evaluation Dataset for 1337 Coding School

## Overview

This directory contains the golden evaluation dataset for the 1337 Coding School RAG assistant. The dataset consists of 32 high-quality question-answer pairs covering the school's core domains: admissions, piscine, curriculum, evaluations, and campus life.

## Dataset Composition

### Language Distribution

- **English (EN)**: 10 pairs
- **French (FR)**: 10 pairs
- **Arabic (AR)**: 5 pairs
- **Mixed/Code-switched**: 7 pairs

Total: 32 QA pairs

### Category Distribution

Each category has at least 4 QA pairs:

- **Admissions**: 6 pairs (eligibility, application process, selection criteria)
- **Piscine**: 6 pairs (duration, structure, programming focus, evaluation)
- **Curriculum**: 5 pairs (project-based learning, specializations, progression)
- **Evaluations**: 8 pairs (peer-correction, grading, retry rules, gamification)
- **Campus Life**: 7 pairs (locations, 24/7 access, events, partnerships, support)

### Difficulty Distribution

- **Easy** (40%): 13 pairs - Single-chunk lookup questions requiring direct retrieval from knowledge base
- **Medium** (40%): 13 pairs - Multi-chunk synthesis questions requiring combination of information
- **Hard** (20%): 6 pairs - Reasoning and comparison questions requiring inference

## Curation Methodology

### Source Documents

All questions and answers are grounded in the placeholder knowledge base files:

- `rag/knowledge_base/01_admissions.md` - Admissions process, eligibility, selection stages
- `rag/knowledge_base/02_curriculum.md` - Project-based learning, specializations, progression
- `rag/knowledge_base/03_piscine.md` - 26-day bootcamp structure, daily activities, evaluation
- `rag/knowledge_base/04_evaluations.md` - Peer-correction, grading, retry rules, gamification
- `rag/knowledge_base/05_campus_life.md` - Campus locations, 24/7 access, events, partnerships

### Question Design

Questions were designed to:

1. Cover core student concerns (admissions, piscine, curriculum progression)
2. Test both factual recall and conceptual understanding
3. Represent realistic user queries in multiple languages
4. Span difficulty levels to evaluate RAG performance across retrieval complexity

### Language Coverage Rationale

- **English**: Primary language for international students and documentation
- **French**: Official language of Morocco and many 1337 students
- **Arabic**: Native language of Morocco, important for accessibility
- **Mixed**: Realistic code-switching patterns (e.g., "C'est quoi le Piscine?", "كيف تنجح في الـ piscine؟")

### Difficulty Rationale

- **Easy** (40%): Validates basic retrieval capability; questions with direct answers in single chunks
- **Medium** (40%): Tests multi-chunk retrieval and synthesis; answers requiring combination of concepts
- **Hard** (20%): Evaluates reasoning; questions requiring inference, comparison, or cross-domain understanding

## Schema

Each row in `golden_set.jsonl` follows this schema (see `schema.json` for JSON Schema):

```json
{
  "id": "qa-001",
  "query": "What is the age requirement for applying to 1337 Coding School?",
  "ground_truth_answer": "The typical age requirement for 1337 is 18 to 30 years old...",
  "ground_truth_contexts": [
    "Age: typically 18 to 30, but variations exist.",
    "Motivation, perseverance, and team spirit emphasized over diplomas."
  ],
  "language": "en",
  "category": "admissions",
  "difficulty": "easy"
}
```

### Field Descriptions

- **id**: Unique identifier in format `qa-NNN` (sequential)
- **query**: The question in the specified language
- **ground_truth_answer**: Expected answer based on knowledge base content
- **ground_truth_contexts**: List of substrings from knowledge base files supporting the answer
- **language**: One of `en`, `fr`, `ar`, `mixed`
- **category**: One of `admissions`, `piscine`, `curriculum`, `evaluations`, `campus_life`
- **difficulty**: One of `easy`, `medium`, `hard`

## Validation

Run the validator to check dataset integrity:

```bash
python evals/validate_dataset.py evals/golden_set.jsonl
```

Expected output:
```
OK: 32 rows, languages={'en': 10, 'fr': 10, 'ar': 5, 'mixed': 7}, categories={...}, difficulties={...}
```

## Adding New QA Pairs

To add new questions to the dataset:

1. Ensure the answer is grounded in one of the placeholder knowledge base files
2. Extract relevant substrings as `ground_truth_contexts`
3. Assign appropriate language, category, and difficulty
4. Increment the `id` field (e.g., `qa-033` for the next pair)
5. Append the JSON line to `golden_set.jsonl`
6. Run the validator to confirm

Example:

```json
{"id": "qa-033", "query": "...", "ground_truth_answer": "...", "ground_truth_contexts": [...], "language": "en", "category": "curriculum", "difficulty": "medium"}
```

## Smoke Testing

A 5-row smoke fixture is provided in `tests/fixtures/eval_smoke.jsonl` for quick validation during development. It includes:

- 1 English easy (piscine)
- 1 French medium (admissions)
- 1 Arabic easy (admissions)
- 1 English medium (evaluations)
- 1 Mixed medium (admissions)

## Future Enhancements

- Expand to 100+ pairs covering edge cases and advanced topics
- Add multi-turn conversation examples
- Include common misconceptions and clarifications
- Integrate with automated evaluation pipeline (F3 territory)
