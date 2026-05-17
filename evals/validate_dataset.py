#!/usr/bin/env python3
"""
Validator for 1337 golden evaluation dataset.

Validates that each row in the JSONL file conforms to the schema and reports
language/category/difficulty distribution.

Usage:
    python evals/validate_dataset.py evals/golden_set.jsonl
"""

import json
import sys
from collections import Counter
from pathlib import Path


def load_schema(schema_path="evals/schema.json"):
    """Load JSON schema from file."""
    with open(schema_path, "r") as f:
        return json.load(f)


def validate_row(row, schema, row_index):
    """
    Validate a single row against the schema.
    
    Returns (is_valid, error_message).
    """
    required_fields = schema.get("required", [])
    
    # Check required fields
    for field in required_fields:
        if field not in row:
            return False, f"Row {row_index}: missing required field '{field}'"
    
    # Check field types and constraints
    properties = schema.get("properties", {})
    
    # id: must match pattern qa-NNN
    if "id" in row:
        id_val = row["id"]
        if not isinstance(id_val, str) or not id_val.startswith("qa-"):
            return False, f"Row {row_index}: id must be in format qa-NNN, got '{id_val}'"
    
    # query: must be string with minLength 10
    if "query" in row:
        query = row["query"]
        if not isinstance(query, str) or len(query) < 10:
            return False, f"Row {row_index}: query must be string with length >= 10"
    
    # ground_truth_answer: must be string with minLength 20
    if "ground_truth_answer" in row:
        answer = row["ground_truth_answer"]
        if not isinstance(answer, str) or len(answer) < 20:
            return False, f"Row {row_index}: ground_truth_answer must be string with length >= 20"
    
    # ground_truth_contexts: must be array of strings with minItems 1
    if "ground_truth_contexts" in row:
        contexts = row["ground_truth_contexts"]
        if not isinstance(contexts, list) or len(contexts) < 1:
            return False, f"Row {row_index}: ground_truth_contexts must be non-empty array"
        for i, ctx in enumerate(contexts):
            if not isinstance(ctx, str) or len(ctx) < 10:
                return False, f"Row {row_index}: context[{i}] must be string with length >= 10"
    
    # language: must be one of enum values
    if "language" in row:
        lang = row["language"]
        valid_langs = ["en", "fr", "ar", "mixed"]
        if lang not in valid_langs:
            return False, f"Row {row_index}: language must be one of {valid_langs}, got '{lang}'"
    
    # category: must be one of enum values
    if "category" in row:
        cat = row["category"]
        valid_cats = ["admissions", "piscine", "curriculum", "evaluations", "campus_life"]
        if cat not in valid_cats:
            return False, f"Row {row_index}: category must be one of {valid_cats}, got '{cat}'"
    
    # difficulty: must be one of enum values
    if "difficulty" in row:
        diff = row["difficulty"]
        valid_diffs = ["easy", "medium", "hard"]
        if diff not in valid_diffs:
            return False, f"Row {row_index}: difficulty must be one of {valid_diffs}, got '{diff}'"
    
    # Check no additional properties
    allowed_fields = set(properties.keys())
    for field in row.keys():
        if field not in allowed_fields:
            return False, f"Row {row_index}: unexpected field '{field}'"
    
    return True, None


def validate_dataset(jsonl_path):
    """
    Validate entire JSONL dataset.
    
    Returns (is_valid, rows, error_message).
    """
    if not Path(jsonl_path).exists():
        return False, [], f"File not found: {jsonl_path}"
    
    schema = load_schema()
    rows = []
    
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    return False, rows, f"Row {line_num}: invalid JSON: {e}"
                
                is_valid, error = validate_row(row, schema, line_num)
                if not is_valid:
                    return False, rows, error
                
                rows.append(row)
    
    except Exception as e:
        return False, rows, f"Error reading file: {e}"
    
    if len(rows) < 30:
        return False, rows, f"Dataset has {len(rows)} rows, expected at least 30"
    
    return True, rows, None


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python evals/validate_dataset.py <path_to_golden_set.jsonl>")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    
    is_valid, rows, error = validate_dataset(jsonl_path)
    
    if not is_valid:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
    
    # Compute statistics
    languages = Counter(row.get("language") for row in rows)
    categories = Counter(row.get("category") for row in rows)
    difficulties = Counter(row.get("difficulty") for row in rows)
    
    # Print success message
    print(f"OK: {len(rows)} rows, languages={dict(languages)}, categories={dict(categories)}, difficulties={dict(difficulties)}")
    sys.exit(0)


if __name__ == "__main__":
    main()
