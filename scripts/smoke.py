from __future__ import annotations

import sys

import httpx

_API_BASE = "http://localhost:8000"
_QUESTIONS = [
    "What is the 1337 Coding School?",
    "How does the Piscine selection process work at 1337?",
    "What programming language is used in the 1337 curriculum?",
]


def _ask(question: str) -> dict:  # type: ignore[type-arg]
    resp = httpx.post(
        f"{_API_BASE}/chat/sync",
        json={"query": question, "session_id": "smoke-test"},
        timeout=300.0,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    failures: list[str] = []

    for q in _QUESTIONS:
        print(f"[smoke] Q: {q}")
        try:
            data = _ask(q)
        except Exception as exc:
            failures.append(f"  ✗ Request failed: {exc}")
            print(failures[-1])
            continue

        citations = data.get("citations", [])
        if not citations:
            msg = f"  ✗ No citations returned for: {q!r}"
            failures.append(msg)
            print(msg)
        else:
            print(f"  ✓ {len(citations)} citation(s) — {data.get('text', '')[:80]!r}")

    if failures:
        print(f"\n[smoke] FAILED — {len(failures)} assertion(s):")
        for f in failures:
            print(f)
        sys.exit(1)

    print("\n[smoke] All checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
