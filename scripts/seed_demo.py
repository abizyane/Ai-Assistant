from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "RAG_VECTORSTORE__DATABASE_URL", "postgresql+asyncpg://raguser:ragpassword@localhost:5432/ragdb"
)


async def _count_existing_chunks(engine) -> int:  # type: ignore[no-untyped-def]
    from sqlalchemy import text

    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT COUNT(*) FROM chunks"))
        row = result.fetchone()
        return int(row[0]) if row else 0


async def main() -> None:
    from src.config.settings import Settings
    from src.infrastructure import di

    settings = Settings()
    engine = di.build_engine(settings)

    try:
        existing = await _count_existing_chunks(engine)
    except Exception:
        existing = 0

    if existing > 0:
        print(f"[seed_demo] Vector store already has {existing} chunks — skipping ingestion.")
        await engine.dispose()
        return

    kb_path = ROOT / "data" / "knowledge_base"
    if not kb_path.exists():
        print(f"[seed_demo] No knowledge base found at {kb_path}. Drop documents there first.")
        await engine.dispose()
        return

    # Collect documents with supported suffixes: .pdf, .md, .html
    supported_suffixes = [".pdf", ".md", ".html"]
    documents = []
    for suffix in supported_suffixes:
        documents.extend(kb_path.glob(f"**/*{suffix}"))

    if not documents:
        print(
            f"[seed_demo] No documents found in {kb_path}. Drop PDFs, Markdown, or HTML files there first."
        )
        await engine.dispose()
        return

    print(f"[seed_demo] Found {len(documents)} document(s). Starting ingestion…")

    use_case = di.build_ingest_use_case(settings)
    report = await use_case.execute(path=kb_path)

    print(
        f"[seed_demo] Ingestion complete: {report.files_processed} docs, "
        f"{report.chunks_created} chunks, {len(report.errors)} errors."
    )
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
