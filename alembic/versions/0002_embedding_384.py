"""Resize embedding column from vector(1024) to vector(384).

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-17 00:00:00.000000
"""

from __future__ import annotations

from alembic import op

revision: str = "0002"
down_revision: str = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Resize embedding column from vector(1024) to vector(384) and rebuild index."""
    op.drop_index("chunks_embedding_hnsw_idx", table_name="chunks")
    op.execute(
        "ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(384)"
        " USING embedding::text::vector(384)"
    )
    op.create_index(
        "chunks_embedding_hnsw_idx",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    """Restore embedding column from vector(384) to vector(1024) and rebuild index."""
    op.drop_index("chunks_embedding_hnsw_idx", table_name="chunks")
    op.execute(
        "ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(1024)"
        " USING embedding::text::vector(1024)"
    )
    op.create_index(
        "chunks_embedding_hnsw_idx",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
