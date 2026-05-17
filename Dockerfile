# ─────────────────────────────────────────────────────────────────────────────
#  Stage 1 — dependency installer (uv)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Build the venv at its final runtime path so console-script shebangs
# (uvicorn, streamlit, alembic, …) resolve correctly in the runtime stage.
WORKDIR /app

# Copy only dependency manifests first (cache-friendly)
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

# ─────────────────────────────────────────────────────────────────────────────
#  Stage 2 — runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

# System deps: libpq (asyncpg native binding), wget (healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd --gid 1000 rag && useradd --uid 1000 --gid rag --no-create-home rag

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY src/ /app/src/
COPY alembic/ /app/alembic/
COPY alembic.ini /app/alembic.ini
COPY pyproject.toml /app/pyproject.toml

# Ensure .venv binaries take priority
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOME=/tmp
ENV HF_HOME=/tmp/hf_cache
ENV XDG_CACHE_HOME=/tmp/hf_cache

# Run as non-root
USER rag

# Default: FastAPI API server
# Override CMD for the web service: ["streamlit", "run", "src/interface/web/app.py", ...]
CMD ["uvicorn", "src.interface.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
