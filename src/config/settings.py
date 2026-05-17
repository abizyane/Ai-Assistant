"""Configuration module — pydantic-settings, .env loading, get_settings()."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_LLM__", env_file=".env", extra="ignore"
    )

    provider: str = Field(
        default="gemini",
        description="LLM provider: gemini | openai | openrouter | nvidia | anthropic | ollama",
    )
    model: str = Field(
        default="gemini-2.0-flash", description="Model name/ID"
    )
    api_key: SecretStr = Field(description="API key for the LLM provider")
    base_url: str | None = Field(
        default=None,
        description=(
            "Optional override for OpenAI-compatible endpoints "
            "(OpenRouter, NVIDIA NIM, local servers)"
        ),
    )
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int = Field(default=4096, gt=0, description="Maximum output tokens")
    max_retries: int = Field(
        default=3, ge=0, description="Max retry attempts on transient LLM errors"
    )


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_EMBEDDING__", env_file=".env", extra="ignore"
    )

    model: str = Field(
        default="BAAI/bge-m3", description="HuggingFace embedding model ID"
    )
    batch_size: int = Field(default=32, gt=0, description="Batch size for encoding")
    cache_dir: str = Field(
        default="/tmp/hf_cache", description="HuggingFace model cache directory"
    )


class VectorStoreSettings(BaseSettings):
    """Vector store (pgvector) configuration."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_VECTORSTORE__", env_file=".env", extra="ignore"
    )

    database_url: str = Field(
        default="postgresql+asyncpg://raguser:ragpassword@localhost:5432/ragdb",
        description="Async PostgreSQL connection URL",
    )
    table_name: str = Field(default="chunks", description="pgvector table name")
    hnsw_m: int = Field(default=16, gt=0, description="HNSW M parameter")
    hnsw_ef_construction: int = Field(
        default=64, gt=0, description="HNSW ef_construction parameter"
    )
    top_k_dense: int = Field(default=20, gt=0, description="Dense retrieval top-k")
    top_k_sparse: int = Field(
        default=20, gt=0, description="Sparse (BM25) retrieval top-k"
    )
    top_k_rerank: int = Field(
        default=5, gt=0, description="After-rerank top-k returned"
    )


class RerankerSettings(BaseSettings):
    """Cross-encoder reranker configuration."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_RERANKER__", env_file=".env", extra="ignore"
    )

    model: str = Field(
        default="BAAI/bge-reranker-v2-m3", description="Reranker model ID"
    )
    batch_size: int = Field(default=16, gt=0, description="Batch size for reranking")


class LangfuseSettings(BaseSettings):
    """Langfuse observability tracing configuration."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_LANGFUSE__", env_file=".env", extra="ignore"
    )

    host: str = Field(default="http://localhost:3000", description="Langfuse server URL")
    public_key: str = Field(default="", description="Langfuse public key")
    secret_key: SecretStr = Field(
        default=SecretStr(""), description="Langfuse secret key"
    )
    enabled: bool = Field(default=False, description="Enable Langfuse tracing")


class PrometheusSettings(BaseSettings):
    """Prometheus metrics configuration."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_PROMETHEUS__", env_file=".env", extra="ignore"
    )

    port: int = Field(default=9090, gt=0, description="Prometheus scrape port")
    path: str = Field(default="/metrics", description="Metrics endpoint path")


class EvalSettings(BaseSettings):
    """Ragas evaluation quality thresholds."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_EVAL__", env_file=".env", extra="ignore"
    )

    faithfulness: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Minimum faithfulness score"
    )
    answer_relevancy: float = Field(
        default=0.80, ge=0.0, le=1.0, description="Minimum answer relevancy score"
    )
    context_precision: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Minimum context precision score"
    )
    context_recall: float = Field(
        default=0.70, ge=0.0, le=1.0, description="Minimum context recall score"
    )
    answer_correctness: float = Field(
        default=0.70, ge=0.0, le=1.0, description="Minimum answer correctness score"
    )


class AgentSettings(BaseSettings):
    """LangGraph agent loop configuration."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_AGENT__", env_file=".env", extra="ignore"
    )

    max_rewrite_attempts: int = Field(
        default=1, ge=0, le=3, description="Max query rewrite iterations"
    )
    max_regen_attempts: int = Field(
        default=1, ge=0, le=3, description="Max answer regeneration retries"
    )
    max_steps: int = Field(default=15, gt=0, description="Max LangGraph steps before force-halt")


class ChunkingSettings(BaseSettings):
    """Document chunking configuration."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_CHUNKING__", env_file=".env", extra="ignore"
    )

    chunk_size: int = Field(default=800, gt=0, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=100, ge=0, description="Overlap between consecutive chunks")


class Settings(BaseSettings):
    """Root settings — aggregates all sub-settings with RAG_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    prometheus: PrometheusSettings = Field(default_factory=PrometheusSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    eval: EvalSettings = Field(default_factory=EvalSettings)
    api_url: str = Field(
        default="http://localhost:8000",
        description="RAG API base URL used by the Streamlit web UI",
    )
    langfuse_base_url: str = Field(
        default="http://localhost:3000",
        description="Langfuse UI base URL for trace deep-links in the web UI",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return singleton Settings instance, loaded from .env."""
    return Settings()
