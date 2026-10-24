"""Settings for the whole pipeline, read from environment variables or a .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

# I load .env once here so every module sees the same settings
load_dotenv()

# project root, so relative paths work no matter where I run a script from
ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    # I treat an empty value in .env the same as not setting it
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    # accepts true/false, yes/no and 1/0 so .env is forgiving
    return _env(name, str(default)).strip().lower() in {"1", "true", "yes", "y"}


def _path(name: str, default: str) -> Path:
    # relative paths are resolved against the project root
    p = Path(_env(name, default))
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Settings:
    # llm backend choice and per-provider details
    llm_backend: str = field(default_factory=lambda: _env("LLM_BACKEND", "ollama"))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5:7b"))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-haiku-5-5"))

    # embedding and reranker models
    embed_backend: str = field(default_factory=lambda: _env("EMBED_BACKEND", "local"))
    local_embed_model: str = field(default_factory=lambda: _env("LOCAL_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    openai_embed_model: str = field(default_factory=lambda: _env("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    rerank_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "BAAI/bge-reranker-base"))

    # retrieval settings I compare in the evaluation
    retrieval_mode: str = field(default_factory=lambda: _env("RETRIEVAL_MODE", "hybrid"))
    use_reranker: bool = field(default_factory=lambda: _env_bool("USE_RERANKER", True))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    # how many candidates I pull before reranking cuts them down to top_k
    candidate_k: int = field(default_factory=lambda: int(_env("CANDIDATE_K", "30")))

    # chunking settings (size and overlap are in words)
    chunk_strategy: str = field(default_factory=lambda: _env("CHUNK_STRATEGY", "structured"))
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "220")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "40")))

    # folders
    raw_dir: Path = field(default_factory=lambda: _path("RAW_DIR", "data/raw"))
    index_dir: Path = field(default_factory=lambda: _path("INDEX_DIR", "data/index"))

    def with_overrides(self, **changes) -> "Settings":
        # the eval runner uses this to try different configs without touching .env
        return replace(self, **changes)

    def index_path(self) -> Path:
        # each chunking + embedding combo gets its own index folder so experiments don't overwrite each other
        embed_name = {"local": self.local_embed_model, "openai": self.openai_embed_model}.get(self.embed_backend, self.embed_backend)
        safe = embed_name.replace("/", "_").replace(":", "_")
        return self.index_dir / f"{self.chunk_strategy}_{self.chunk_size}_{self.chunk_overlap}_{self.embed_backend}_{safe}"


def get_settings() -> Settings:
    return Settings()
