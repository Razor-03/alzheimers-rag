"""Validated environment settings for the HTTP query service."""

import os
from dataclasses import dataclass
from pathlib import Path

from ..config import LLMConfig


@dataclass(frozen=True)
class ServiceConfig:
    llm: LLMConfig
    host: str = "localhost"
    port: int = 8080
    grpc_port: int = 50051
    collection: str = "AlzheimersChunk"
    data_dir: Path = Path("data")
    enable_reranking: bool = False

    def __post_init__(self):
        if not self.host.strip() or not self.collection.strip():
            raise ValueError("Weaviate host and collection must be nonempty")
        if not 1 <= self.port <= 65535 or not 1 <= self.grpc_port <= 65535:
            raise ValueError("Weaviate ports must be between 1 and 65535")

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        provider = env.get("RAG_LLM_PROVIDER", "ollama")
        base_url = env.get("RAG_LLM_BASE_URL")
        if base_url is None:
            if provider != "ollama":
                raise ValueError("Set RAG_LLM_BASE_URL for openai-compatible")
            base_url = "http://localhost:11434"
        reranking = env.get("RAG_ENABLE_RERANKING", "false").lower()
        if reranking not in {"true", "false", "1", "0"}:
            raise ValueError("RAG_ENABLE_RERANKING must be true, false, 1 or 0")
        return cls(
            llm=LLMConfig(
                model=env.get("RAG_LLM_MODEL", ""),
                provider=provider,
                base_url=base_url,
                api_key_env=env.get("RAG_LLM_API_KEY_ENV", "RAG_LLM_API_KEY"),
                context_window=int(env.get("RAG_CONTEXT_WINDOW", "8192")),
                context_token_budget=int(env.get("RAG_CONTEXT_TOKEN_BUDGET", "4096")),
                max_output_tokens=int(env.get("RAG_MAX_OUTPUT_TOKENS", "1024")),
                safety_tokens=int(env.get("RAG_SAFETY_TOKENS", "128")),
                timeout=float(env.get("RAG_LLM_TIMEOUT", "120")),
            ),
            host=env.get("RAG_WEAVIATE_HOST", "localhost"),
            port=int(env.get("RAG_WEAVIATE_PORT", "8080")),
            grpc_port=int(env.get("RAG_WEAVIATE_GRPC_PORT", "50051")),
            collection=env.get("RAG_WEAVIATE_COLLECTION", "AlzheimersChunk"),
            data_dir=Path(env.get("RAG_DATA_DIR", "data")),
            enable_reranking=reranking in {"true", "1"},
        )
