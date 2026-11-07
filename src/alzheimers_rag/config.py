"""Explicit model and retrieval settings."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from urllib.parse import urlsplit

ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
RERANK_MODEL = "ncbi/MedCPT-Cross-Encoder"
DEFAULT_QUERY = "Alzheimer Disease AND (amyloid OR tau OR biomarker OR treatment)"


@dataclass(frozen=True)
class DataPaths:
    root: Path = Path("data")

    def processed(self, name):
        return self.root / "processed" / name

    @property
    def raw(self):
        return self.root / "raw" / "pmc_corpus.jsonl"

    @property
    def cleaned(self):
        return self.processed("pmc_alzheimers_cleaned.jsonl")

    @property
    def parents(self):
        return self.processed("pmc_alzheimers_parents.jsonl")

    @property
    def children(self):
        return self.processed("pmc_alzheimers_children.jsonl")

    @property
    def embeddings(self):
        return self.processed("child_embeddings.npy")

    @property
    def id_map(self):
        return self.processed("child_id_map.jsonl")


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    alpha: float = 0.5
    section_type_filter: str | None = None
    kwd_filter: str | None = None

    def __post_init__(self):
        if self.top_k < 1 or not 0 <= self.alpha <= 1:
            raise ValueError("top_k must be positive and alpha must be between 0 and 1")


@dataclass(frozen=True)
class RerankConfig:
    candidate_pool_size: int = 30
    rerank_top_k: int = 5
    batch_size: int = 16

    def __post_init__(self):
        if min(self.candidate_pool_size, self.rerank_top_k, self.batch_size) < 1:
            raise ValueError("Reranking sizes must be positive")


@dataclass(frozen=True)
class LLMConfig:
    model: str
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    api_key_env: str = "RAG_LLM_API_KEY"
    context_window: int = 8192
    context_token_budget: int = 4096
    max_output_tokens: int = 1024
    safety_tokens: int = 128
    timeout: float = 120.0

    def __post_init__(self):
        if not self.model.strip():
            raise ValueError("Set --model or RAG_LLM_MODEL")
        if self.provider not in {"ollama", "openai-compatible"}:
            raise ValueError("provider must be ollama or openai-compatible")
        url = urlsplit(self.base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.netloc
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "base_url must be an HTTP(S) URL without credentials, query or fragment"
            )
        if min(self.context_window, self.context_token_budget, self.max_output_tokens) < 1:
            raise ValueError("Token budgets must be positive")
        if (
            self.safety_tokens < 0
            or self.max_output_tokens + self.safety_tokens >= self.context_window
        ):
            raise ValueError("Context window must leave room for the prompt")
        if not isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be positive and finite")
