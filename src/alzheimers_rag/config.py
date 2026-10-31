"""Explicit model and retrieval settings."""

from dataclasses import dataclass
from pathlib import Path

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
