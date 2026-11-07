"""Validated research-query input and structured answer output."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=4096)
    top_k: int = Field(default=5, ge=1, le=50)
    alpha: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    section_type: str | None = Field(default=None, min_length=1, max_length=100)
    keyword: str | None = Field(default=None, min_length=1, max_length=200)
    rerank: bool = False
    candidate_pool_size: int = Field(default=30, ge=1, le=200)
    context: Literal["child", "parent"] = "child"

    @model_validator(mode="after")
    def validate_pool(self):
        if self.rerank and self.candidate_pool_size < self.top_k:
            raise ValueError("candidate_pool_size must be at least top_k when reranking")
        return self


class Statement(BaseModel):
    text: str
    source_ids: list[str]


class Source(BaseModel):
    source_id: str
    chunk_ids: list[str]
    parent_id: str | None
    doc_id: str | None
    title: str
    pmid: str | None
    pmcid: str | None
    doi: str | None
    source_url: str | None
    heading_path: list[str] | str
    context_kind: Literal["child", "parent"]
    parent_missing: bool
    text: str


class Budget(BaseModel):
    counter: str
    context_tokens: int
    input_tokens: int
    input_limit: int
    context_token_budget: int
    reserved_output_tokens: int
    safety_tokens: int
    skipped_candidates: int


class Generation(BaseModel):
    provider: Literal["ollama", "openai-compatible"]
    model: str


class AnswerResponse(BaseModel):
    question: str
    status: Literal["answered", "insufficient_evidence"]
    statements: list[Statement]
    limitations: list[str]
    answer: str
    citations: list[Source]
    evidence: list[Source]
    budget: Budget
    generation: Generation
