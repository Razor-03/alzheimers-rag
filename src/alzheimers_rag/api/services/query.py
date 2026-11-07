"""Process-scoped query resources; no dependency on the HTTP framework."""

from contextlib import contextmanager
from threading import Lock

from ...config import DataPaths
from ...generation import answer_question, assemble_evidence
from ...io import read_jsonl
from ...llm import HTTPChatModel


class ServiceRequestError(ValueError):
    """A valid JSON request asks for unavailable options or exceeds the input budget."""


class ServiceBusyError(RuntimeError):
    """The process is already handling a query."""


class QueryService:
    def __init__(self, config, retriever, llm, *, reranker=None, parents=None, ready_check=None):
        self.config, self.retriever, self.llm = config, retriever, llm
        self.reranker, self.parents = reranker, parents
        self._ready_check = ready_check or (lambda: True)
        self._lock = Lock()
        self._closed = False

    @property
    def capabilities(self):
        return {"reranking": self.reranker is not None, "parent_context": self.parents is not None}

    def is_ready(self):
        if self._closed:
            return False
        try:
            return bool(self._ready_check())
        except Exception:
            return False

    def close(self):
        self._closed = True

    def answer(self, query, retrieval_config, *, rerank_config=None, context="child"):
        if self._closed:
            raise RuntimeError("Query service is closed")
        if rerank_config is not None and self.reranker is None:
            raise ServiceRequestError(
                "Reranking is disabled; enable RAG_ENABLE_RERANKING at startup"
            )
        if context not in {"child", "parent"}:
            raise ServiceRequestError("Context must be child or parent")
        if context == "parent" and self.parents is None:
            raise ServiceRequestError("Parent context requires a matching parents JSONL at startup")
        # Reject oversized questions before querying a database or model.
        try:
            assemble_evidence(query, [], config=self.config.llm)
        except ValueError:
            raise ServiceRequestError("Question exceeds the configured input budget") from None
        if not self._lock.acquire(blocking=False):
            raise ServiceBusyError("Query service is busy")
        try:
            return answer_question(
                query,
                self.retriever,
                self.llm,
                config=self.config.llm,
                retrieval_config=retrieval_config,
                reranker=self.reranker if rerank_config is not None else None,
                rerank_config=rerank_config,
                parents=self.parents if context == "parent" else None,
            )
        finally:
            self._lock.release()


@contextmanager
def open_service(config):
    """Load once per worker and always close the existing-index connection."""
    from ...embeddings import MedCPTEncoder
    from ...retrieval.search import HybridRetriever
    from ...storage.weaviate import connect

    parent_path = DataPaths(config.data_dir).parents
    parents = None
    if parent_path.is_file():
        parents = {p["parent_id"]: p for p in read_jsonl(parent_path)}
    with connect(host=config.host, port=config.port, grpc_port=config.grpc_port) as client:
        if not client.collections.exists(config.collection):
            raise RuntimeError("Configured Weaviate collection does not exist; index it separately")
        retriever = HybridRetriever(
            client.collections.get(config.collection), MedCPTEncoder.for_queries()
        )
        reranker = None
        if config.enable_reranking:
            from ...retrieval.reranking import MedCPTReranker

            reranker = MedCPTReranker()
        service = QueryService(
            config,
            retriever,
            HTTPChatModel(),
            reranker=reranker,
            parents=parents,
            ready_check=client.is_ready,
        )
        try:
            yield service
        finally:
            service.close()
