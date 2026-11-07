"""Shared query-service lifecycle tests using injected offline resources."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from alzheimers_rag.api.config import ServiceConfig
from alzheimers_rag.api.services.query import (
    QueryService,
    ServiceRequestError,
    open_service,
)
from alzheimers_rag.config import LLMConfig, RerankConfig, RetrievalConfig
from alzheimers_rag.generation import GenerationError
from alzheimers_rag.io import write_jsonl


def sample_hit():
    return {
        "chunk_id": "c1",
        "parent_id": "p1",
        "doc_id": "PMC123",
        "pmid": "456",
        "doc_title": "Synthetic study",
        "text": "A synthetic finding.",
        "score": 0.5,
    }


def model_response():
    return json.dumps(
        {
            "status": "answered",
            "statements": [{"text": "A synthetic finding.", "source_ids": ["S1"]}],
            "limitations": [],
        }
    )


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.config = ServiceConfig(llm=LLMConfig(model="offline"))
        self.retriever, self.llm = Mock(), Mock()
        self.retriever.search.return_value = [sample_hit()]
        self.llm.complete.return_value = model_response()
        self.service = QueryService(self.config, self.retriever, self.llm)

    def test_legacy_service_imports_reference_the_same_implementation(self):
        from alzheimers_rag import service

        self.assertIs(service.QueryService, QueryService)
        self.assertIs(service.ServiceConfig, ServiceConfig)
        self.assertIs(service.open_service, open_service)

    def test_settings_use_explicit_server_environment(self):
        config = ServiceConfig.from_env(
            {
                "RAG_LLM_MODEL": "configured",
                "RAG_LLM_PROVIDER": "openai-compatible",
                "RAG_LLM_BASE_URL": "http://localhost:9000/v1",
                "RAG_WEAVIATE_HOST": "db",
                "RAG_WEAVIATE_PORT": "8081",
                "RAG_WEAVIATE_GRPC_PORT": "50052",
                "RAG_WEAVIATE_COLLECTION": "ExistingResearch",
                "RAG_DATA_DIR": "corpus",
                "RAG_ENABLE_RERANKING": "true",
                "RAG_CONTEXT_WINDOW": "16000",
                "RAG_CONTEXT_TOKEN_BUDGET": "6000",
                "RAG_MAX_OUTPUT_TOKENS": "2000",
                "RAG_SAFETY_TOKENS": "200",
                "RAG_LLM_TIMEOUT": "60",
                "RAG_LLM_API_KEY_ENV": "CUSTOM_KEY",
            }
        )
        self.assertEqual(config.llm.provider, "openai-compatible")
        self.assertEqual(config.llm.model, "configured")
        self.assertEqual(config.llm.context_window, 16000)
        self.assertEqual(config.llm.context_token_budget, 6000)
        self.assertEqual(config.llm.max_output_tokens, 2000)
        self.assertEqual(config.llm.safety_tokens, 200)
        self.assertEqual(config.llm.timeout, 60)
        self.assertEqual(config.llm.api_key_env, "CUSTOM_KEY")
        self.assertEqual((config.host, config.port, config.grpc_port), ("db", 8081, 50052))
        self.assertEqual(config.collection, "ExistingResearch")
        self.assertEqual(config.data_dir, Path("corpus"))
        self.assertTrue(config.enable_reranking)

    def test_invalid_startup_settings_fail_early(self):
        for values in (
            {},
            {"RAG_ENABLE_RERANKING": "maybe"},
            {"RAG_WEAVIATE_PORT": "0"},
            {"RAG_WEAVIATE_PORT": "invalid"},
            {"RAG_WEAVIATE_COLLECTION": " "},
            {"RAG_LLM_PROVIDER": "openai-compatible"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ServiceConfig.from_env({"RAG_LLM_MODEL": "" if not values else "offline", **values})

    def test_unsupported_options_and_large_question_never_retrieve(self):
        for kwargs in (
            {"context": "parent"},
            {"context": "other"},
            {"rerank_config": RerankConfig()},
        ):
            with self.assertRaises(ServiceRequestError):
                self.service.answer("Q", RetrievalConfig(), **kwargs)
        with self.assertRaises(ServiceRequestError):
            self.service.answer("α" * 4000, RetrievalConfig())
        self.retriever.search.assert_not_called()
        self.llm.complete.assert_not_called()

    def test_reranker_is_used_only_when_requested(self):
        reranker = Mock()
        reranker.rerank.return_value = [sample_hit()]
        self.service.reranker = reranker
        self.service.answer("Q", RetrievalConfig())
        reranker.rerank.assert_not_called()
        self.service.answer("Q", RetrievalConfig(), rerank_config=RerankConfig())
        reranker.rerank.assert_called_once()
        self.assertEqual(self.retriever.search.call_args.args[1].top_k, 30)

    def test_query_failure_releases_capacity(self):
        self.llm.complete.return_value = "invalid JSON"
        with self.assertRaises(GenerationError):
            self.service.answer("Q", RetrievalConfig())
        self.llm.complete.return_value = model_response()
        self.assertEqual(self.service.answer("Q", RetrievalConfig())["status"], "answered")

    def test_readiness_tracks_database_and_closed_state(self):
        check = Mock(return_value=True)
        self.service._ready_check = check
        self.assertTrue(self.service.is_ready())
        check.return_value = False
        self.assertFalse(self.service.is_ready())
        check.side_effect = RuntimeError("backend details")
        self.assertFalse(self.service.is_ready())
        self.service.close()
        self.assertFalse(self.service.is_ready())
        with self.assertRaises(RuntimeError):
            self.service.answer("Q", RetrievalConfig())

    def test_resources_loaded_once_and_connection_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(self.config, data_dir=Path(directory), enable_reranking=True)
            write_jsonl(
                [{"parent_id": "p1", "text": "Full parent finding."}],
                config.data_dir / "processed" / "pmc_alzheimers_parents.jsonl",
            )
            with (
                patch("alzheimers_rag.storage.weaviate.connect") as connect,
                patch("alzheimers_rag.embeddings.MedCPTEncoder.for_queries") as encoder,
                patch("alzheimers_rag.retrieval.reranking.MedCPTReranker") as reranker,
                patch(
                    "alzheimers_rag.retrieval.search.HybridRetriever.search",
                    return_value=[sample_hit()],
                ),
                patch("alzheimers_rag.llm.HTTPChatModel.complete", return_value=model_response()),
            ):
                client = connect.return_value.__enter__.return_value
                with open_service(config) as service:
                    self.assertTrue(service.capabilities["parent_context"])
                    self.assertTrue(service.capabilities["reranking"])
                    for _ in range(2):
                        result = service.answer("Q", RetrievalConfig(), context="parent")
                        self.assertEqual(result["citations"][0]["text"], "Full parent finding.")
                    encoder.assert_called_once()
                    reranker.assert_called_once()
                    connect.assert_called_once()
                    client.collections.create.assert_not_called()
                self.assertFalse(service.is_ready())
                connect.return_value.__exit__.assert_called_once()

    def test_startup_failure_closes_database_without_creating_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("alzheimers_rag.storage.weaviate.connect") as connect,
                patch("alzheimers_rag.embeddings.MedCPTEncoder.for_queries") as encoder,
            ):
                client = connect.return_value.__enter__.return_value
                client.collections.exists.return_value = False
                with self.assertRaisesRegex(RuntimeError, "does not exist"):
                    with open_service(replace(self.config, data_dir=Path(directory))):
                        self.fail("must not start with a missing collection")
                encoder.assert_not_called()
                client.collections.create.assert_not_called()
                connect.return_value.__exit__.assert_called_once()
                connect.return_value.__exit__.reset_mock()
                client.collections.exists.return_value = True
                encoder.side_effect = RuntimeError("model load failed")
                with self.assertRaisesRegex(RuntimeError, "model load failed"):
                    with open_service(replace(self.config, data_dir=Path(directory))):
                        self.fail("must not start without the encoder")
                connect.return_value.__exit__.assert_called_once()
