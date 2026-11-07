"""In-process HTTP tests using the real pipeline with a mocked retriever and LLM."""

import importlib.util
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event
from unittest.mock import Mock

from alzheimers_rag.api import create_app
from alzheimers_rag.api.config import ServiceConfig
from alzheimers_rag.api.services.query import QueryService
from alzheimers_rag.config import LLMConfig

API_AVAILABLE = importlib.util.find_spec("fastapi") is not None and any(
    importlib.util.find_spec(name) for name in ("httpx2", "httpx")
)


@unittest.skipUnless(API_AVAILABLE, "requires api and dev extras")
class APITests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        self.config = ServiceConfig(llm=LLMConfig(model="offline-fixture"))
        self.retriever, self.llm = Mock(), Mock()
        self.retriever.search.return_value = [
            {
                "chunk_id": "c1",
                "parent_id": "p1",
                "doc_id": "PMC123",
                "pmid": "456",
                "doc_title": "Synthetic study",
                "text": "A synthetic finding.",
                "score": 0.5,
            }
        ]
        self.llm.complete.return_value = json.dumps(
            {
                "status": "answered",
                "statements": [{"text": "A synthetic finding.", "source_ids": ["S1"]}],
                "limitations": ["Synthetic evidence."],
            }
        )
        self.service = QueryService(self.config, self.retriever, self.llm)
        self.factory_calls = 0

        @contextmanager
        def factory(config):
            self.assertIs(config, self.config)
            self.factory_calls += 1
            try:
                yield self.service
            finally:
                self.service.close()

        self.app = create_app(config=self.config, service_factory=factory)
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_query_runs_generation_and_returns_citation_schema(self):
        result = self.client.post(
            "/query",
            json={
                "query": " What was reported? ",
                "top_k": 2,
                "alpha": 0.7,
                "section_type": "results",
                "keyword": "tau",
            },
        )
        self.assertEqual(result.status_code, 200, result.text)
        data = result.json()
        self.assertEqual(data["status"], "answered")
        self.assertEqual(data["question"], "What was reported?")
        self.assertIn("[S1]", data["answer"])
        self.assertEqual(data["citations"][0]["chunk_ids"], ["c1"])
        self.assertEqual(data["citations"][0]["parent_id"], "p1")
        self.assertEqual(data["citations"][0]["pmid"], "456")
        self.assertEqual(
            data["citations"][0]["source_url"], "https://pmc.ncbi.nlm.nih.gov/articles/PMC123/"
        )
        cfg = self.retriever.search.call_args.args[1]
        self.assertEqual(
            (cfg.top_k, cfg.alpha, cfg.section_type_filter, cfg.kwd_filter),
            (2, 0.7, "results", "tau"),
        )
        self.assertEqual(self.factory_calls, 1)

    def test_no_evidence_is_successful_insufficient_evidence_response(self):
        self.retriever.search.return_value = []
        result = self.client.post("/query", json={"query": "Q"})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "insufficient_evidence")
        self.assertEqual(result.json()["citations"], [])
        self.llm.complete.assert_not_called()

    def test_validation_rejects_bad_or_untrusted_configuration(self):
        invalid = [
            {},
            {"query": " "},
            {"query": "q" * 4097},
            {"query": 123},
            {"query": "Q", "top_k": 0},
            {"query": "Q", "top_k": 51},
            {"query": "Q", "top_k": True},
            {"query": "Q", "alpha": 1.1},
            {"query": "Q", "context": "other"},
            {"query": "Q", "keyword": " "},
            {"query": "Q", "candidate_pool_size": 201},
            {"query": "Q", "rerank": True, "top_k": 5, "candidate_pool_size": 2},
            {"query": "Q", "base_url": "http://attacker"},
            {"query": "Q", "model": "other"},
            {"query": "Q", "api_key": "secret"},
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/query", json=payload).status_code, 422)
        self.retriever.search.assert_not_called()
        self.llm.complete.assert_not_called()

    def test_disabled_features_and_input_budget_return_422(self):
        for payload in (
            {"query": "Q", "rerank": True},
            {"query": "Q", "context": "parent"},
            {"query": "α" * 4000},
        ):
            with self.subTest(payload=payload):
                result = self.client.post("/query", json=payload)
                self.assertEqual(result.status_code, 422)
        self.retriever.search.assert_not_called()

    def test_parent_context_and_reranking_use_existing_pipeline(self):
        self.service.parents = {"p1": {"text": "Full parent evidence."}}
        reranker = Mock()
        reranker.rerank.return_value = self.retriever.search.return_value
        self.service.reranker = reranker
        result = self.client.post(
            "/query",
            json={"query": "Q", "context": "parent", "rerank": True, "candidate_pool_size": 7},
        )
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["citations"][0]["context_kind"], "parent")
        self.assertEqual(result.json()["citations"][0]["text"], "Full parent evidence.")
        self.assertEqual(self.retriever.search.call_args.args[1].top_k, 7)
        reranker.rerank.assert_called_once()

    def test_bad_model_output_is_502_and_backend_failure_is_sanitized_503(self):
        self.llm.complete.return_value = self.llm.complete.return_value.replace("S1", "S999")
        result = self.client.post("/query", json={"query": "Q"})
        self.assertEqual(result.status_code, 502)
        self.assertNotIn("S999", result.text)
        self.retriever.search.side_effect = RuntimeError("private credentials and corpus text")
        with self.assertLogs("alzheimers_rag.api", level="WARNING") as logs:
            result = self.client.post("/query", json={"query": "Q"})
        self.assertEqual(result.status_code, 503)
        self.assertNotIn("private", result.text)
        self.assertNotIn("private", " ".join(logs.output))

    def test_health_readiness_and_docs_do_not_invoke_llm(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        self.assertEqual(
            self.client.get("/ready").json(),
            {"status": "ready", "reranking": False, "parent_context": False},
        )
        self.assertEqual(self.client.get("/docs").status_code, 200)
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/query", schema["paths"])
        self.assertIn("AnswerResponse", schema["components"]["schemas"])
        self.assertIn("Source", schema["components"]["schemas"])
        self.llm.complete.assert_not_called()
        self.retriever.search.assert_not_called()
        self.service._ready_check = Mock(side_effect=RuntimeError("private details"))
        self.assertEqual(self.client.get("/ready").status_code, 503)
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_busy_worker_rejects_second_query_but_health_stays_responsive(self):
        started, release = Event(), Event()
        hits = self.retriever.search.return_value

        def blocked_search(*args):
            started.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test timed out")
            return hits

        self.retriever.search.side_effect = blocked_search
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self.client.post, "/query", json={"query": "Q"})
            try:
                self.assertTrue(started.wait(timeout=5))
                result = self.client.post("/query", json={"query": "Q"})
                self.assertEqual(result.status_code, 429)
                self.assertEqual(result.headers["Retry-After"], "1")
                self.assertEqual(self.client.get("/health").status_code, 200)
            finally:
                release.set()
            self.assertEqual(pending.result(timeout=5).status_code, 200)

    def test_lifespan_owns_service_cleanup(self):
        self.assertFalse(self.service._closed)
        self.client.__exit__(None, None, None)
        self.assertTrue(self.service._closed)
        self.assertIsNone(self.app.state.service)
        self.assertEqual(self.factory_calls, 1)

    def test_public_dependency_can_be_overridden(self):
        from alzheimers_rag.api.dependencies import get_service

        retriever = Mock()
        retriever.search.return_value = []
        replacement = QueryService(self.config, retriever, Mock())
        self.app.dependency_overrides[get_service] = lambda: replacement
        try:
            result = self.client.post("/query", json={"query": "Q"})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()["status"], "insufficient_evidence")
            self.retriever.search.assert_not_called()
        finally:
            self.app.dependency_overrides.clear()
            replacement.close()

    def test_app_instances_keep_resources_and_overrides_separate(self):
        from fastapi.testclient import TestClient

        from alzheimers_rag.api.dependencies import get_service
        from alzheimers_rag.api.main import create_app as app_factory

        retriever = Mock()
        retriever.search.return_value = []
        service = QueryService(self.config, retriever, Mock())

        @contextmanager
        def factory(config):
            try:
                yield service
            finally:
                service.close()

        app = app_factory(config=self.config, service_factory=factory)
        with TestClient(app) as other_client:
            app.dependency_overrides[get_service] = lambda: service
            result = other_client.post("/query", json={"query": "Q"})
            self.assertEqual(result.json()["status"], "insufficient_evidence")
            result = self.client.post("/query", json={"query": "Q"})
            self.assertEqual(result.json()["status"], "answered")
            self.assertEqual(self.app.dependency_overrides, {})
        self.assertTrue(service._closed)
        self.assertFalse(self.service._closed)
